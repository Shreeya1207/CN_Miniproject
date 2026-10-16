"""Unit tests for topology, routing, congestion, failure, and recalculation.

Covers:
- Topology creation and attribute verification
- Dijkstra hop-count routing correctness
- Dijkstra minimum-cost routing correctness
- Effective cost formula: base_cost * (1 + (util/100)^2)
- Alternative paths enumeration
- Failed link exclusion
- Congestion-triggered effective cost increase and rerouting
- Dynamic cost change rerouting
- Per-link utilization adjustment (new)
- Per-link failure and restoration (new)
- No available route / unreachable destination (new)
- Source == destination edge case
- Automatic rerouting after link failure via controller
- Congestion simulation triggers reroute
- Restore and reset network
- SEVERE congestion tier (new)
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from controller import SDNController
from monitor import NetworkMonitor
from routing import RoutingEngine
from simulation import NetworkSimulator
from topology import NetworkTopology

CONFIG = Path(__file__).resolve().parents[1] / "data" / "network_config.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def topology() -> NetworkTopology:
    return NetworkTopology(CONFIG)


@pytest.fixture
def engine(topology: NetworkTopology) -> RoutingEngine:
    return RoutingEngine(topology)


@pytest.fixture
def controller(topology: NetworkTopology) -> SDNController:
    return SDNController(topology)


@pytest.fixture
def simulator(topology: NetworkTopology) -> NetworkSimulator:
    return NetworkSimulator(topology)


# ---------------------------------------------------------------------------
# Topology creation
# ---------------------------------------------------------------------------

def test_topology_creation(topology: NetworkTopology) -> None:
    assert topology.graph.number_of_nodes() >= 6
    assert topology.graph.number_of_edges() >= 8
    for node in ("S1", "S2", "S3", "S4", "S5", "S6"):
        assert topology.has_node(node)
    info = topology.get_link_info("S1", "S2")
    assert info["status"] == "UP"
    assert info["capacity"] > 0
    assert 0 <= info["utilization"] <= 100
    assert info["latency"] > 0


# ---------------------------------------------------------------------------
# Effective cost formula
# ---------------------------------------------------------------------------

def test_effective_cost_formula(topology: NetworkTopology) -> None:
    """effective_cost = base_cost * (1 + (util/100)^2)."""
    topology.update_link_utilization("S1", "S2", 50.0)
    info = topology.get_link_info("S1", "S2")
    base = info["cost"]
    expected = base * (1.0 + (50.0 / 100.0) ** 2)
    assert abs(info["effective_cost"] - expected) < 1e-9, (
        f"effective_cost mismatch: got {info['effective_cost']}, expected {expected}"
    )


def test_effective_cost_zero_utilization(topology: NetworkTopology) -> None:
    """At 0% util, effective_cost == base_cost."""
    topology.update_link_utilization("S1", "S2", 0.0)
    info = topology.get_link_info("S1", "S2")
    assert abs(info["effective_cost"] - info["cost"]) < 1e-9


def test_effective_cost_full_utilization(topology: NetworkTopology) -> None:
    """At 100% util, effective_cost == 2 * base_cost."""
    topology.update_link_utilization("S1", "S2", 100.0)
    info = topology.get_link_info("S1", "S2")
    expected = info["cost"] * 2.0
    assert abs(info["effective_cost"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# Dijkstra routing
# ---------------------------------------------------------------------------

def test_dijkstra_hop_count_gives_minimum_hops(engine: RoutingEngine) -> None:
    """Minimum Hop Count must return the globally shortest hop path via Dijkstra."""
    result = engine.compute_route("S1", "S6", "Minimum Hop Count")
    assert result.has_route
    assert result.best_path is not None
    hops = result.best_path.hops
    # Verify no candidate with fewer hops was overlooked
    assert hops == min(item.hops for item in result.candidates)
    assert hops <= 3


def test_dijkstra_hop_count_matches_networkx(topology: NetworkTopology, engine: RoutingEngine) -> None:
    """Dijkstra hop-count result must equal NetworkX's own shortest path."""
    active = topology.get_active_graph()
    nx_path = nx.dijkstra_path(active, "S1", "S6", weight=lambda u, v, d: 1)
    result = engine.compute_route("S1", "S6", "Minimum Hop Count")
    assert result.best_path is not None
    assert result.best_path.path == nx_path


def test_dijkstra_min_cost_gives_minimum_effective_cost(engine: RoutingEngine) -> None:
    """Minimum Link Cost must return the path with the smallest total effective cost."""
    result = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert result.has_route
    assert result.best_path is not None
    best_cost = result.best_path.total_cost
    assert best_cost == min(item.total_cost for item in result.candidates)
    expected = ["S1", "S2", "S4", "S6"]
    assert result.best_path.path == expected, (
        f"Expected {expected} but got {result.best_path.path}"
    )


def test_dijkstra_min_cost_matches_networkx(topology: NetworkTopology, engine: RoutingEngine) -> None:
    """Dijkstra min-cost result must equal NetworkX's weighted shortest path."""
    active = topology.get_active_graph()
    nx_path = nx.dijkstra_path(active, "S1", "S6", weight="effective_cost")
    result = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert result.best_path is not None
    assert result.best_path.path == nx_path


def test_minimum_hop_routing(engine: RoutingEngine) -> None:
    result = engine.compute_route("S1", "S6", "Minimum Hop Count")
    assert result.has_route
    assert result.best_path is not None
    hops = result.best_path.hops
    assert hops == min(item.hops for item in result.candidates)
    assert hops <= 3
    for item in result.candidates:
        if item.hops < hops:
            pytest.fail("A shorter hop path existed but was not selected")


def test_minimum_cost_routing(engine: RoutingEngine) -> None:
    result = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert result.has_route
    assert result.best_path is not None
    best_cost = result.best_path.total_cost
    assert best_cost == min(item.total_cost for item in result.candidates)
    expected = ["S1", "S2", "S4", "S6"]
    assert result.best_path.path == expected


# ---------------------------------------------------------------------------
# Alternative paths
# ---------------------------------------------------------------------------

def test_alternative_paths_exist(engine: RoutingEngine) -> None:
    result = engine.compute_route("S1", "S6", "Minimum Link Cost")
    labels = {tuple(item.path) for item in result.candidates}
    assert ("S1", "S2", "S4", "S6") in labels
    assert ("S1", "S3", "S5", "S6") in labels
    assert len(result.candidates) >= 3


# ---------------------------------------------------------------------------
# Failed link exclusion
# ---------------------------------------------------------------------------

def test_failed_link_is_excluded(topology: NetworkTopology, engine: RoutingEngine) -> None:
    topology.set_link_status("S2", "S4", "DOWN")
    result = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert result.has_route
    for item in result.candidates:
        hops = list(zip(item.path, item.path[1:]))
        assert ("S2", "S4") not in hops and ("S4", "S2") not in hops
    active = topology.get_active_graph()
    assert not active.has_edge("S2", "S4")


# ---------------------------------------------------------------------------
# Congestion and rerouting
# ---------------------------------------------------------------------------

def test_congestion_increases_effective_cost(topology: NetworkTopology, engine: RoutingEngine) -> None:
    before = topology.effective_cost("S2", "S4")
    topology.update_link_utilization("S2", "S4", 95)
    after = topology.effective_cost("S2", "S4")
    assert after > before
    monitor = NetworkMonitor(topology)
    state = monitor.congestion_state(95, "UP")
    assert state in ("CONGESTED", "SEVERE")
    # Congest the whole S1->S2->S4->S6 path to make it expensive enough to reroute
    topology.update_link_utilization("S1", "S2", 99)
    topology.update_link_utilization("S4", "S6", 99)
    result = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert result.best_path is not None
    assert result.best_path.path != ["S1", "S2", "S4", "S6"]


def test_congestion_reroutes_to_cheaper_path(topology: NetworkTopology, engine: RoutingEngine) -> None:
    """Congesting all links on S1→S2→S4→S6 makes it more expensive, causing rerouting."""
    # With quadratic formula, need to congest the whole path to beat alternatives
    topology.update_link_utilization("S1", "S2", 99)
    topology.update_link_utilization("S2", "S4", 99)
    topology.update_link_utilization("S4", "S6", 99)
    result = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert result.best_path is not None
    # The congested path should not be selected
    assert result.best_path.path != ["S1", "S2", "S4", "S6"]
    # Verify best path actually has lowest cost
    best_cost = result.best_path.total_cost
    for candidate in result.candidates:
        assert candidate.total_cost >= best_cost - 1e-9


def test_dynamic_cost_change_selects_cheaper_path(
    topology: NetworkTopology,
    engine: RoutingEngine,
) -> None:
    initial = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert initial.best_path is not None
    topology.update_link_cost("S2", "S4", 50)
    updated = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert updated.best_path is not None
    assert updated.best_path.path != initial.best_path.path
    assert updated.best_path.total_cost == min(p.total_cost for p in updated.candidates)


# ---------------------------------------------------------------------------
# Per-link traffic control
# ---------------------------------------------------------------------------

def test_set_link_utilization_updates_correctly(
    topology: NetworkTopology,
    simulator: NetworkSimulator,
) -> None:
    """set_link_utilization changes only utilization, not status."""
    ok = simulator.set_link_utilization("S1", "S2", 75.0)
    assert ok is True
    info = topology.get_link_info("S1", "S2")
    assert info["utilization"] == 75.0
    assert info["status"] == "UP"  # status unchanged


def test_set_link_utilization_updates_effective_cost(
    topology: NetworkTopology,
    simulator: NetworkSimulator,
) -> None:
    simulator.set_link_utilization("S1", "S2", 80.0)
    info = topology.get_link_info("S1", "S2")
    expected = info["cost"] * (1.0 + (80.0 / 100.0) ** 2)
    assert abs(info["effective_cost"] - expected) < 1e-9


def test_set_link_utilization_on_down_link_rejected(
    topology: NetworkTopology,
    simulator: NetworkSimulator,
) -> None:
    """Cannot set utilization on a DOWN link (congestion ≠ failure)."""
    topology.set_link_status("S1", "S2", "DOWN")
    ok = simulator.set_link_utilization("S1", "S2", 50.0)
    assert ok is False


def test_high_utilization_causes_reroute_in_min_cost_mode(
    topology: NetworkTopology,
    engine: RoutingEngine,
    simulator: NetworkSimulator,
) -> None:
    """Under Minimum Link Cost, congesting the entire active path forces rerouting."""
    initial = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert initial.best_path is not None
    initial_path = initial.best_path.path[:]
    # With the quadratic formula, need to congest all 3 links on S1->S2->S4->S6
    # to make its total effective cost exceed the alternative S1->S3->S5->S6
    simulator.set_link_utilization("S1", "S2", 99.0)
    simulator.set_link_utilization("S2", "S4", 99.0)
    simulator.set_link_utilization("S4", "S6", 99.0)
    updated = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert updated.best_path is not None
    # Path must have changed
    assert updated.best_path.path != initial_path
    # Best path does not use congested S1-S2
    hops = list(zip(updated.best_path.path, updated.best_path.path[1:]))
    assert ("S1", "S2") not in hops and ("S2", "S1") not in hops


def test_utilization_change_does_not_cause_link_failure(
    topology: NetworkTopology,
    simulator: NetworkSimulator,
) -> None:
    """High utilization should NOT mark a link DOWN."""
    simulator.set_link_utilization("S3", "S5", 100.0)
    info = topology.get_link_info("S3", "S5")
    assert info["status"] == "UP"
    assert info["utilization"] == 100.0


# ---------------------------------------------------------------------------
# Per-link failure and restoration
# ---------------------------------------------------------------------------

def test_fail_specific_link(
    topology: NetworkTopology,
    simulator: NetworkSimulator,
) -> None:
    ok = simulator.fail_specific_link("S1", "S2")
    assert ok is True
    info = topology.get_link_info("S1", "S2")
    assert info["status"] == "DOWN"


def test_fail_specific_link_idempotent(
    topology: NetworkTopology,
    simulator: NetworkSimulator,
) -> None:
    """Failing an already-DOWN link returns False without corrupting state."""
    topology.set_link_status("S1", "S2", "DOWN")
    ok = simulator.fail_specific_link("S1", "S2")
    assert ok is False
    info = topology.get_link_info("S1", "S2")
    assert info["status"] == "DOWN"


def test_restore_specific_link(
    topology: NetworkTopology,
    simulator: NetworkSimulator,
) -> None:
    topology.set_link_status("S1", "S2", "DOWN")
    ok = simulator.restore_specific_link("S1", "S2")
    assert ok is True
    info = topology.get_link_info("S1", "S2")
    assert info["status"] == "UP"


def test_restore_specific_link_already_up(
    topology: NetworkTopology,
    simulator: NetworkSimulator,
) -> None:
    """Restoring an UP link returns False without corrupting state."""
    ok = simulator.restore_specific_link("S1", "S2")
    assert ok is False
    info = topology.get_link_info("S1", "S2")
    assert info["status"] == "UP"


def test_fail_then_reroute_then_restore(
    topology: NetworkTopology,
    engine: RoutingEngine,
    simulator: NetworkSimulator,
) -> None:
    initial = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert initial.best_path is not None
    initial_path = initial.best_path.path[:]
    first_hop_u, first_hop_v = initial_path[0], initial_path[1]
    # Fail the first link on the route
    simulator.fail_specific_link(first_hop_u, first_hop_v)
    rerouted = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert rerouted.has_route
    assert rerouted.best_path is not None
    assert rerouted.best_path.path != initial_path
    # Restore and verify original path comes back
    simulator.restore_specific_link(first_hop_u, first_hop_v)
    restored = engine.compute_route("S1", "S6", "Minimum Link Cost")
    assert restored.best_path is not None
    assert restored.best_path.path == initial_path


# ---------------------------------------------------------------------------
# Unreachable destination
# ---------------------------------------------------------------------------

def test_no_available_route(topology: NetworkTopology, engine: RoutingEngine) -> None:
    for link in topology.get_links():
        topology.set_link_status(link["source"], link["destination"], "DOWN")
    result = engine.compute_route("S1", "S6", "Minimum Hop Count")
    assert not result.has_route
    assert result.best_path is None
    assert "No valid route" in result.message


def test_destination_unreachable_message(
    topology: NetworkTopology,
    engine: RoutingEngine,
) -> None:
    """Failing all links from S1 makes S6 unreachable."""
    topology.set_link_status("S1", "S2", "DOWN")
    topology.set_link_status("S1", "S3", "DOWN")
    result = engine.compute_route("S1", "S6", "Minimum Hop Count")
    assert not result.has_route
    assert result.best_path is None
    assert result.message  # some informative message must be set


def test_source_equals_destination(engine: RoutingEngine) -> None:
    result = engine.compute_route("S1", "S1", "Minimum Hop Count")
    assert result.has_route
    assert result.best_path is not None
    assert result.best_path.path == ["S1"]
    assert result.best_path.hops == 0


# ---------------------------------------------------------------------------
# Controller automatic rerouting
# ---------------------------------------------------------------------------

def test_route_recalculation_after_failure(controller: SDNController) -> None:
    first = controller.calculate_route("S1", "S6", "Minimum Link Cost")
    assert first.has_route
    old = first.best_path.path[:]
    u, v = old[0], old[1]
    controller.topology.set_link_status(u, v, "DOWN")
    second = controller.recalculate(f"Link {u}-{v} failed.")
    assert second is not None
    assert second.has_route
    assert second.best_path.path != old
    types = [event["event_type"] for event in controller.event_log]
    assert "REROUTE" in types or "ROUTE" in types


def test_controller_reroute_on_specific_link_failure(controller: SDNController) -> None:
    """Using fail_specific_link via simulator triggers automatic rerouting."""
    sim = NetworkSimulator(controller.topology)
    first = controller.calculate_route("S1", "S6", "Minimum Link Cost")
    old_path = first.best_path.path[:]
    # Fail a link on the active route
    fs, fd = old_path[0], old_path[1]
    ok = sim.fail_specific_link(fs, fd)
    assert ok is True
    second = controller.recalculate(f"Link {fs}-{fd} manually set to DOWN.")
    assert second is not None
    assert second.has_route
    assert second.best_path.path != old_path
    # Failed link not on new route
    new_hops = list(zip(second.best_path.path, second.best_path.path[1:]))
    assert (fs, fd) not in new_hops and (fd, fs) not in new_hops


def test_congestion_simulation_reroutes(controller: SDNController) -> None:
    """Congesting all links on the active path triggers automatic rerouting."""
    simulator = NetworkSimulator(controller.topology)
    first = controller.calculate_route("S1", "S6", "Minimum Link Cost")
    assert first.best_path is not None
    old = first.best_path.path[:]
    # Congest every link on the active path to ensure total cost exceeds alternatives
    for u, v in zip(old, old[1:]):
        controller.topology.update_link_utilization(u, v, 99.0)
    second = controller.recalculate("All links on active path congested to 99%.")
    assert second is not None
    assert second.has_route
    assert second.best_path.path != old


def test_restore_and_reset(controller: SDNController) -> None:
    simulator = NetworkSimulator(controller.topology)
    controller.calculate_route("S1", "S6", "Minimum Link Cost")
    simulator.simulate_link_failure(controller.active_route.path)
    simulator.restore_network()
    assert all(link["status"] == "UP" for link in controller.topology.get_links())
    simulator.reset_network()
    info = controller.topology.get_link_info("S2", "S4")
    assert info["status"] == "UP"
    assert info["cost"] == 2


# ---------------------------------------------------------------------------
# Monitor tiers
# ---------------------------------------------------------------------------

def test_monitor_normal(topology: NetworkTopology) -> None:
    monitor = NetworkMonitor(topology)
    assert monitor.congestion_state(30, "UP") == "NORMAL"


def test_monitor_moderate(topology: NetworkTopology) -> None:
    monitor = NetworkMonitor(topology)
    assert monitor.congestion_state(70, "UP") == "MODERATE"


def test_monitor_congested(topology: NetworkTopology) -> None:
    monitor = NetworkMonitor(topology)
    assert monitor.congestion_state(85, "UP") == "CONGESTED"


def test_monitor_severe(topology: NetworkTopology) -> None:
    monitor = NetworkMonitor(topology)
    assert monitor.congestion_state(95, "UP") == "SEVERE"


def test_monitor_failed(topology: NetworkTopology) -> None:
    monitor = NetworkMonitor(topology)
    assert monitor.congestion_state(0, "DOWN") == "FAILED"
