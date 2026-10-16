"""Network topology management for the SDN simulation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

VALID_STATUS = {"UP", "DOWN"}


class NetworkTopology:
    """Undirected graph of SDN switches with per-link traffic attributes."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.graph = nx.Graph()
        self._initial_links: list[dict[str, Any]] = []
        self.thresholds = {
            "normal_max": 60,
            "moderate_max": 80,
            "congested_min": 81,
        }
        self.penalty_weight = 10.0
        if config_path is not None:
            self.load_config(config_path)

    def load_config(self, config_path: str | Path) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Network configuration not found: {path}")
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        self.config_path = path
        self.thresholds.update(config.get("thresholds", {}))
        model = config.get("congestion_model", {})
        self.penalty_weight = float(model.get("penalty_weight", 10.0))
        self.create_topology(config.get("nodes", []), config.get("links", []))

    def create_topology(
        self,
        nodes: Iterable[str],
        links: Iterable[dict[str, Any]],
    ) -> nx.Graph:
        """Build a fresh graph from node and link definitions."""
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        self._initial_links = [copy.deepcopy(link) for link in links]
        for link in self._initial_links:
            self.add_link(
                source=link["source"],
                destination=link["destination"],
                cost=float(link["cost"]),
                capacity=float(link["capacity"]),
                utilization=float(link["utilization"]),
                latency=float(link["latency"]),
                status=str(link.get("status", "UP")).upper(),
            )
        return self.graph

    def add_node(self, node: str) -> None:
        """Add a single switch node to the topology."""
        self.graph.add_node(node)

    def fail_link(self, source: str, destination: str) -> None:
        """Set a link status to DOWN (convenience alias)."""
        self.set_link_status(source, destination, "DOWN")

    def add_link(
        self,
        source: str,
        destination: str,
        cost: float,
        capacity: float,
        utilization: float,
        latency: float,
        status: str = "UP",
    ) -> None:
        status = status.upper()
        if status not in VALID_STATUS:
            raise ValueError(f"Invalid link status: {status}")
        utilization = self._clamp_utilization(utilization)
        if source not in self.graph:
            self.graph.add_node(source)
        if destination not in self.graph:
            self.graph.add_node(destination)
        self.graph.add_edge(
            source,
            destination,
            cost=float(cost),
            capacity=float(capacity),
            utilization=float(utilization),
            latency=float(latency),
            status=status,
        )
        self._refresh_effective_cost(source, destination)

    def _edge_key(self, source: str, destination: str) -> tuple[str, str]:
        if not self.graph.has_edge(source, destination):
            raise KeyError(f"Link {source}-{destination} does not exist")
        return source, destination

    def _clamp_utilization(self, utilization: float) -> float:
        return max(0.0, min(100.0, float(utilization)))

    def effective_cost(self, source: str, destination: str) -> float:
        """effective_cost = base_cost * (1 + (utilization / 100) ** 2).

        Congestion has a quadratic impact so heavily loaded links become
        disproportionately expensive, causing the routing engine to prefer
        alternative paths under high utilization.
        """
        u, v = self._edge_key(source, destination)
        data = self.graph[u][v]
        util_fraction = float(data["utilization"]) / 100.0
        return float(data["cost"]) * (1.0 + util_fraction ** 2)

    def _refresh_effective_cost(self, source: str, destination: str) -> None:
        u, v = self._edge_key(source, destination)
        self.graph[u][v]["effective_cost"] = self.effective_cost(u, v)

    def update_link_cost(self, source: str, destination: str, cost: float) -> None:
        u, v = self._edge_key(source, destination)
        self.graph[u][v]["cost"] = float(cost)
        self._refresh_effective_cost(u, v)

    def update_link_utilization(self, source: str, destination: str, utilization: float) -> None:
        u, v = self._edge_key(source, destination)
        self.graph[u][v]["utilization"] = self._clamp_utilization(utilization)
        self._refresh_effective_cost(u, v)

    def update_link_latency(self, source: str, destination: str, latency: float) -> None:
        u, v = self._edge_key(source, destination)
        self.graph[u][v]["latency"] = float(latency)

    def set_link_status(self, source: str, destination: str, status: str) -> None:
        status = status.upper()
        if status not in VALID_STATUS:
            raise ValueError(f"Invalid link status: {status}")
        u, v = self._edge_key(source, destination)
        self.graph[u][v]["status"] = status

    def restore_link(self, source: str, destination: str) -> None:
        self.set_link_status(source, destination, "UP")

    def get_active_graph(self) -> nx.Graph:
        """Return a copy of the topology containing only UP links."""
        active = nx.Graph()
        active.add_nodes_from(self.graph.nodes)
        for u, v, data in self.graph.edges(data=True):
            if data.get("status", "UP") == "UP":
                active.add_edge(u, v, **dict(data))
        return active

    def get_links(self) -> list[dict[str, Any]]:
        links = []
        for u, v, data in sorted(self.graph.edges(data=True)):
            info = self.get_link_info(u, v)
            links.append(info)
        return links

    def get_link_info(self, source: str, destination: str) -> dict[str, Any]:
        u, v = self._edge_key(source, destination)
        data = self.graph[u][v]
        effective = float(data["effective_cost"]) if "effective_cost" in data else self.effective_cost(u, v)
        return {
            "source": u,
            "destination": v,
            "cost": float(data["cost"]),
            "effective_cost": effective,
            "capacity": float(data["capacity"]),
            "utilization": float(data["utilization"]),
            "latency": float(data["latency"]),
            "status": str(data.get("status", "UP")),
        }

    def get_nodes(self) -> list[str]:
        return sorted(self.graph.nodes)

    def has_node(self, node: str) -> bool:
        return node in self.graph

    def overview(self) -> dict[str, int]:
        links = self.get_links()
        failed = sum(1 for link in links if link["status"] == "DOWN")
        congested = sum(
            1
            for link in links
            if link["status"] == "UP"
            and link["utilization"] >= self.thresholds["congested_min"]
        )
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_links": self.graph.number_of_edges(),
            "active_links": self.graph.number_of_edges() - failed,
            "failed_links": failed,
            "congested_links": congested,
        }

    def reset(self) -> None:
        """Restore the topology to the originally loaded configuration."""
        nodes = list(self.graph.nodes) if not self._initial_links else []
        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)
            return
        if self._initial_links:
            unique_nodes = set()
            for link in self._initial_links:
                unique_nodes.add(link["source"])
                unique_nodes.add(link["destination"])
            self.create_topology(sorted(unique_nodes), self._initial_links)
            return
        self.create_topology(nodes, [])
