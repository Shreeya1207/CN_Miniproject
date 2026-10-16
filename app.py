"""Streamlit dashboard for SDN-based dynamic traffic routing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from controller import SDNController
from simulation import NetworkSimulator
from topology import NetworkTopology
from visualization import build_network_figure

CONFIG_PATH = Path(__file__).resolve().parent / "data" / "network_config.json"

st.set_page_config(
    page_title="SDN-Based Dynamic Traffic Routing",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Main application background and base typography */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .stApp p, .stApp span, .stApp label {
        color: #1e293b !important;
    }
    .stApp .stCaption {
        color: #475569 !important;
    }

    /* Metric cards styling */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    /* Text formatting in markdown blocks */
    .stMarkdown p {
        color: #1e293b !important;
    }
    .stMarkdown strong {
        color: #0f172a !important;
    }

    /* Status badges */
    .status-green { color: #15803d !important; font-weight: 700 !important; }
    .status-yellow { color: #a16207 !important; font-weight: 700 !important; }
    .status-red { color: #b91c1c !important; font-weight: 700 !important; }
    .status-blue { color: #1d4ed8 !important; font-weight: 700 !important; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #1e293b !important;
    }

    /* Selectbox styling */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-baseweb="select"] * {
        color: #0f172a !important;
    }

    /* Sidebar buttons styling for high contrast */
    .stButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border-color: #94a3b8 !important;
    }
    .stButton > button p, .stButton > button span {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Primary action button ("Calculate Route") */
    .stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }
    .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Sidebar info box contrast */
    div[data-testid="stAlert"] {
        background-color: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 6px !important;
    }
    div[data-testid="stAlert"] * {
        color: #1e40af !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session() -> None:
    if "controller" not in st.session_state:
        if not CONFIG_PATH.exists():
            st.error(f"Missing configuration file: {CONFIG_PATH}")
            st.stop()
        topology = NetworkTopology(CONFIG_PATH)
        st.session_state.controller = SDNController(topology)
        st.session_state.simulator = NetworkSimulator(topology)


def badge(text: str, kind: str) -> str:
    css = {
        "green": "status-green",
        "yellow": "status-yellow",
        "red": "status-red",
        "blue": "status-blue",
    }[kind]
    return f'<span class="{css}">{text}</span>'


def _link_label(source: str, destination: str) -> str:
    return f"{source}–{destination}"


def _parse_link_label(label: str) -> tuple[str, str]:
    parts = label.split("–")
    return parts[0], parts[1]


init_session()
controller: SDNController = st.session_state.controller
simulator: NetworkSimulator = st.session_state.simulator
topology = controller.topology
nodes = topology.get_nodes()

# Build sorted link labels for dropdowns
all_links = topology.get_links()
link_labels = [_link_label(lk["source"], lk["destination"]) for lk in all_links]
up_link_labels = [
    _link_label(lk["source"], lk["destination"])
    for lk in all_links
    if lk["status"] == "UP"
]
down_link_labels = [
    _link_label(lk["source"], lk["destination"])
    for lk in all_links
    if lk["status"] == "DOWN"
]

st.title("SDN-Based Dynamic Traffic Routing")
st.caption("Centralized and Programmable Network Traffic Management")

with st.sidebar:
    st.header("Controller Panel")
    source = st.selectbox("Source Node", nodes, index=nodes.index("S1") if "S1" in nodes else 0)
    destination = st.selectbox(
        "Destination Node",
        nodes,
        index=nodes.index("S6") if "S6" in nodes else min(1, len(nodes) - 1),
    )
    algorithm = st.selectbox(
        "Routing Algorithm",
        ["Minimum Hop Count", "Minimum Link Cost"],
        index=1,
    )

    calculate = st.button("Calculate Route", type="primary", use_container_width=True)
    st.markdown("**Network simulation**")
    col_a, col_b = st.columns(2)
    with col_a:
        inc_traffic = st.button("Increase Traffic", use_container_width=True)
        congestion = st.button("Simulate Congestion", use_container_width=True)
        restore = st.button("Restore Network", use_container_width=True)
    with col_b:
        dec_traffic = st.button("Decrease Traffic", use_container_width=True)
        failure = st.button("Simulate Link Failure", use_container_width=True)
        reset = st.button("Reset Network", use_container_width=True)

    # ------------------------------------------------------------------
    # Per-link traffic control
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("**Per-link traffic control**")
    if up_link_labels:
        selected_util_link = st.selectbox(
            "Select link to adjust",
            up_link_labels,
            key="util_link_select",
        )
        util_src, util_dst = _parse_link_label(selected_util_link)
        try:
            current_util = topology.get_link_info(util_src, util_dst)["utilization"]
        except KeyError:
            current_util = 0.0
        new_util = st.slider(
            "Link utilization (%)",
            min_value=0,
            max_value=100,
            value=int(current_util),
            step=1,
            key="util_slider",
        )
        apply_util = st.button("Apply Utilization", use_container_width=True)
    else:
        st.caption("No UP links available.")
        apply_util = False
        selected_util_link = None
        new_util = 0

    # ------------------------------------------------------------------
    # Per-link failure / restoration
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("**Per-link failure control**")
    if up_link_labels:
        selected_fail_link = st.selectbox(
            "Select link to fail",
            up_link_labels,
            key="fail_link_select",
        )
        fail_specific = st.button("Fail Selected Link", use_container_width=True)
    else:
        st.caption("No UP links available.")
        fail_specific = False
        selected_fail_link = None

    if down_link_labels:
        selected_restore_link = st.selectbox(
            "Select link to restore",
            down_link_labels,
            key="restore_link_select",
        )
        restore_specific = st.button("Restore Selected Link", use_container_width=True)
    else:
        st.caption("No DOWN links to restore.")
        restore_specific = False
        selected_restore_link = None

    st.info(
        "Simulation changes the live topology. "
        "The controller then recalculates routes from current link state."
    )

active_path = controller.active_route.path if controller.active_route else None

# ------------------------------------------------------------------
# Button handlers
# ------------------------------------------------------------------

if calculate:
    if source == destination:
        st.sidebar.warning("Source and destination are the same node.")
    result = controller.calculate_route(source, destination, algorithm)
    if not result.has_route and source != destination:
        st.sidebar.error(result.message)
    else:
        st.sidebar.success(result.message)

if inc_traffic:
    updated = simulator.increase_traffic(active_path)
    names = ", ".join(f"{u}-{v}" for u, v in updated) if updated else "no UP links"
    controller.recalculate(f"Traffic increased on {names}.")

if dec_traffic:
    updated = simulator.decrease_traffic(active_path)
    names = ", ".join(f"{u}-{v}" for u, v in updated) if updated else "no links"
    controller.recalculate(f"Traffic decreased on {names}.")

if congestion:
    target = simulator.simulate_congestion(active_path)
    if target is None:
        st.sidebar.error("No UP link is available to congest.")
    else:
        u, v = target
        info = topology.get_link_info(u, v)
        state = controller.monitor.congestion_state(info["utilization"], info["status"])
        controller.recalculate(
            f"Traffic increased on {u}-{v}. {u}-{v} became {state} "
            f"(utilization {info['utilization']:.0f}%)."
        )

if failure:
    target = simulator.simulate_link_failure(active_path)
    if target is None:
        st.sidebar.error("No UP link is available to fail.")
    else:
        u, v = target
        controller.recalculate(f"Link {u}-{v} failed. Link status set to DOWN.")

if restore:
    restored = simulator.restore_failed_links()
    simulator.restore_network()
    extra = ", ".join(f"{u}-{v}" for u, v in restored) if restored else "no failed links"
    controller.recalculate(f"Network restored. Recovered links: {extra}. Congestion eased.")

if reset:
    simulator.reset_network()
    controller.active_route = None
    controller.last_result = None
    controller.source = None
    controller.destination = None
    controller.log_event("RESET", "Network reset to the original configuration.")
    st.sidebar.success("Topology restored from network_config.json.")

# Per-link utilization apply
if apply_util and selected_util_link:
    util_src, util_dst = _parse_link_label(selected_util_link)
    ok = simulator.set_link_utilization(util_src, util_dst, float(new_util))
    if ok:
        info = topology.get_link_info(util_src, util_dst)
        state = controller.monitor.congestion_state(info["utilization"], info["status"])
        controller.recalculate(
            f"Utilization on {util_src}-{util_dst} set to {new_util}% ({state})."
        )
    else:
        st.sidebar.error(f"Cannot set utilization: {util_src}-{util_dst} is DOWN or does not exist.")

# Per-link failure
if fail_specific and selected_fail_link:
    fs, fd = _parse_link_label(selected_fail_link)
    ok = simulator.fail_specific_link(fs, fd)
    if ok:
        controller.recalculate(f"Link {fs}-{fd} manually set to DOWN.")
    else:
        st.sidebar.warning(f"Link {fs}-{fd} is already DOWN or does not exist.")

# Per-link restoration
if restore_specific and selected_restore_link:
    rs, rd = _parse_link_label(selected_restore_link)
    ok = simulator.restore_specific_link(rs, rd)
    if ok:
        controller.recalculate(f"Link {rs}-{rd} restored to UP.")
    else:
        st.sidebar.warning(f"Link {rs}-{rd} is already UP or does not exist.")

# ------------------------------------------------------------------
# Main dashboard display (unchanged layout)
# ------------------------------------------------------------------

overview = topology.overview()
st.subheader("Network Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Nodes", overview["total_nodes"])
m2.metric("Total Links", overview["total_links"])
m3.metric("Active Links", overview["active_links"])
m4.metric("Failed Links", overview["failed_links"])
m5.metric("Congested Links", overview["congested_links"])

st.subheader("Current Route")
summary = controller.current_route_summary()
status = summary["Route Status"]
status_kind = {"ACTIVE": "green", "IDLE": "blue", "UNAVAILABLE": "red"}.get(status, "yellow")
r1, r2, r3 = st.columns(3)
with r1:
    st.markdown(f"**Source:** {summary['Source']}")
    st.markdown(f"**Destination:** {summary['Destination']}")
    st.markdown(f"**Routing Algorithm:** {summary['Routing Algorithm']}")
with r2:
    st.markdown(f"**Current Path:** {summary['Current Path']}")
    st.markdown(f"**Hop Count:** {summary['Hop Count']}")
    st.markdown(f"**Total Cost:** {summary['Total Cost']}")
with r3:
    st.markdown(f"**Total Latency:** {summary['Total Latency']}")
    st.markdown(f"**Average Utilization:** {summary['Average Utilization']}")
    st.markdown(f"**Route Status:** {badge(status, status_kind)}", unsafe_allow_html=True)

st.subheader("Network Topology")
figure = build_network_figure(
    topology,
    controller.monitor,
    controller.active_route.path if controller.active_route else None,
)
st.plotly_chart(figure, use_container_width=True)

st.subheader("Available Paths")
if controller.last_result and controller.last_result.candidates:
    path_rows = [item.as_dict() for item in controller.last_result.candidates]
    st.dataframe(pd.DataFrame(path_rows), use_container_width=True, hide_index=True)
elif controller.source:
    st.warning("No valid route available.")
else:
    st.info("Click **Calculate Route** to compute candidate paths from the current topology.")

if controller.source and controller.destination:
    st.subheader("Routing Performance Comparison")
    comparison = controller.comparison()
    if comparison:
        st.dataframe(pd.DataFrame(comparison).T.reset_index().rename(columns={"index": "Algorithm"}), use_container_width=True, hide_index=True)

st.subheader("Link Status")
link_rows = []
for link in controller.monitor.inspect_all_links():
    link_rows.append(
        {
            "Source": link["source"],
            "Destination": link["destination"],
            "Cost": link["cost"],
            "Effective Cost": round(link["effective_cost"], 2),
            "Capacity (Mbps)": link["capacity"],
            "Utilization (%)": round(link["utilization"], 1),
            "Latency (ms)": link["latency"],
            "Status": link["status"],
            "Congestion": link["congestion"],
        }
    )
st.dataframe(pd.DataFrame(link_rows), use_container_width=True, hide_index=True)

st.subheader("Event Log")
if controller.event_log:
    st.dataframe(pd.DataFrame(controller.event_log), use_container_width=True, hide_index=True)
else:
    st.info("No controller events yet.")
