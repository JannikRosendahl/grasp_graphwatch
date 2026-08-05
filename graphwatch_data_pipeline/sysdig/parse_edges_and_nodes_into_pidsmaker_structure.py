#!/usr/bin/env python3

"""Export normalized graph tables from parse_data.py nodes.csv and edges.csv.

Input:
    output/edges_and_nodes/nodes.csv
    output/edges_and_nodes/edges.csv

Output:
    output/tables/subject_node_table.csv
    output/tables/file_node_table.csv
    output/tables/netflow_node_table.csv
    output/tables/event_table/event_table_batch_00000.csv
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

# ------------------------------
# CONFIG
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "output" / "edges_and_nodes"
OUTPUT_DIR = BASE_DIR / "output" / "tables"
EVENT_DIR = OUTPUT_DIR / "event_table"

NODES_FILE = INPUT_DIR / "nodes.csv"
EDGES_FILE = INPUT_DIR / "edges.csv"


# Operations produced by parse_data.py.
# Keep this list lowercase.
EDGE_OPERATION_FILTERS = {
    "accept",
    "close",
    "connect",
    "exec",
    "execve",
    "fork",
    "clone",
    "open",
    "read",
    "recv",
    "send",
    "spawn",
    "write",
    "exit",
}


NODE_INPUT_COLUMNS = [
    "id",
    "type",
    "name",
    "tid",
    "exited",
    "exit_timestamp",
]


EDGE_INPUT_COLUMNS = [
    "src",
    "dst",
    "action",
    "timestamp",
    "evt_num",
    "syscall",
    "direction",
    "proc_name",
    "thread_tid",
    "resource",
    "size",
    "source_file",
]


SUBJECT_COLUMNS = [
    "node_uuid",
    "hash_id",
    "path",
    "cmd",
    "index_id",
]


FILE_COLUMNS = [
    "node_uuid",
    "hash_id",
    "path",
    "index_id",
]


NETFLOW_COLUMNS = [
    "node_uuid",
    "hash_id",
    "src_addr",
    "src_port",
    "dst_addr",
    "dst_port",
    "index_id",
]


EVENT_COLUMNS = [
    "src_node",
    "src_index_id",
    "operation",
    "dst_node",
    "dst_index_id",
    "event_uuid",
    "timestamp_rec",
    "_id",
]


# ------------------------------
# BASIC HELPERS
# ------------------------------


def _read_csv_required(path: Path) -> pd.DataFrame:
    """Read a required CSV file using pandas string dtype."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    try:
        return pd.read_csv(path, dtype="string")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file exists but is empty: {path}") from exc


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure that all expected columns exist."""
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def _safe_string(value, default: str = "") -> str:
    """Convert a value to a safe stripped string, handling pd.NA first."""
    if pd.isna(value):
        return default

    text = str(value).strip()

    if text == "":
        return default

    return text


def _strip_node_prefix(node_id: str, prefix: str) -> str:
    """Remove a node prefix like file: or socket: if present."""
    if node_id.startswith(prefix):
        return node_id[len(prefix) :]
    return node_id


def _write_table(df: pd.DataFrame, output_file: Path, columns: list[str]) -> None:
    """Write only selected columns, creating missing columns if needed."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA

    df.loc[:, columns].to_csv(output_file, index=False)
    print(f"saved {len(df)} rows to {output_file}")


# ------------------------------
# TIMESTAMP HANDLING
# ------------------------------


def _normalize_timestamp_to_ns(value):
    """Normalize timestamp values to nanoseconds.

    parse_data.py normally writes Sysdig evt.outputtime directly.
    That value is already nanoseconds, for example:

        1781597042007568406

    This function also accepts seconds, milliseconds, and microseconds.
    """
    text = _safe_string(value)

    if text == "":
        return pd.NA

    try:
        number = Decimal(text)
    except InvalidOperation:
        return pd.NA

    try:
        ts = int(number)
    except (TypeError, ValueError, OverflowError):
        return pd.NA

    # Already nanoseconds.
    if ts >= 10**17:
        return ts

    # Microseconds.
    if ts >= 10**14:
        return ts * 1_000

    # Milliseconds.
    if ts >= 10**11:
        return ts * 1_000_000

    # Seconds.
    return ts * 1_000_000_000


# ------------------------------
# SOCKET / NETFLOW HANDLING
# ------------------------------


def _split_host_port(endpoint) -> tuple[str, str]:
    """Split an endpoint into host and port.

    Examples:
        141.71.30.148:137      -> 141.71.30.148, 137
        [::1]:443              -> ::1, 443
        /tmp/socket            -> /tmp/socket, ""

    This function is safe for pd.NA.
    """
    endpoint_text = _safe_string(endpoint)

    if endpoint_text == "":
        return "", ""

    # Bracketed IPv6 form: [::1]:443
    if endpoint_text.startswith("[") and "]:" in endpoint_text:
        host, port = endpoint_text.rsplit("]:", 1)
        host = host.lstrip("[")
        return host.strip(), port.strip()

    # Normal IPv4 or hostname with port.
    if ":" in endpoint_text:
        host, port = endpoint_text.rsplit(":", 1)
        return host.strip(), port.strip()

    return endpoint_text, ""


def _parse_socket_name(socket_name) -> tuple[str, str, str, str]:
    """Parse socket node names from parse_data.py.

    Expected examples:
        141.71.30.148:137->141.71.31.198:137
        141.71.30.148:137
        socket:141.71.30.148:137->141.71.31.198:137

    Returns:
        src_addr, src_port, dst_addr, dst_port
    """
    socket_text = _safe_string(socket_name)

    if socket_text == "":
        return "", "", "", ""

    socket_text = _strip_node_prefix(socket_text, "socket:")

    if "->" not in socket_text:
        src_addr, src_port = _split_host_port(socket_text)
        return src_addr, src_port, "", ""

    left, right = socket_text.split("->", 1)

    src_addr, src_port = _split_host_port(left)
    dst_addr, dst_port = _split_host_port(right)

    return src_addr, src_port, dst_addr, dst_port


# ------------------------------
# NODE TABLE PREPARATION
# ------------------------------


def _prepare_node_frame() -> pd.DataFrame:
    """Load nodes.csv and create normalized node metadata."""
    node_df = _read_csv_required(NODES_FILE)
    node_df = _ensure_columns(node_df, NODE_INPUT_COLUMNS)

    print(f"loaded {len(node_df)} raw nodes")

    node_df["id"] = node_df["id"].map(_safe_string)
    node_df = node_df[node_df["id"].ne("")].copy()

    node_df = node_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    print(f"{len(node_df)} unique valid nodes remain")

    node_df["node_uuid"] = node_df["id"]
    node_df["hash_id"] = node_df["id"]

    node_df["index_id"] = pd.RangeIndex(
        start=0,
        stop=len(node_df),
        step=1,
    )

    node_df["type_norm"] = node_df["type"].fillna("").str.lower().str.strip()

    # Make name reliable.
    node_df["name"] = node_df["name"].fillna("").astype("string")

    # If name is empty, derive it from id.
    missing_name = node_df["name"].fillna("").str.strip().eq("")

    node_df.loc[
        missing_name & node_df["id"].str.startswith("file:"),
        "name",
    ] = node_df.loc[
        missing_name & node_df["id"].str.startswith("file:"),
        "id",
    ].map(lambda x: _strip_node_prefix(str(x), "file:"))

    node_df.loc[
        missing_name & node_df["id"].str.startswith("socket:"),
        "name",
    ] = node_df.loc[
        missing_name & node_df["id"].str.startswith("socket:"),
        "id",
    ].map(lambda x: _strip_node_prefix(str(x), "socket:"))

    node_df.loc[
        missing_name & node_df["id"].str.startswith("proc:"),
        "name",
    ] = node_df.loc[
        missing_name & node_df["id"].str.startswith("proc:"),
        "id",
    ]

    return node_df


def _build_subject_node_table(node_df: pd.DataFrame) -> None:
    """Build process / subject table."""
    subject_df = node_df[node_df["type_norm"].eq("process")].copy()

    subject_df["cmd"] = subject_df["name"].fillna("unknown")

    subject_df["path"] = subject_df["name"].where(
        subject_df["name"].fillna("").str.startswith("/"),
        "",
    )

    _write_table(
        subject_df,
        OUTPUT_DIR / "subject_node_table.csv",
        SUBJECT_COLUMNS,
    )


def _build_file_node_table(node_df: pd.DataFrame) -> None:
    """Build file node table."""
    file_df = node_df[node_df["type_norm"].eq("file")].copy()

    file_df["path"] = file_df["name"].fillna("")

    _write_table(
        file_df,
        OUTPUT_DIR / "file_node_table.csv",
        FILE_COLUMNS,
    )


def _build_netflow_node_table(node_df: pd.DataFrame) -> None:
    """Build network flow / socket table."""
    netflow_df = node_df[node_df["type_norm"].eq("socket")].copy()

    if netflow_df.empty:
        netflow_df["src_addr"] = pd.Series(dtype="string")
        netflow_df["src_port"] = pd.Series(dtype="string")
        netflow_df["dst_addr"] = pd.Series(dtype="string")
        netflow_df["dst_port"] = pd.Series(dtype="string")

        _write_table(
            netflow_df,
            OUTPUT_DIR / "netflow_node_table.csv",
            NETFLOW_COLUMNS,
        )
        return

    parsed = netflow_df["name"].apply(_parse_socket_name)

    netflow_df["src_addr"] = parsed.apply(lambda item: item[0])
    netflow_df["src_port"] = parsed.apply(lambda item: item[1])
    netflow_df["dst_addr"] = parsed.apply(lambda item: item[2])
    netflow_df["dst_port"] = parsed.apply(lambda item: item[3])

    _write_table(
        netflow_df,
        OUTPUT_DIR / "netflow_node_table.csv",
        NETFLOW_COLUMNS,
    )


def _build_node_tables(node_df: pd.DataFrame) -> None:
    _build_subject_node_table(node_df)
    _build_file_node_table(node_df)
    _build_netflow_node_table(node_df)


def _build_vertex_index_map(node_df: pd.DataFrame) -> pd.Series:
    """Map node id to index id."""
    return node_df.set_index("id")["index_id"]


# ------------------------------
# EVENT TABLE PREPARATION
# ------------------------------


def _make_event_uuid(edge_df: pd.DataFrame) -> pd.Series:
    """Create stable event UUIDs.

    Prefer:
        source_file:evt_num

    Fallback:
        event:<row_number>
    """
    source_file = edge_df["source_file"].fillna("").astype("string").str.strip()
    evt_num = edge_df["evt_num"].fillna("").astype("string").str.strip()

    base_uuid = source_file + ":" + evt_num

    fallback_uuid = pd.Series(
        [f"event:{i}" for i in range(len(edge_df))],
        index=edge_df.index,
        dtype="string",
    )

    valid_base = evt_num.ne("") | source_file.ne("")

    return base_uuid.where(valid_base, fallback_uuid)


def _prepare_edge_frame() -> pd.DataFrame:
    """Load and clean edges.csv."""
    edge_df = _read_csv_required(EDGES_FILE)
    edge_df = _ensure_columns(edge_df, EDGE_INPUT_COLUMNS)

    print(f"loaded {len(edge_df)} raw edges")

    edge_df["src"] = edge_df["src"].map(_safe_string)
    edge_df["dst"] = edge_df["dst"].map(_safe_string)
    edge_df["operation"] = edge_df["action"].fillna("").str.lower().str.strip()

    edge_df = edge_df[
        edge_df["src"].ne("") & edge_df["dst"].ne("") & edge_df["operation"].ne("")
    ].copy()

    print(f"{len(edge_df)} edges remain after basic cleanup")

    edge_df = edge_df[edge_df["operation"].isin(EDGE_OPERATION_FILTERS)].copy()

    print(f"{len(edge_df)} edges remain after operation filter")

    return edge_df


def _build_event_table(vertex_index_map: pd.Series) -> None:
    """Build the normalized event table."""
    edge_df = _prepare_edge_frame()

    edge_df["src_node"] = edge_df["src"]
    edge_df["dst_node"] = edge_df["dst"]

    edge_df["src_index_id"] = edge_df["src_node"].map(vertex_index_map)
    edge_df["dst_index_id"] = edge_df["dst_node"].map(vertex_index_map)

    before_mapping_filter = len(edge_df)

    edge_df = edge_df[edge_df["src_index_id"].notna() & edge_df["dst_index_id"].notna()].copy()

    dropped = before_mapping_filter - len(edge_df)

    if dropped:
        print(f"dropped {dropped} edges because src/dst node mapping was missing")

    edge_df["src_index_id"] = edge_df["src_index_id"].astype("int64")
    edge_df["dst_index_id"] = edge_df["dst_index_id"].astype("int64")

    edge_df["event_uuid"] = _make_event_uuid(edge_df)

    edge_df["timestamp_rec"] = edge_df["timestamp"].apply(_normalize_timestamp_to_ns)
    edge_df["timestamp_rec"] = pd.to_numeric(
        edge_df["timestamp_rec"],
        errors="coerce",
    ).astype("Int64")

    edge_df["_id"] = pd.RangeIndex(
        start=0,
        stop=len(edge_df),
        step=1,
    )

    event_df = edge_df.loc[:, EVENT_COLUMNS].copy()

    EVENT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale event batches from previous runs.
    for old_batch in EVENT_DIR.glob("event_table_batch_*.csv"):
        old_batch.unlink()

    _write_table(
        event_df,
        EVENT_DIR / "event_table_batch_00000.csv",
        EVENT_COLUMNS,
    )


# ------------------------------
# SANITY CHECKS
# ------------------------------


def _run_sanity_checks() -> None:
    """Check that required input files exist before doing work."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    if not NODES_FILE.exists():
        raise FileNotFoundError(f"Missing nodes file: {NODES_FILE}")

    if not EDGES_FILE.exists():
        raise FileNotFoundError(f"Missing edges file: {EDGES_FILE}")


# ------------------------------
# MAIN
# ------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"input nodes: {NODES_FILE}")
    print(f"input edges: {EDGES_FILE}")
    print(f"output dir:  {OUTPUT_DIR}")

    _run_sanity_checks()

    node_df = _prepare_node_frame()
    vertex_index_map = _build_vertex_index_map(node_df)

    _build_node_tables(node_df)
    _build_event_table(vertex_index_map)

    print("[*] Normalized graph tables exported")


if __name__ == "__main__":
    main()
