#!/usr/bin/env python3

import csv
import html
import json
import re
import sqlite3
import tempfile
from functools import lru_cache
from pathlib import Path

# ------------------------------
# CONFIG
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "output" / "json"
OUTPUT_DIR = BASE_DIR / "output" / "edges_and_nodes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODES_FILE = OUTPUT_DIR / "nodes.csv"
EDGES_FILE = OUTPUT_DIR / "edges.csv"
PROGRESS_INTERVAL = 1_000_000
JSON_CHUNK_SIZE = 1024 * 1024

# ------------------------------
# Regex helpers
# ------------------------------

FD_RE = re.compile(r"fd=(?P<fd>-?\d+)\((?P<tag><[^>]+>)(?P<value>[^)]*)\)")
NAME_RE = re.compile(r"\bname=(?P<name>(\"[^\"]+\"|'[^']+'|\S+))")
TUPLE_RE = re.compile(r"\btuple=(?P<tuple>\S+)")
EXE_RE = re.compile(r"\bexe=(?P<exe>\S+)")
COMM_RE = re.compile(r"\bcomm=(?P<comm>\S+)")
SIZE_RE = re.compile(r"\bsize=(?P<size>\d+)")
DATA_RE = re.compile(r"\bdata=(?P<data>.*?)(?:\s+fd=|\s+size=|\s+tuple=|$)")
PROCESS_NUMERIC_FIELD_TEMPLATE = r"\b{key}=(?P<value>-?\d+)(?:\([^)]*\))?"

# ------------------------------
# Node helpers
# ------------------------------


def proc_node(tid):
    return f"proc:{tid}"


def file_node(path):
    return f"file:{path}"


def socket_node(addr):
    return f"socket:{addr}"


# ------------------------------
# General helpers
# ------------------------------


def normalize_event(event):
    normalized = {}
    for key, value in event.items():
        if isinstance(value, str):
            value = html.unescape(html.unescape(value))
        normalized[key] = value
    return normalized


def clean_name(value):
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
        value = value[1:-1]
    return value


@lru_cache(maxsize=None)
def numeric_field_pattern(key):
    return re.compile(PROCESS_NUMERIC_FIELD_TEMPLATE.format(key=re.escape(key)))


def extract_numeric_field(info, key):
    match = numeric_field_pattern(key).search(info)
    return int(match.group("value")) if match else None


def extract_exe(info):
    match = EXE_RE.search(info)
    return match.group("exe") if match else None


def extract_comm(info):
    match = COMM_RE.search(info)
    return match.group("comm") if match else None


def extract_size(info):
    match = SIZE_RE.search(info)
    return int(match.group("size")) if match else None


def extract_data_preview(info, max_len=120):
    match = DATA_RE.search(info)
    if not match:
        return ""
    data = match.group("data").strip()
    return data[:max_len] + "..." if len(data) > max_len else data


def extract_process_identity(event):
    info = event.get("evt.info", "")
    tid = event.get("thread.tid")
    if tid is None:
        tid = extract_numeric_field(info, "tid")
    if tid is not None:
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            tid = None
    return {
        "tid": tid,
        "pid": extract_numeric_field(info, "pid"),
        "ptid": extract_numeric_field(info, "ptid"),
        "exe": extract_exe(info),
        "comm": extract_comm(info),
        "proc_name": event.get("proc.name"),
    }


# ------------------------------
# Resource parsing
# ------------------------------


def extract_fd_resource(info):
    match = FD_RE.search(info)
    if not match:
        return None, None, None
    fd = int(match.group("fd"))
    tag = match.group("tag")
    value = match.group("value").strip()
    fd_tag = tag.strip("<>")
    if fd_tag in ("f", "d"):
        return fd, "file", value
    if "->" in value:
        return fd, "socket", value
    return fd, "fd", value


def extract_name_resource(info):
    match = NAME_RE.search(info)
    if not match:
        return None, None
    name = clean_name(match.group("name"))
    return ("file", name) if name else (None, None)


def extract_tuple_resource(info):
    match = TUPLE_RE.search(info)
    if not match:
        return None, None
    return "socket", match.group("tuple")


def extract_resource(info):
    fd, resource_type, resource_value = extract_fd_resource(info)
    if fd is not None:
        return fd, resource_type, resource_value
    resource_type, resource_value = extract_name_resource(info)
    if resource_type:
        return None, resource_type, resource_value
    resource_type, resource_value = extract_tuple_resource(info)
    if resource_type:
        return None, resource_type, resource_value
    return None, None, None


# ------------------------------
# Streaming event input
# ------------------------------


def discover_input_files(input_dir):
    files = {
        path for pattern in ("*.json", "*.jsonl", "*.ndjson") for path in input_dir.glob(pattern)
    }
    return sorted(files)


def iter_json_values(file_path, chunk_size=JSON_CHUNK_SIZE):
    """Stream a JSON array, one JSON object, or adjacent JSON/JSONL values."""
    decoder = json.JSONDecoder()
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        buffer = ""
        pos = 0
        eof = False
        array_mode = None
        array_finished = False

        while True:
            if not eof and (len(buffer) - pos < chunk_size // 2):
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer = buffer[pos:] + chunk
                    pos = 0
                else:
                    eof = True

            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1

            if array_mode is None:
                if pos >= len(buffer):
                    if eof:
                        return
                    continue
                array_mode = buffer[pos] == "["
                if array_mode:
                    pos += 1

            while pos < len(buffer) and (
                buffer[pos].isspace() or (array_mode and buffer[pos] == ",")
            ):
                pos += 1

            if array_mode and pos < len(buffer) and buffer[pos] == "]":
                pos += 1
                array_finished = True
                return

            if pos >= len(buffer):
                if eof:
                    if array_mode and not array_finished:
                        raise ValueError(f"Unterminated JSON array in {file_path}")
                    return
                continue

            try:
                value, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(
                        f"Invalid JSON in {file_path} near character {pos}: {exc}"
                    ) from exc
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                    continue
                eof = True
                continue

            pos = end
            yield value


def iter_events_from_file(file_path):
    """Stream valid events in exactly the order stored in the source file."""
    item_number = 0
    try:
        for value in iter_json_values(file_path):
            item_number += 1
            if not isinstance(value, dict) or "evt.outputtime" not in value:
                continue
            event = normalize_event(value)
            event["_source_file"] = file_path.name
            event["_line_no"] = item_number
            yield event
    except OSError as exc:
        raise RuntimeError(f"Could not read {file_path}: {exc}") from exc


def iter_all_events(input_dir):
    """
    Stream files in deterministic filename order and preserve the event order
    found inside each file. No sorting or temporary sorted runs are used.
    """
    for file_path in discover_input_files(input_dir):
        print(f"[*] Reading: {file_path.name}")
        yield from iter_events_from_file(file_path)


# ------------------------------
# Node and process helpers
# ------------------------------


def add_node(nodes, node_id, node_type, name, extra=None):
    if node_id not in nodes:
        nodes[node_id] = {"id": node_id, "type": node_type, "name": name}
    elif name:
        nodes[node_id]["name"] = name
    if extra:
        for key, value in extra.items():
            if value is not None:
                nodes[node_id][key] = value


def ensure_process(processes, tid):
    if tid not in processes:
        processes[tid] = {
            "tid": tid,
            "pid": None,
            "ptid": None,
            "exe": None,
            "comm": None,
            "proc_name": None,
            "first_seen": None,
            "last_seen": None,
            "exited": None,
            "exit_timestamp": None,
        }
    return processes[tid]


def update_process_from_event(processes, event):
    identity = extract_process_identity(event)
    tid = identity["tid"]
    if tid is None:
        return None
    proc = ensure_process(processes, tid)
    timestamp = event.get("evt.outputtime")
    if proc["first_seen"] is None:
        proc["first_seen"] = timestamp
    proc["last_seen"] = timestamp
    for key in ("pid", "ptid", "exe", "comm", "proc_name"):
        value = identity.get(key)
        if value is not None:
            proc[key] = value
    return tid


# ------------------------------
# Pass 1: collect all nodes
# ------------------------------


def collect_nodes(input_dir):
    nodes = {}
    processes = {}
    event_count = 0

    for event in iter_all_events(input_dir):
        event_count += 1
        evt_type = event.get("evt.type")
        evt_info = event.get("evt.info", "")
        timestamp = event.get("evt.outputtime")
        tid = update_process_from_event(processes, event)
        if tid is None:
            continue

        if evt_type in ("clone", "fork", "vfork"):
            child_tid = extract_numeric_field(evt_info, "res")
            if child_tid is not None and child_tid > 0:
                ensure_process(processes, child_tid)
            continue

        if evt_type in ("execve", "execveat"):
            executable = extract_exe(evt_info)
            if executable:
                ensure_process(processes, tid)["exe"] = executable
                add_node(nodes, file_node(executable), "file", executable)
            continue

        if evt_type in ("exit", "exit_group"):
            process = ensure_process(processes, tid)
            process["exited"] = "true"
            process["exit_timestamp"] = timestamp
            continue

        _, resource_type, resource_value = extract_resource(evt_info)
        if resource_type == "file" and resource_value:
            add_node(nodes, file_node(resource_value), "file", resource_value)
        elif resource_type == "socket" and resource_value:
            add_node(nodes, socket_node(resource_value), "socket", resource_value)

        if event_count % PROGRESS_INTERVAL == 0:
            print(
                f"[*] Pass 1 events: {event_count:,}, nodes: {len(nodes):,}, "
                f"processes: {len(processes):,}"
            )

    resolved_process_tids = set()
    for tid, process in processes.items():
        executable = process.get("exe")
        if not executable:
            continue
        add_node(
            nodes,
            proc_node(tid),
            "process",
            executable,
            extra={
                "tid": tid,
                "pid": process.get("pid"),
                "ptid": process.get("ptid"),
                "comm": process.get("comm"),
                "proc_name": process.get("proc_name"),
                "first_seen": process.get("first_seen"),
                "last_seen": process.get("last_seen"),
                "exited": process.get("exited"),
                "exit_timestamp": process.get("exit_timestamp"),
            },
        )
        resolved_process_tids.add(tid)

    return nodes, resolved_process_tids, event_count


# ------------------------------
# Pass 2: stream edges to CSV
# ------------------------------

EDGE_FIELDNAMES = [
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


def make_edge(src, dst, action, timestamp, event, resource_value=None):
    size = extract_size(event.get("evt.info", ""))
    return {
        "src": src,
        "dst": dst,
        "action": action,
        "timestamp": timestamp,
        "evt_num": event.get("evt.num"),
        "syscall": event.get("evt.type"),
        "direction": event.get("evt.dir"),
        "proc_name": event.get("proc.name"),
        "thread_tid": event.get("thread.tid"),
        "resource": resource_value or "",
        "size": size if size is not None else "",
        "source_file": event.get("_source_file", ""),
    }


def open_dedup_database():
    temp = tempfile.NamedTemporaryFile(
        prefix="sysdig_edge_dedup_",
        suffix=".sqlite3",
        dir=OUTPUT_DIR,
        delete=False,
    )
    path = Path(temp.name)
    temp.close()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("CREATE TABLE seen (edge_key TEXT PRIMARY KEY) WITHOUT ROWID")
    return connection, path


def write_edge_if_new(writer, dedup_connection, edge):
    key = json.dumps(
        [edge["src"], edge["dst"], edge["action"], edge["timestamp"], edge["evt_num"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    cursor = dedup_connection.execute("INSERT OR IGNORE INTO seen(edge_key) VALUES (?)", (key,))
    if cursor.rowcount == 0:
        return False
    writer.writerow(edge)
    return True


def process_and_write_edges(input_dir, resolved_process_tids):
    fd_table = {}
    edge_count = 0
    event_count = 0
    dedup_connection, dedup_path = open_dedup_database()

    try:
        with EDGES_FILE.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EDGE_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()

            for event in iter_all_events(input_dir):
                event_count += 1
                evt_type = event.get("evt.type")
                evt_info = event.get("evt.info", "")
                direction = event.get("evt.dir")
                timestamp = event.get("evt.outputtime")
                tid = extract_process_identity(event)["tid"]
                if tid is None:
                    continue

                fd_table.setdefault(tid, {})
                process_id = proc_node(tid)

                if evt_type in ("clone", "fork", "vfork"):
                    child_tid = extract_numeric_field(evt_info, "res")
                    if child_tid is not None and child_tid > 0:
                        fd_table.setdefault(child_tid, dict(fd_table.get(tid, {})))
                        if tid in resolved_process_tids and child_tid in resolved_process_tids:
                            edge = make_edge(
                                proc_node(tid),
                                proc_node(child_tid),
                                evt_type,
                                timestamp,
                                event,
                                str(child_tid),
                            )
                            edge_count += write_edge_if_new(writer, dedup_connection, edge)
                    continue

                if evt_type in ("execve", "execveat"):
                    executable = extract_exe(evt_info)
                    if executable and tid in resolved_process_tids:
                        edge = make_edge(
                            process_id,
                            file_node(executable),
                            "exec",
                            timestamp,
                            event,
                            executable,
                        )
                        edge_count += write_edge_if_new(writer, dedup_connection, edge)
                    continue

                if evt_type in ("exit", "exit_group"):
                    continue

                fd, resource_type, resource_value = extract_resource(evt_info)
                if fd is not None and resource_type and resource_value:
                    fd_table[tid][fd] = (resource_type, resource_value)
                if fd is not None and (not resource_type or not resource_value):
                    stored = fd_table[tid].get(fd)
                    if stored:
                        resource_type, resource_value = stored

                edge_spec = None
                if resource_type == "file" and resource_value:
                    resource_id = file_node(resource_value)
                    if evt_type == "read" and direction == "<":
                        edge_spec = (resource_id, process_id, "read")
                    elif evt_type == "write" and direction == "<":
                        edge_spec = (process_id, resource_id, "write")
                    elif evt_type in ("open", "openat") and direction == "<":
                        edge_spec = (process_id, resource_id, "open")
                    elif evt_type == "close" and direction == "<":
                        edge_spec = (process_id, resource_id, "close")
                elif resource_type == "socket" and resource_value:
                    resource_id = socket_node(resource_value)
                    if evt_type == "recvfrom" and direction == "<":
                        edge_spec = (resource_id, process_id, "recv")
                    elif evt_type == "sendto" and direction == "<":
                        edge_spec = (process_id, resource_id, "send")
                    elif evt_type == "read" and direction == "<":
                        edge_spec = (resource_id, process_id, "recv")
                    elif evt_type == "write" and direction == "<":
                        edge_spec = (process_id, resource_id, "send")
                    elif evt_type == "connect" and direction == "<":
                        edge_spec = (process_id, resource_id, "connect")
                    elif evt_type in ("accept", "accept4") and direction == "<":
                        edge_spec = (resource_id, process_id, "accept")
                    elif evt_type == "close" and direction == "<":
                        edge_spec = (process_id, resource_id, "close")

                if edge_spec and tid in resolved_process_tids:
                    edge = make_edge(
                        edge_spec[0],
                        edge_spec[1],
                        edge_spec[2],
                        timestamp,
                        event,
                        resource_value,
                    )
                    edge_count += write_edge_if_new(writer, dedup_connection, edge)

                if evt_type == "close" and fd is not None:
                    fd_table[tid].pop(fd, None)

                if event_count % PROGRESS_INTERVAL == 0:
                    dedup_connection.commit()
                    print(f"[*] Pass 2 events: {event_count:,}, edges written: {edge_count:,}")

        dedup_connection.commit()
    finally:
        dedup_connection.close()
        try:
            dedup_path.unlink()
        except OSError:
            pass

    return edge_count, event_count


# ------------------------------
# Save nodes
# ------------------------------


def save_nodes(nodes):
    fieldnames = [
        "id",
        "type",
        "name",
        "tid",
        "pid",
        "ptid",
        "comm",
        "proc_name",
        "first_seen",
        "last_seen",
        "exited",
        "exit_timestamp",
    ]
    with NODES_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(nodes.values())


# ------------------------------
# Main
# ------------------------------


def main():
    print(f"[*] Input directory:  {INPUT_DIR}")
    print(f"[*] Output directory: {OUTPUT_DIR}")
    files = discover_input_files(INPUT_DIR)
    if not files:
        raise SystemExit(f"[!] No .json, .jsonl, or .ndjson files found in {INPUT_DIR}")

    print(f"[*] Input files: {len(files)}")
    print("[*] Sorting: disabled")
    print("[*] Pass 1: collecting nodes and process metadata")
    nodes, resolved_process_tids, pass1_count = collect_nodes(INPUT_DIR)
    print(f"[*] Pass 1 events:       {pass1_count:,}")
    print(f"[*] Resolved processes:  {len(resolved_process_tids):,}")
    print(f"[*] Collected nodes:     {len(nodes):,}")

    save_nodes(nodes)
    print(f"[*] Nodes written: {len(nodes):,} -> {NODES_FILE}")

    print("[*] Pass 2: streaming and deduplicating edges")
    edge_count, pass2_count = process_and_write_edges(INPUT_DIR, resolved_process_tids)
    print(f"[*] Pass 2 events: {pass2_count:,}")
    print(f"[*] Edges written: {edge_count:,} -> {EDGES_FILE}")
    print("[*] Provenance graph exported")


if __name__ == "__main__":
    main()
