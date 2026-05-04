from enum import Enum
import string


class NodeType(str, Enum):
    """Canonical node types used across the project."""

    SUBJECT = "subject"
    FILE = "file"
    NETFLOW = "netflow"


class EdgeColumns(str, Enum):
    """Edge column names expected from the database layer."""

    SRC_NODE = "src_node"
    SRC_INDEX_ID = "src_index_id"
    OPERATION = "operation"
    DST_NODE = "dst_node"
    DST_INDEX_ID = "dst_index_id"
    TIMESTAMP = "timestamp_rec"


class NodeColumns(str, Enum):
    """Node column names expected from the database layer."""

    UUID = "node_uuid"
    INDEX_ID = "index_id"
    TYPE = "type"
    CMD = "cmd"
    PATH = "path"
    SRC_ADDR = "src_addr"
    SRC_PORT = "src_port"
    DST_ADDR = "dst_addr"
    DST_PORT = "dst_port"


class NodeTableName(str, Enum):
    """Database table names for graph nodes."""

    SUBJECT = "subject_node_table"
    FILE = "file_node_table"
    # SUBJECT = "subject_node_table_lotl"
    # FILE = "file_node_table_lotl"
    NETFLOW = "netflow_node_table"


class EventTable(str, Enum):
    """Database table names for events/edges."""

    EVENT = "event_table"


class OptcOperation(str, Enum):
    """Operations used for OPTC datasets."""

    OPEN = "OPEN"
    READ = "READ"
    CREATE = "CREATE"
    MESSAGE = "MESSAGE"
    MODIFY = "MODIFY"
    START = "START"
    RENAME = "RENAME"
    DELETE = "DELETE"
    TERMINATE = "TERMINATE"
    WRITE = "WRITE"


class TcOperation(str, Enum):
    """Operations used for TC datasets."""

    EVENT_CONNECT = "EVENT_CONNECT"
    EVENT_EXECUTE = "EVENT_EXECUTE"
    EVENT_OPEN = "EVENT_OPEN"
    EVENT_READ = "EVENT_READ"
    EVENT_RECVFROM = "EVENT_RECVFROM"
    EVENT_RECVMSG = "EVENT_RECVMSG"
    EVENT_SENDMSG = "EVENT_SENDMSG"
    EVENT_SENDTO = "EVENT_SENDTO"
    EVENT_WRITE = "EVENT_WRITE"
    EVENT_CLONE = "EVENT_CLONE"


class TcOperationModified(str, Enum):
    """Operations used for TC datasets."""

    EVENT_CONNECT = "EVENT_CONNECT"
    EVENT_EXECUTE = "EVENT_EXECUTE"
    EVENT_OPEN = "EVENT_OPEN"
    EVENT_READ = "EVENT_READ"
    EVENT_RECVFROM = "EVENT_RECVFROM"
    EVENT_RECVMSG = "EVENT_RECVMSG"
    EVENT_SENDMSG = "EVENT_SENDMSG"
    EVENT_SENDTO = "EVENT_SENDTO"
    EVENT_WRITE = "EVENT_WRITE"
    EVENT_CLONE = "EVENT_CLONE"


class DatasetName(str, Enum):
    """Supported dataset identifiers from experiment config."""

    CADETS_E3 = "cadets_e3"
    CADETS_E3_LOTL = "cadets_e3_lotl"
    CADETS_E5 = "cadets_e5"
    CLEARSCOPE_E3 = "clearscope_e3"
    CLEARSCOPE_E5 = "clearscope_e5"
    THEIA_E3 = "theia_e3"
    THEIA_E3_LOTL = "theia_e3_lotl"
    THEIA_E5 = "theia_e5"
    OPTC_051 = "optc_051"
    OPTC_201 = "optc_201"
    OPTC_501 = "optc_501"


class PathTransformerConstants(str, Enum):
    """Constants for the transformer-based path representation."""

    CHARS = string.printable
    PADDING_CHAR = 0


class TimeStringFormat(str, Enum):
    """Standardized time string format."""

    FMT = "%Y-%m-%d %H:%M:%S"
