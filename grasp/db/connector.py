"""
A simple database connection module based on Records.

This module provides a DBConnector class that facilitates sending raw SQL queries
to a PostgreSQL database to fetch the required data. It leverages Records for
query execution.
"""

from typing import Literal
import records
from sqlalchemy.pool import NullPool

from grasp.schema import EdgeColumns, NodeColumns, NodeTableName, EventTable


class DBConnector:
    """
    A simple database connector class for PostgreSQL using Records.

    This class handles the connection to the database and allows execution
    of raw SQL queries to retrieve data.
    """

    def __init__(self, connection_string, engine_options: dict | None = None):
        """
        Initialize the database connector.

        Args:
            connection_string (str): The PostgreSQL connection string.
            engine_options (dict | None): Optional SQLAlchemy engine options
                forwarded to Records. By default uses NullPool to avoid
                holding on to idle connections.
        """
        self.connection_string = connection_string
        default_options = {"poolclass": NullPool, "pool_pre_ping": True}
        if engine_options:
            default_options.update(engine_options)
        self.db = records.Database(connection_string, **default_options)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _execute_query(self, sql: str, **params) -> records.RecordCollection:
        """
        Execute a SQL query with optional parameters.

        Args:
            sql (str): The SQL query to execute.
            **params: Parameters to pass to the query.

        Returns:
            records.RecordCollection: The result of the query.
        """
        return self.db.query(sql, **params)

    def _fetch_edges(
        self, operations: tuple[str, ...], start_time_ns: int, end_time_ns: int
    ) -> records.RecordCollection:
        """
        Fetch events between the specified start and end times.

        Args:
            operations (tuple): A tuple of operation names to
                filter events (like ("EVENT_READ", "EVENT_LSEEK").)
            start_time_ns (int): Start time in nanoseconds
                (like 11522707766463352938).
            end_time_ns (int): End time in nanoseconds.
        Returns:
            records.RecordCollection: The fetched events.
        """

        columns: list[str] = [
            EdgeColumns.SRC_NODE.value,
            EdgeColumns.SRC_INDEX_ID.value,
            EdgeColumns.OPERATION.value,
            EdgeColumns.DST_NODE.value,
            EdgeColumns.DST_INDEX_ID.value,
            EdgeColumns.TIMESTAMP.value,
        ]
        columns_str: str = ", ".join(columns)
        sql: str = (
            f"SELECT {columns_str} "
            f"FROM {EventTable.EVENT.value} WHERE operation in :operations "
            "AND timestamp_rec BETWEEN :start_time_ns AND :end_time_ns"
        )

        return self._execute_query(
            sql,
            operations=operations,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
        )

    def _fetch_nodes(
        self, node_ids: tuple[int, ...]
    ) -> records.RecordCollection:
        """
        Fetch nodes with the specified node IDs from multiple tables.

        Args:
            node_ids (tuple): A tuple of node IDs to fetch.

        Returns:
            records.RecordCollection: The fetched nodes from
            the subject_node_table, file_node_table, and netflow_node_table.
        """

        if not isinstance(node_ids, tuple):
            node_ids = tuple(node_ids)

        common_columns: list[str] = [
            NodeColumns.UUID.value,
            NodeColumns.INDEX_ID.value,
        ]
        common_columns_str: str = ", ".join(common_columns)

        cmd_column: Literal["cmd"] = NodeColumns.CMD.value
        path_column: Literal["path"] = NodeColumns.PATH.value
        network_columns: list[str] = [
            NodeColumns.SRC_ADDR.value,
            NodeColumns.SRC_PORT.value,
            NodeColumns.DST_ADDR.value,
            NodeColumns.DST_PORT.value,
        ]
        network_columns_str = ", ".join(network_columns)
        tables: list[str] = [
            NodeTableName.SUBJECT.value,
            NodeTableName.FILE.value,
            NodeTableName.NETFLOW.value,
        ]

        # Build UNION query with consistent column structure across all tables
        # Define null placeholders for missing columns
        null_cmd = "NULL::text AS cmd"
        null_path = "NULL::text AS path"
        null_network: str = (
            f"NULL::text AS {network_columns[0]}, "
            f"NULL::text AS {network_columns[1]}, "
            f"NULL::text AS {network_columns[2]}, "
            f"NULL::text AS {network_columns[3]}"
        )

        union_parts: list[str] = [
            f"SELECT {common_columns_str}, {cmd_column}, {path_column}, "
            f"{null_network}, 'subject'::text AS type "
            f"FROM {tables[0]} WHERE index_id IN :node_ids",
            f"SELECT {common_columns_str}, {path_column} AS cmd, "
            f"{path_column}, {null_network}, 'file'::text AS type "
            f"FROM {tables[1]} WHERE index_id IN :node_ids",
            f"SELECT {common_columns_str}, {null_cmd}, {null_path}, "
            f"{network_columns_str}, 'netflow'::text AS type "
            f"FROM {tables[2]} WHERE index_id IN :node_ids",
        ]

        sql: str = " UNION ALL ".join(union_parts)

        return self._execute_query(sql, node_ids=node_ids)

    def fetch_graph_data(
        self,
        operations: tuple[str, ...],
        start_time_ns: int,
        end_time_ns: int,
    ) -> tuple[records.RecordCollection, records.RecordCollection]:
        """
        Fetch edges and corresponding nodes for the graph data.

        Args:
            operations (tuple): A tuple of operation names to filter events.
            start_time_ns (int): Start time in nanoseconds.
            end_time_ns (int): End time in nanoseconds.
        Returns:
            tuple: A tuple containing:
                - records.RecordCollection: The fetched edges.
                - records.RecordCollection: The fetched nodes.
        """

        edges: records.RecordCollection = self._fetch_edges(
            operations, start_time_ns, end_time_ns
        )

        # Extract unique node IDs from edges
        node_ids: set[int] = set()
        for row in edges:
            node_ids.add(row.src_index_id)
            node_ids.add(row.dst_index_id)

        if not node_ids:
            nodes: records.RecordCollection = records.RecordCollection(
                iter([])
            )
        else:
            nodes: records.RecordCollection = self._fetch_nodes(
                tuple(node_ids)
            )
        # nodes: records.RecordCollection = self._fetch_nodes(tuple(node_ids))

        return edges, nodes

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()
