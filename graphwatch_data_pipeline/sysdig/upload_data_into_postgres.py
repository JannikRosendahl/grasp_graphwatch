#!/usr/bin/env python3

import csv
import os
from collections.abc import Iterable
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# ------------------ CONFIG ------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PARSED_DIR = BASE_DIR / "output" / "tables"
EVENT_DIR = PARSED_DIR / "event_table"

DB_HOST = os.environ["db_host"]
DB_PORT = os.environ["db_port"]
DB_NAME = os.getenv("db_name", "sysdig")
DB_USER = os.environ["db_user"]
DB_PASSWORD = os.environ["db_password"]

# psycopg2 prefers individual params instead of full URL
DB_CONN_ARGS = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
}

TABLES = (
    "event_table",
    "file_node_table",
    "subject_node_table",
    "netflow_node_table",
)

# ------------------ HELPERS ------------------


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def get_event_table_files() -> list[Path]:
    return sorted(EVENT_DIR.glob("*.csv"))


def drop_table(cur, table_name: str):
    cur.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)};")


def build_create_table_sql(table_name: str, columns: Iterable[str]) -> str:
    col_defs_list = []

    for col in columns:
        if table_name == "event_table" and col == "timestamp_rec":
            col_type = "BIGINT"
        elif (
            (
                table_name in {"file_node_table", "subject_node_table", "netflow_node_table"}
                and col == "index_id"
            )
            or table_name == "event_table"
            and (col == "src_index_id" or col == "dst_index_id")
        ):
            col_type = "INT"
        else:
            col_type = "TEXT"

        col_defs_list.append(f"{quote_identifier(col)} {col_type}")

    col_defs = ", ".join(col_defs_list)
    return f"CREATE TABLE IF NOT EXISTS {quote_identifier(table_name)} ({col_defs});"


def read_csv_header(file_path: Path) -> list:
    with file_path.open(newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def create_table_if_needed(cur, table_name: str, file_path: Path):
    columns = read_csv_header(file_path)

    if not columns:
        raise ValueError(f"CSV file is empty: {file_path}")

    sql = build_create_table_sql(table_name, columns)
    cur.execute(sql)


def copy_csv_into_table(cur, table_name: str, file_path: Path):
    with file_path.open("r") as f:
        cur.copy_expert(
            f"""
            COPY {quote_identifier(table_name)}
            FROM STDIN WITH CSV HEADER
            """,
            f,
        )


def upload_file(cur, table_name: str, file_path: Path):
    print(f"\n➡️ Processing {file_path}")

    create_table_if_needed(cur, table_name, file_path)

    copy_csv_into_table(cur, table_name, file_path)

    print(f"Loaded into table: {table_name}")


# ------------------ MAIN ------------------


def main():
    print("🔌 Connecting to database...")

    conn = psycopg2.connect(**DB_CONN_ARGS)  # type: ignore
    print(f"Connected to database: {DB_NAME} at {DB_HOST}:{DB_PORT} as {DB_USER}")
    conn.autocommit = True  # required for COPY

    try:
        with conn.cursor() as cur:
            # ---- Single table CSVs ----
            for table_name in TABLES[1:]:
                file_path = PARSED_DIR / f"{table_name}.csv"

                if not file_path.exists():
                    print(f"Missing: {file_path}, skipping")
                    continue

                drop_table(cur, table_name)
                upload_file(cur, table_name, file_path)

            # ---- Event table (multiple CSVs) ----
            print(f"\n Loading event_table from: {EVENT_DIR}")

            event_files = get_event_table_files()

            if not event_files:
                print("No event files found")
            else:
                drop_table(cur, "event_table")
                for file_path in event_files:
                    upload_file(cur, "event_table", file_path)

            print("\n DATA SUCCESSFULLY LOADED")

    finally:
        conn.close()
        print("🔌 Connection closed")


if __name__ == "__main__":
    main()
