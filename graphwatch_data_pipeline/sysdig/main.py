#!/usr/bin/env python3


import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> None:
    script_path = Path(__file__).parent / script_name
    subprocess.check_call([sys.executable, str(script_path)])


def main() -> None:
    run_script("get_json_from_scap.py")
    run_script("parse_json_into_edges_and_nodes.py")
    run_script("parse_edges_and_nodes_into_pidsmaker_structure.py")
    run_script("upload_data_into_postgres.py")


if __name__ == "__main__":
    main()
