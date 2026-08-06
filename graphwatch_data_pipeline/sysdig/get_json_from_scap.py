#!/usr/bin/env python3

import json
import subprocess
import threading
from pathlib import Path

# ------------------------------
# CONFIG
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent

SCAP_DIR = BASE_DIR / "input" / "sysdig_scaps"


# Output directory
OUTPUT_DIR = BASE_DIR / "output" / "json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def scap_to_json(scap_file, output_file):
    """
    Converts a single sysdig .scap file to JSON lines using sysdig CLI
    """

    print(f"[*] Processing {scap_file} -> {output_file}")

    cmd = ["sysdig", "-r", str(scap_file), "-j"]

    with open(output_file, "w") as outfile:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Drain stderr on a separate thread so a chatty sysdig process can't
        # fill the stderr pipe buffer and deadlock while we're only reading stdout.
        stderr_lines: list[str] = []

        def drain_stderr():
            for line in proc.stderr:  # type: ignore
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()

        for line in proc.stdout:  # type: ignore
            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
                outfile.write(json.dumps(event) + "\n")
            except json.JSONDecodeError:
                continue

        proc.wait()
        stderr_thread.join()

        if proc.returncode != 0:
            raise RuntimeError(f"sysdig failed for {scap_file}:\n{''.join(stderr_lines)}")


if __name__ == "__main__":
    print("[*] Searching for scap files...")

    scap_files = list(SCAP_DIR.glob("*.scap*"))

    if not scap_files:
        raise FileNotFoundError(f"No scap* files found in {SCAP_DIR}")

    print(f"[*] Found {len(scap_files)} files")

    for scap_file in sorted(scap_files):
        # Create output file per input file
        output_file = OUTPUT_DIR / f"{scap_file.name}.json"

        scap_to_json(scap_file, output_file)

    print(f"[*] Done. Output files written to {OUTPUT_DIR}")
