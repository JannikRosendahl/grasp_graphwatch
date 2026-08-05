#!/usr/bin/env python3

import json
import subprocess
from pathlib import Path

# ------------------------------
# CONFIG
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent

SCAP_DIR = Path("output/sysdig_scaps")


# Output directory
OUTPUT_DIR = BASE_DIR / "output" / "json"
OUTPUT_DIR.mkdir(exist_ok=True)


def scap_to_json(scap_file, output_file):
    """
    Converts a single sysdig .scap file to JSON lines using sysdig CLI
    """

    print(f"[*] Processing {scap_file} -> {output_file}")

    cmd = ["sysdig", "-r", str(scap_file), "-j"]

    with open(output_file, "w") as outfile:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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

        if proc.returncode != 0:
            error = proc.stderr.read()  # type: ignore
            raise RuntimeError(f"sysdig failed for {scap_file}:\n{error}")


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
