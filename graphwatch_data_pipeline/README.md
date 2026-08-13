# Graphwatch Data Pipeline — Overview

End-to-end flow for turning a recorded scenario into a GRASP experiment: record on the
target host, ship the recording to the GRASP machine, load it into Postgres, configure the time ranges, and run the experiment.

## Pipeline at a glance

1. [Record a scenario](#1-record-a-scenario-on-the-target-host) on the target host with `sysdig`.
2. [Transfer the recording](#2-transfer-the-recording-to-the-grasp-machine) to the GRASP machine.
3. [Set up the database](#3-set-up-the-database) (once per environment).
4. [Process the recording](#4-process-the-recording-into-the-database) into Postgres.
5. [Determine the recorded time range](#5-determine-the-recorded-time-range).
6. [Create an experiment config](#6-create-an-experiment-config) for that time range.
7. [Run the experiment](#7-run-the-experiment).
8. [Review the results](#8-review-the-results).
9. [Optional: Explore reported anomalies with contextualization](#9-optional-explore-reported-anomalies-with-contextualization).

## 1. Record a scenario on the target host

On the target host, capture the syscalls relevant to provenance tracking with `sysdig`:

```bash
sudo sysdig \
  -s 0 \
  -C 1000 \
  -W 200 \
  -w "./sysdig_scaps/sysdig.scap" \
  -q \
  '(evt.type=execve or evt.type=execveat or evt.type=clone or evt.type=fork or evt.type=vfork or evt.type=exit or evt.type=exit_group or evt.type=open or evt.type=openat or evt.type=close or evt.type=read or evt.type=write or evt.type=connect or evt.type=accept or evt.type=accept4 or evt.type=sendto or evt.type=recvfrom) and proc.pid != 1 and proc.name != sysdig'
```

- `-C 1000 -W 200` rotates the capture into ~1000 MB chunks, keeping the last 200 files —
  this is why you'll end up with multiple `sysdig.scap00`, `sysdig.scap01`, ... files
  instead of one large one.
- The event filter is intentionally narrow: it only keeps the syscalls the parsing stage
  in [`sysdig/`](sysdig/) actually understands (process lifecycle, file I/O, network I/O).
- `proc.pid != 1 and proc.name != sysdig` excludes PID 1 and sysdig capturing itself.

Stop the capture (`Ctrl+C`) once the scenario has run.

> **maybe helpful:** Start the capture within a `tmux` session to ensure it continues running after closing the window.
> ```bash
> tmux new-session -d -s sysdig 'sudo sysdig ...'
> ```


## 2. Transfer the recording to the GRASP machine

If the target host and the GRASP machine are different systems, copy the resulting
`sysdig_scaps/` directory over. The pipeline expects it at
`graphwatch_data_pipeline/sysdig/input/sysdig_scaps/`.

**scp:**

```bash
scp -r <remote_user>@<remote_host>:<remote_path>/sysdig_scaps graphwatch_data_pipeline/sysdig/input/
```

**rsync (recommended for larger/repeated transfers — resumable, only re-sends changed data):**

```bash
rsync -avzP -e ssh <remote_user>@<remote_host>:<remote_path>/sysdig_scaps/ graphwatch_data_pipeline/sysdig/input/sysdig_scaps/
```

- `-a` archive mode (recursive, preserves permissions/timestamps)
- `-v` verbose
- `-z` compress in transit
- `-P` equivalent to `--partial --progress`: shows progress and lets an interrupted
  transfer resume instead of restarting from scratch
- Trailing slashes matter: a trailing `/` on the source copies its *contents* into the
  destination, so both sides end up as `.../sysdig/input/sysdig_scaps/sysdig.scap00`, etc.

**Same machine:** skip the transfer entirely — just move (or symlink) the directory into
place:

```bash
mv ./sysdig_scaps graphwatch_data_pipeline/sysdig/input/
```

## 3. Set up the database

One-time setup (or reset) of the Postgres instance the pipeline loads data into — see
[`db_setup/README.md`](db_setup/README.md) for details (Docker Compose, `.env`, start/stop).

## 4. Process the recording into the database

With the scap files in place under `sysdig/input/sysdig_scaps/` and the database running
and reachable, run the sysdig pipeline — see [`sysdig/README.md`](sysdig/README.md) for
prerequisites and the `.env` setup:
Furthermore, ensure the GRASP virtual environment is activated and at least the
`requirements.txt` has been installed.

```bash
cp .env_example .env
cd graphwatch_data_pipeline/sysdig
source .env
./main.py
```

This converts the `.scap` files to JSON, builds the provenance node/edge graph, normalizes
it into GRASP's table format, and loads everything into Postgres (`event_table`,
`subject_node_table`, `file_node_table`, `netflow_node_table`).

## 5. Determine the recorded time range

Experiment configs need explicit `train`/`test` start and end timestamps. Query the
loaded data for the actual recorded range:

```sql
SELECT min(timestamp_rec), max(timestamp_rec) FROM event_table;
```

`timestamp_rec` is a Unix timestamp in **nanoseconds**. Convert it to a human-readable UTC
timestamp with:

```bash
nsdate() {
    local ts="$1"
    date -u -d "@${ts:0:-9}.${ts: -9}" "+%Y-%m-%d %H:%M:%S.%N UTC"
}

nsdate <timestamp_in_nanoseconds> 
```
e.g.
```bash
nsdate 1785762805520602334
```

Run this for both the min and max value to get the usable time window for the recording. 

## 6. Create an experiment config

Create (or copy an existing one as a template, e.g.
[`grasp/experiments/sysdig_spade/experiment_sysdig_1.yaml`](../grasp/experiments/sysdig_spade/experiment_sysdig_1.yaml))
an experiment YAML under `grasp/experiments/`, and fill in the `times:` section using the
human-readable timestamps from step 5:

```yaml
times:
  train_start: ["2026-08-03 13:00:00"]
  train_end: ["2026-08-03 13:59:59"]
  test_start: ["2026-08-03 13:40:00"]
  test_end: ["2026-08-03 13:59:59"]
  context_size: 30 # in minutes
  step_size: 10 # in minutes
```

`train`/`test` windows must fall within the min/max range determined in step 5.

## 7. Run the experiment

From the repository root:

```bash
python main.py --experiment-config grasp/experiments/sysdig_spade/experiment_sysdig_1.yaml
```

Replace `--experiment-config` argument with the path to the config created in step 6.

## 8. Review the results

Results are written under `data/<dataset_name>/reports/` — e.g. `data/sysdig/reports/` for
the `sysdig` dataset — named after the experiment prefix and run parameters:

- `<experiment_prefix>_..._report.json` — a short summary: detection summary, time-window
  breakdown, evaluation hits against the ground truth, and overall graph stats.
- `<experiment_prefix>_..._detailed_report.json` — the detailed report, listing every
  detected anomaly individually so each one can be traced back and further analyzed.

## 9. Optional: Explore reported anomalies with contextualization

The reports from step 8 list *which* nodes were flagged as anomalies, but not *why*.
The PIDS contextualization workbench compares each reported anomaly against sampled
training executables for its predicted and true labels. This should help in determining
whether it is a false positive or a true anomaly. See the
[grasp_contextualization README](../grasp_contextualization/README.md) for the full
workflow; from the repository root:

```bash
uv pip install -r grasp_contextualization/requirements.txt
streamlit run grasp_contextualization/pids_workbench_app.py
```

`--dataset`/`--experiment-prefix`/`--context-size`/`--step-size` must match the
experiment config used in step 6.

