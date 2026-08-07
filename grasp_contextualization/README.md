# GRASP Contextualization — PIDS Explainability Tools

Contextualizes anomalies reported by GRASP against sampled training examples,
so an analyst has a first-pass basis to decide whether a reported anomaly is a
false positive or a real one. For each detected anomaly in an experiment run,
`pids_analysis_engine.py` samples training executables for both the model's
**predicted label** and the **true label**, computes a distance metric between
the anomaly's neighborhood and each sample, and checks which label the
anomaly's neighborhood actually looks closer to — plus a leave-one-view-out
ablation showing which feature view drove that call.
`pids_workbench_app.py` is a Streamlit UI to run that engine and visualize its
output for an analyst.

## Setup

1. Make sure the repo root's dependencies are installed (`pip install -r
   requirements.txt` from the repo root, including the `torch` /
   `torch_geometric` install steps documented there).
2. Install the additional dependencies these two tools need:

    ```bash
    pip install -r grasp_contextualization/requirements.txt
    ```

Both scripts locate the repo automatically (they resolve their own path), so
no `PYTHONPATH`/`GRASP_PROJECT_ROOT` setup is required when the repo is used
as-is. `GRASP_PROJECT_ROOT` can still be set to override this if you run
either tool against a different checkout.

## 1. Run the analysis engine

Run against a completed GRASP experiment (i.e. after `python main.py
--experiment-config ...` has produced a `..._detailed_report.json` under
`data/<dataset>/reports/` — see [graphwatch_data_pipeline/README.md](../graphwatch_data_pipeline/README.md)
steps 7-8 for how that gets created).

From the repo root:

```bash
python grasp_contextualization/pids_analysis_engine.py \
  --data-dir . \
  --dataset sysdig \
  --experiment-prefix sysdig_1 \
  --context-size 30 \
  --step-size 10 \
  --output-dir pids_runs/sysdig_1_report
```

- `--data-dir` is the directory containing `data/<dataset>/...` (the repo root, if you followed the pipeline's default paths).
- `--dataset`, `--experiment-prefix`, `--context-size`, `--step-size` must match the experiment config that produced the report (`--experiment-prefix` is the config's `experiment_prefix` field, e.g. `sysdig_1`).
- `--output-dir` is where the CSV/JSON report is written (defaults to `./pids_simple_metric_visual_report`).

Other useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--candidate-pool-per-label` | 50 | Max training samples considered per true/predicted label |
| `--max-anomalies` | all | Cap the number of anomalies analyzed (useful for a quick test run) |
| `--hop-mode` | `two` | `one` or `two`-hop neighborhoods |
| `--include-unknown-exec` | off | Include anomalies caused by unresolved/unknown executables |
| `--fail-fast` | off | Stop on the first anomaly that errors instead of logging and continuing |

This writes ~13 CSV files plus `run_config.json` and `pyg_distance_metric.json`
into `--output-dir`.

## 2. Explore the results in the workbench

From the repo root:

```bash
streamlit run grasp_contextualization/pids_workbench_app.py
```

Open the printed local URL in a browser. Pages, in the sidebar:

- **Create report** — runs the engine from the UI (fills in the same flags as above) instead of the CLI. Its form defaults to a smaller candidate pool (5 per label) and no anomaly cap, geared towards quick, interactive runs rather than the CLI's defaults.
- **Overview** — aggregate metrics across all anomalies: conclusions, per-view score distributions, ablation summary, per-anomaly classifier quality (learning quality), prediction confidence, and any failures.
- **Context View** — the analyst's decision view for one anomaly at a time: TARGET vs. a predicted-label training sample vs. a true-label training sample, with side-by-side graphs, event sequences, tables, and diffs. Both samples default to the closest match but each has its own dropdown to pick a different ranked candidate instead. Also shows the target node's own raw classifier confidence (top-3). This is the primary page for deciding false positive vs. real anomaly. If one side has no training samples at all (most commonly the true-label side, when that true class was never seen in training), the page degrades gracefully — it shows whatever side *is* available instead of failing outright, and only errors if neither side has anything to compare against.

**Prediction confidence** (`prediction_confidence.csv`, shown in both Overview and Context View): the classifier's raw top-3 softmax output for each anomaly's own node, read directly from that experiment's `test_cls_storage.pt`. This is the **raw, uncorrected** model output — it is *not* the same as `pred_label` shown elsewhere in the report, which is GRASP's cluster-corrected prediction (see `grasp/evaluation/evaluation_storage.py`). The two can legitimately disagree; each row is flagged with whether it matches the reported `pred_label`, so a disagreement reads as "cluster correction overrode the raw model here" rather than looking like a bug. Each row also flags `true_label_known_in_training` — when false, the classifier never saw that true class during training at all, so it had no way to predict it correctly (this is common: in a typical sysdig run, roughly a third of anomalies have a true class unseen in training).

**Anomaly occurrences:** the same `anomaly_id` (a process's `node_index_id`) can be detected in multiple overlapping sliding time-windows, each with its own independent classification — sometimes with a different predicted label per window. Context View shows an "Occurrence (time window)" picker whenever an anomaly_id has more than one; every downstream value (target graph, prediction/true samples, confidence) is scoped to whichever occurrence is selected, so nothing gets silently mixed across occurrences.

Each page's sidebar has a "Report search roots" box (also settable via the
`PIDS_REPORT_ROOTS` env var, colon-separated) — it auto-discovers report
directories produced by step 1 under those roots.

## Notes

- The engine and workbench must agree on dataset/context/step-size naming — both resolve files as `<prefix>_dataset-<dataset>_context_size-<c>_step_size-<s>_<suffix>`, matching what `main.py`'s experiment run produces.
- Large EDR-style datasets (name containing `e5`, `optc`, or `carbanak`) automatically get a capped two-hop neighborhood (`[10000, 20]`) instead of the full `[-1, -1]` fanout, to avoid exploding neighborhoods — see `grasp_contextualization/shared.py`.
