import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from pathlib import Path
import json
from grasp.utils import time_helpers


def visualize_misclassification_timeframes(
    detailed_report_path: str,
    output_path: Path,
    attack_windows=None,
    min_duration_seconds: int = 0,
    to_display: bool = False,
    x_tick_count: int = 8,
):
    # --- Constants (publication-ready setup) ---
    BASE_COLOR = "#1a1a1a"  # dark gray bars
    ATTACK_COLOR = "#d62728"  # red overlay
    ATTACK_ALPHA = 0.5  # transparency
    FIGSIZE = (7, 3.5)
    DPI = 300

    # --- Matplotlib style ---
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # --- Helpers ---
    def _parse_time(ts):
        try:
            return pd.to_datetime(ts)
        except Exception:
            return None

    def _collect_attack_intervals(windows=None):
        if not windows:
            return []
        intervals = []
        for w in windows:
            try:
                start, end = (
                    pd.to_datetime(w["start_time"]),
                    pd.to_datetime(w["end_time"]),
                )
                if pd.notnull(start) and pd.notnull(end):
                    if end < start:
                        start, end = end, start
                    intervals.append((start, end))
            except Exception:
                continue
        return intervals

    def _interval_count_series(
        starts: pd.Series, ends: pd.Series
    ) -> pd.DataFrame:
        events = []
        for s, e in zip(starts, ends):
            if s is None or e is None:
                continue
            if s == e:
                e += pd.Timedelta(seconds=1)
            events.append((s, 1))
            events.append((e, -1))
        if not events:
            return pd.DataFrame(
                columns=["t_from", "t_to", "count", "duration_s"]
            )

        events.sort(key=lambda x: x[0])
        segments, active = [], 0
        for i in range(len(events) - 1):
            t, delta = events[i]
            active += delta
            t_next = events[i + 1][0]
            if t_next > t and active > 0:
                segments.append(
                    {
                        "t_from": t,
                        "t_to": t_next,
                        "count": active,
                        "duration_s": (t_next - t).total_seconds(),
                    }
                )
        return pd.DataFrame(segments)

    def _configure_time_axis(
        ax, tmin: pd.Timestamp, tmax: pd.Timestamp, tick_count: int
    ):
        span = (tmax - tmin).total_seconds()

        # Choose date format based on total range
        if span > 14 * 86400:
            fmt = "%Y-%m-%d"
        elif span > 3 * 86400:
            fmt = "%m-%d"
        elif span > 86400:
            fmt = "%d. %H:%M"
        elif span > 3600:
            fmt = "%H:%M"
        else:
            fmt = "%H:%M:%S"

        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
        locator = mdates.AutoDateLocator(maxticks=tick_count)
        ax.xaxis.set_major_locator(locator)

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(alpha=0.3, axis="y")

    def _shade_attack_intervals(ax, intervals):
        for start, end in intervals:
            ax.axvspan(
                mdates.date2num(start),
                mdates.date2num(end),
                facecolor=ATTACK_COLOR,
                alpha=ATTACK_ALPHA,
                zorder=5,
            )

    # --- Data Preparation ---
    with open(detailed_report_path, "r") as f:
        dr = json.load(f)

    df = pd.DataFrame(dr.get("detailed_detection").get("anomalies"))
    df[["start_ts_ns", "end_ts_ns"]] = (
        df["time_window"]
        .str.extract(r".*_(\d+)_to_(\d+)_extended\.pt$")
        .astype("int64")
    )
    df["tf_start_ts"] = df["start_ts_ns"].apply(
        time_helpers.ns_time_to_datetime_US_reverse
    )
    df["tf_end_ts"] = df["end_ts_ns"].apply(
        time_helpers.ns_time_to_datetime_US_reverse
    )

    df["start_ts"] = df["tf_start_ts"].apply(_parse_time)
    df["end_ts"] = df["tf_end_ts"].apply(_parse_time)
    df = df.dropna(subset=["start_ts", "end_ts"]).copy()

    swap_mask = df["end_ts"] < df["start_ts"]
    df.loc[swap_mask, ["start_ts", "end_ts"]] = df.loc[
        swap_mask, ["end_ts", "start_ts"]
    ].values

    if min_duration_seconds > 0:
        short_mask = (
            df["end_ts"] - df["start_ts"]
        ).dt.total_seconds() < min_duration_seconds  # type: ignore
        df.loc[short_mask, "end_ts"] = df.loc[
            short_mask, "start_ts"
        ] + pd.to_timedelta(min_duration_seconds, unit="s")  # type: ignore

    # --- Visualization ---
    attack_intervals = _collect_attack_intervals(attack_windows)
    segments = _interval_count_series(df["start_ts"], df["end_ts"])
    if segments.empty:
        return []

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x_left = mdates.date2num(segments["t_from"].tolist())
    widths = mdates.date2num(segments["t_to"]) - x_left

    ax.bar(
        x_left,
        segments["count"],
        width=widths,
        align="edge",
        color=BASE_COLOR,
        edgecolor="white",
        linewidth=0.2,
        zorder=2,
    )

    _shade_attack_intervals(ax, attack_intervals)

    ax.set_ylabel("# Misclassifications", labelpad=8)
    ax.set_xlabel("Time", labelpad=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    _configure_time_axis(
        ax, segments["t_from"].min(), segments["t_to"].max(), x_tick_count
    )

    if attack_intervals:
        ax.legend(
            [mpatches.Patch(color=ATTACK_COLOR, alpha=ATTACK_ALPHA)],
            ["Attack Window"],
            frameon=False,
            loc="upper right",
        )

    fig.tight_layout(pad=0.5)

    tf_dir: Path = output_path
    tf_dir.mkdir(parents=True, exist_ok=True)

    img_path = tf_dir / f"{tf_dir.name}_timeline.png"
    pdf_path = tf_dir / f"{tf_dir.name}_timeline.pdf"

    fig.savefig(img_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

    if to_display:
        plt.show()
    else:
        plt.close(fig)

    if to_display:
        plt.show()
    else:
        plt.close(fig)
