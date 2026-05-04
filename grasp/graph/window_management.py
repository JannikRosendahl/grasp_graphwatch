from datetime import datetime, timedelta
from grasp.utils.time_helpers import parse, fmt


def create_train_and_test_windows(
    train_start_times: list[str],
    train_end_times: list[str],
    test_start_times: list[str],
    test_end_times: list[str],
    context_size: int,
    step_size: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    train_windows: list[tuple[str, str]] = _create_window_times(
        train_start_times, train_end_times, context_size, step_size
    )
    test_windows: list[tuple[str, str]] = _create_window_times(
        test_start_times, test_end_times, context_size, step_size
    )
    return train_windows, test_windows


def _create_window_times(
    start_times: list[str],
    end_times: list[str],
    context_size: int,
    step_size: int,
) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    for start_time, end_time in zip(start_times, end_times):
        sliced_windows: list[tuple[str, str]] = _slice_time_range(
            start_time, end_time, context_size, step_size
        )
        windows.extend(sliced_windows)
    return windows


def _slice_time_range(
    start_time: str, end_time: str, context_size: int, step_size: int
) -> list[tuple[str, str]]:
    start: datetime = parse(start_time)
    end: datetime = parse(end_time)
    if end <= start:
        return []

    win = timedelta(minutes=context_size)
    stride = timedelta(minutes=step_size if step_size > 0 else context_size)

    windows: list[tuple[str, str]] = []
    cur: datetime = start
    while cur < end:
        w_end = min(cur + win, end)
        windows.append((fmt(cur), fmt(w_end)))
        cur += stride
    return windows
