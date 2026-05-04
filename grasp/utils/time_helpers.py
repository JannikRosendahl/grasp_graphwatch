import pytz
from datetime import datetime
from time import mktime
import time

from grasp.schema import TimeStringFormat


def datetime_to_ns_time_US(date: str) -> int:
    tz = pytz.timezone("US/Eastern")
    timeArray: time.struct_time = time.strptime(
        date, TimeStringFormat.FMT.value
    )
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)


def ns_time_to_datetime_US_reverse(ns_time: int) -> str:
    tz = pytz.timezone("US/Eastern")
    timestamp = ns_time / 1_000_000_000  # Convert nanoseconds to seconds
    dt = datetime.fromtimestamp(timestamp, tz)
    return dt.strftime(TimeStringFormat.FMT.value)


def parse(ts: str) -> datetime:
    return datetime.strptime(ts, TimeStringFormat.FMT.value)


def fmt(dt: datetime) -> str:
    return dt.strftime(TimeStringFormat.FMT.value)
