# 
import cftime
from typing import Union
from datetime import datetime, timedelta
from type_check_decorator import type_check_decorator

@type_check_decorator
def parse_timestamp(
    timestamp:      str,
    date_format:    Union[str, None] = None,
    time_format:    Union[str, None] = None,
    calendar:       str = None
):
    """
    Parses a timestamp in the format to get a datetime object.

    Parameters
    ----------
    timestamp : str
        The timestamp string in the format <date_format>-<time_format>.
    date_format : str, optional
        The format of the date part of the timestamp. Default None yields "%Y-%m-%d".
    time_format : str, optional
        The format of the time part of the timestamp. Default None assumes seconds since midnight.

    Returns
    -------
    datetime
        A datetime object representing the parsed timestamp.

    Raises
    ------
    ValueError
        If the timestamp cannot be parsed using the provided formats.
    """

    date_format = date_format if date_format is not None else "%Y-%m-%d"
    calendar = calendar if calendar is not None else "gregorian"

    if time_format is None or time_format == 'sssss':
        # Default: treat as seconds since midnight
        date_part, seconds_since_midnight = timestamp.rsplit('-', 1)
        dt = datetime.strptime(date_part, date_format)
        total_seconds = int(seconds_since_midnight)
        hour, remainder = divmod(total_seconds, 3600)
        minute, second = divmod(remainder, 60)

        return cftime.datetime(dt.year, dt.month, dt.day, hour, minute, second, calendar=calendar)
    else:
        # Use the provided time format
        full_format = f"{date_format}-{time_format}"

        dt = datetime.strptime(timestamp, full_format)
        return cftime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, calendar=calendar)

@type_check_decorator
def calculate_seconds_diff(
    timestamp1:     str,
    timestamp2:     str,
    date_format:    str = None,
    time_format:    str = None,
    calendar:       str = None
):
    """
    Calculates the difference in seconds between two timestamps.

    Parameters
    ----------
    timestamp1 : str
        The first timestamp string
    timestamp2 : str
        The second timestamp string
    date_format : str, optional
        The format of the date part of the timestamp. Used for parse_timestamp, default None yields "%Y-%m-%d".
    time_format : str, optional
        The format of the time part of the timestamp. Used for parse_timestamp, default None assumes seconds since midnight.

    Returns
    -------
    int
        The absolute difference in seconds between the two timestamps.
    """

    dt1 = parse_timestamp(
        timestamp1,
        date_format=date_format,
        time_format=time_format,
        calendar=calendar
    )
    dt2 = parse_timestamp(
        timestamp2,
        date_format=date_format,
        time_format=time_format,
        calendar=calendar
    )

    # Calculate the difference in seconds
    diff = abs((dt2 - dt1).total_seconds())
    return int(diff)