import sys
import time
import select
import cftime
import numpy as np
from typing import Union
from datetime import datetime

from utils.type_check_decorator import type_check_decorator

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

@type_check_decorator
def timed_input(
    prompt: str,
    timeout: int = 15,
    default_output: Union[str, None] = None
    ) -> Union[str, None]:
    """Get user input with timeout

    Parameters
    ----------
    prompt : str
        Text to prompt user.
    timeout : int, optional
        Timeout in seconds, by default 15.
    default_output : str, optional
        Default output to return if timeout occurs, by default None.

    Returns
    -------
    Union[str, None]
        User input, or default_output if timeout.
    """
    print(prompt, end='', flush=True)
    start_time = time.time()

    if sys.stdin.isatty():
        # Interactive mode
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.readline().strip()
    else:
        # Non-interactive mode
        while True:
            if time.time() - start_time > timeout:
                break
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.readline().strip()
            time.sleep(0.1)

    print('\nInput timed out')
    return default_output

@type_check_decorator
def get_month_weights(
    timestamps:     Union[str, list, np.ndarray],
    date_format:    Union[str, None] = None,
    time_format:    Union[str, None] = None,
    calendar:       Union[str, None] = None,
):
    """Weights for the month of each timestamp. Returns a mapping (dict) of
    timestamp to month weight.
    The month weight is the number of days in the month divided by the total
    number of days in the year.

    Parameters
    ----------
    timestamps : list or np.array
        List of timestamps in the format <date_format>-<time_format>.
    date_format : str, optional
        The format of the date part of the timestamp. Used for parse_timestamp, default None yields "%Y-%m-%d".
    time_format : str, optional
        The format of the time part of the timestamp. Used for parse_timestamp, default None assumes seconds since midnight or 'sssss'.
    calendar : str, optional
        The calendar type, default None yields 'gregorian'.

    Returns
    -------
    dict
        A dictionary mapping each timestamp to its month weight.
    """
    if isinstance(timestamps, str):
        timestamps = [timestamps]
    elif isinstance(timestamps, np.ndarray):
        timestamps = timestamps.tolist()
    elif not isinstance(timestamps, list):
        raise ValueError("timestamps must be a list, np.array or str")
    if len(timestamps) == 0:
        raise ValueError("timestamps cannot be empty")

    date_format = date_format if date_format is not None else "%Y-%m-%d"
    time_format = time_format if time_format is not None else "sssss"
    calendar = calendar if calendar is not None else "gregorian"

    # get the month weights by calculating days-in-timestamp-month/days-in-timestamp-year
    month_weights = {}
    for timestamp in timestamps:
        if isinstance(timestamp, cftime.datetime):
           # check if timestamp has the same calendar as the one provided
           if timestamp.calendar != calendar:
                raise ValueError(f"Timestamp {timestamp} has calendar {timestamp.calendar}, but {calendar} was provided")
           dt = timestamp
        else:
            dt = parse_timestamp(
                timestamp,
                date_format=date_format,
                time_format=time_format,
                calendar=calendar
            )
        month = dt.month
        year = dt.year
        # Calculate the number of days in the month and year
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1

        # Start of the month
        start_of_month = cftime.datetime(year, month, 1, calendar=calendar)
        # Start of the next month
        start_of_next_month = cftime.datetime(next_year, next_month, 1, calendar=calendar)

        # Days in the month
        month_days = (start_of_next_month - start_of_month).days

        # Days in the year (from Jan 1 to Jan 1 of the next year)
        start_of_year = cftime.datetime(year, 1, 1, calendar=calendar)
        start_of_next_year = cftime.datetime(year + 1, 1, 1, calendar=calendar)
        year_days = (start_of_next_year - start_of_year).days

        month_weight = month_days / year_days
        month_weights[timestamp] = month_weight

    return month_weights
