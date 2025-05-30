import cftime
import xarray as xr
import numpy as np
from typing import Union
from type_check_decorator import type_check_decorator

@type_check_decorator
def get_lat_weights(
    ds: Union[xr.Dataset, xr.DataArray]
):
    """
    Get the weights for the latitude dimension of a dataset.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The input dataset or data array.
        The dataset must have a 'lat' dimension.
        If a DataArray is provided, the coordinate must be lat.

    Returns
    -------
    xr.DataArray
        The latitude weights.

    Notes
    -----
    The weights are calculated as the cosine of the latitude in radians.

    """

    if 'lat' not in ds.dims:
        raise ValueError("Latitude is not a dimension in the dataset")

    lat_da = ds.lat if isinstance(ds, xr.Dataset) else ds

    return np.cos(np.deg2rad(lat_da))

def get_month_weights(
    ds: xr.Dataset,
    calendar: str = "gregorian",
    time_dim: str = "time"
) -> xr.DataArray:
    """
    Get the weights for the months of the time dimension in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset. The dataset must have a 'time' dimension.
    calendar : str, optional
        The calendar type, default is 'gregorian'.
    time_dim : str, optional
        The name of the time dimension, default is 'time'.

    Returns
    -------
    xr.DataArray
        The month weights as an xarray.DataArray, with the time dimension as coordinates.

    Notes
    -----
    The month weights are calculated as the number of days in the month divided
    by the total number of days in the year, based on the provided calendar.
    """

    if time_dim not in ds.dims:
        raise ValueError(f"'{time_dim}' is not a dimension in the dataset")

    # Extract the time coordinate
    time_da = ds[time_dim]

    # Validate that time_da contains cftime.datetime objects
    if not np.issubdtype(time_da.dtype, np.datetime64) and not isinstance(
        time_da.values[0], cftime.datetime
    ):
        raise ValueError(f"The '{time_dim}' dimension must contain datetime objects")

    # Calculate month weights
    weights = []
    for timestamp in time_da.values:
        dt = timestamp  # Directly assign, since we've validated earlier
        if dt.calendar != calendar:
            raise ValueError(f"Timestamp {dt} has calendar {dt.calendar}, but {calendar} was provided")

        month = dt.month
        year = dt.year

        # Calculate the number of days in the month and year
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1

        start_of_month = cftime.datetime(year, month, 1, calendar=calendar)
        start_of_next_month = cftime.datetime(next_year, next_month, 1, calendar=calendar)

        month_days = (start_of_next_month - start_of_month).days
        start_of_year = cftime.datetime(year, 1, 1, calendar=calendar)
        start_of_next_year = cftime.datetime(year + 1, 1, 1, calendar=calendar)

        year_days = (start_of_next_year - start_of_year).days
        weights.append(month_days / year_days)

    # Convert weights into an xarray.DataArray
    month_weights = xr.DataArray(
        weights,
        coords={time_dim: time_da},
        dims=[time_dim],
        name="month_weights"
    )

    return month_weights

def calculate_mean(
    ds:             Union[xr.Dataset, xr.DataArray],
    dimension:      Union[str, list],
    time_weights:   Union[xr.DataArray, None] = None
):
    """
    Calculate the mean of a dataset along specified dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    dimension : str, list or array
        The dimension(s) along which to calculate the mean.

    Returns
    -------
    mean : xr.DataArray
        The mean of the dataset along the specified dimensions.

    Notes
    -----
    If 'lat' is included in the dimensions, latitude weights are applied before calculating the mean.

    """

    # remove variable that are not compatible with averaging
    for var in ['time_bnds', 'date_written', 'time_written']:
        ds = ds.drop_vars(var)
    # make sure the dimension is at least 1D
    dimension = np.atleast_1d(dimension)
    # make sure the dimensions are in the dataset
    for dim in dimension:
        if dim not in ds.sizes:
            raise ValueError(f"Dimension '{dim}' is not in the dataset")

    weights = None

    if 'lat' in dimension:
        weights = get_lat_weights(ds) if weights is None else weights * get_lat_weights(ds)
    if 'time' in dimension:
        # Check if time_weights is provided
        if time_weights is None:
            raise ValueError("Time weights must be provided if 'time' is in the dimensions")
        # check if time_weights is the same length as the time dimension
        if len(time_weights) != ds.sizes['time']:
            raise ValueError("Time weights must be the same length as the time dimension")
        weights = time_weights if weights is None else weights * time_weights

    if weights is not None:
        # make sure the weights are aligned with the averaging dimension
        needed_dims = [dim for dim in dimension if dim not in weights.sizes]
        for dim in needed_dims:
            # make a weighted DataArray with the same shape as the dimension but only ones
            needed_weight = xr.DataArray(
                np.ones(ds.sizes[dim]),
                dims=[dim],
                coords={dim: ds[dim].values}
            )
            weights = weights * needed_weight

        weighted_ds = ds.weighted(weights)
        # Use weighted.mean with all dimensions at once
        mean_weighted = weighted_ds.mean(dim=weights.dims)
        mean = mean_weighted.mean(dim=[dim for dim in dimension if dim not in weights.dims])
    else:
        # No weights, just plain mean
        mean = ds.mean(dim=dimension)
    return mean

@type_check_decorator
def calculate_sum(
    ds: Union[xr.Dataset, xr.DataArray],
    dimension: Union[str, list, np.ndarray],
    area_weights,
):
    """
    Calculate the sum of a dataset along specified dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    dimension : str, list or array
        The dimension(s) along which to calculate the sum.

    Returns
    -------
    sum : xr.DataArray
        The sum of the dataset along the specified dimensions.

    Notes
    -----
    If 'lat' is included in the dimensions, latitude weights are applied before calculating the sum.

    """
    valid_time_info = {
        'calendar': [attr.replace('Datetime', '') for attr in dir(cftime) if attr.startswith("Datetime")],
        'wanted_weights': ['month-in-year']
    }
    dimension = np.atleast_1d(dimension)

    if all([dim in dimension for dim in ['lat', 'lon']]):
        if area_weights is None:
            raise ValueError("Area weights must be provided if both 'lat' and 'lon' are in the dimensions")
        ds = ds.weighted(area_weights)

    #ds = earth_radius * (2 * np.pi * earth_radius * ) / len(ds.lon.values)
    return ds.sum(dimension)