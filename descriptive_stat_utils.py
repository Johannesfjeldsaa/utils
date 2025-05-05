from typing import Union
import xarray as xr
import numpy as np
from type_check_decorator import type_check_decorator

@type_check_decorator
def get_lat_weights(
    ds: Union[xr.Dataset, xr.DataArray]
):
    """
    Get the weights for the latitude dimension of a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.

    Returns
    -------
    weights : xr.DataArray
        The weights for the latitude dimension.

    Notes
    -----
    The weights are calculated as the cosine of the latitude in radians.

    """

    assert 'lat' in ds.dims, "Latitude is not a dimension in the dataset"

    return np.cos(np.deg2rad(ds.lat))


@type_check_decorator
def calculate_mean(
    ds: Union[xr.Dataset, xr.DataArray],
    dimension: Union[str, list, np.ndarray]
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

    dimension = np.atleast_1d(dimension)

    if 'lat' in dimension:
        weights = get_lat_weights(ds)
        ds = ds.weighted(weights)

    return ds.mean(dimension)

@type_check_decorator
def calculate_sum(
    ds: Union[xr.Dataset, xr.DataArray],
    dimension: Union[str, list, np.ndarray]
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

    dimension = np.atleast_1d(dimension)

    if 'lat' in dimension:
        weights = get_lat_weights(ds)
        ds = ds.weighted(weights)

    return ds.sum(dimension)