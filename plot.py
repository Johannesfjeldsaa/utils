
import cmocean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np
import xarray as xr
from typing import Union
from descriptive_stat_utils import calculate_sum


def plot_datasets_side_by_side(
    ds_inmodel: xr.DataArray,
    ds_control: xr.DataArray,
    ds_diff: xr.DataArray,
    var_name: Union[str, list, None] = None,
    title: Union[list, None] = None,
    diff_on_last: bool = True
):
    """
    Plot three maps side by side: a model dataset, a control dataset, and their difference,
    using consistent and robust colorscales for easier comparison.

    Parameters
    ----------
    ds_inmodel : xr.DataArray
        The dataset representing the model simulation to be plotted.
    ds_control : xr.DataArray
        The dataset representing the control simulation or reference dataset.
    ds_diff : xr.DataArray
        The precomputed difference between `ds_inmodel` and `ds_control`.
    var_name : Union[str, list, None], optional
        The name(s) of the variable(s) being plotted, used for automatic plot titles.
        If a list, should contain names for inmodel and control datasets.
        If None, titles will be generic.
    title : Union[list, None], optional
        Custom titles for the three plots: [inmodel, control, difference].
        If a single string is provided, it's applied to all plots.
        If None, titles will be generated from `var_name`.
    diff_on_last : bool, optional
        If True, assumes the difference is shown in the last (third) plot and applies
        diverging or sequential colormaps accordingly. Default is True.
    """


    if title is None:
        if isinstance(var_name, list):
            inmodel_var_name = var_name[0]
            control_var_name = var_name[1]
        else:
            inmodel_var_name = var_name
            control_var_name = f'{var_name} control'
        title_inmodel = f'{inmodel_var_name} in model'
        title_control = f'{control_var_name} control calculation'
        title_diff = f'Difference ({inmodel_var_name} - {control_var_name})'
    else:
        if isinstance(title, list):
            title_inmodel, title_control, title_diff = title
        else:
            title_inmodel = title_control = title_diff = title

    if 'time' in ds_inmodel.dims:
        ds_inmodel = ds_inmodel.isel(time=0)
    if 'time' in ds_control.dims:
        ds_control = ds_control.isel(time=0)
    if 'time' in ds_diff.dims:
        ds_diff = ds_diff.isel(time=0)


    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Combine both for consistent scaling
    combined = xr.concat([ds_inmodel, ds_control], dim='z')
    vmin_combined = float(combined.min())
    vmax_combined = float(combined.max())

    # Plot Inmodel
    im0 = ds_inmodel.plot(
        ax=axs[0],
        cmap=cmocean.cm.haline,
        vmin=vmin_combined,
        vmax=vmax_combined,
        add_colorbar=False
    )
    axs[0].set_title(title_inmodel)
    axs[0].add_feature(cfeature.COASTLINE)
    plt.colorbar(im0, ax=axs[0], orientation='horizontal', pad=0.05)

    # Plot Control
    im1 = ds_control.plot(
        ax=axs[1],
        cmap=cmocean.cm.haline,
        vmin=vmin_combined,
        vmax=vmax_combined,
        add_colorbar=False
    )
    axs[1].set_title(title_control)
    axs[1].add_feature(cfeature.COASTLINE)
    plt.colorbar(im1, ax=axs[1], orientation='horizontal', pad=0.05)

    # Plot Difference
    cmapkwargs = {}
    diff = ds_diff
    vmin = float(diff.min())
    vmax = float(diff.max())

    if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
        cmap = cmocean.cm.curl
        norm = None
    elif vmin < 0 < vmax:
        cmap = cmocean.cm.curl
        norm = None#matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    elif vmax <= 0:
        cmap = cmocean.cm.tempo_r
        norm = None
    else:
        cmap = cmocean.cm.tempo
        norm = None

    cmapkwargs.update({'cmap': cmap, 'norm': norm})

    im2 = diff.plot(
        ax=axs[2],
        add_colorbar=False,
        **cmapkwargs
    )
    axs[2].set_title(title_diff)
    axs[2].add_feature(cfeature.COASTLINE)
    plt.colorbar(im2, ax=axs[2], orientation='horizontal', pad=0.05)

    # Annotate with total difference
    total_diff = calculate_sum(ds_diff,['lat', 'lon'])
    axs[2].text(
        x=0.5,
        y=-0.15,
        s=f'Total difference: {total_diff:.2e}',
        transform=axs[2].transAxes,
        ha='center', va='top', fontsize=12
    )

    plt.tight_layout()
    plt.savefig('budget.png', dpi=300)
