import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union
from tabulate import tabulate

sys.path.append(str(
    (Path(__file__).resolve()).joinpath(os.pardir, 'utils').resolve()
))
from plot import plot_datasets_side_by_side
from time_utils import calculate_seconds_diff
from descriptive_stat_utils import calculate_sum
from columnintegration import make_columnintegrater
from type_check_decorator import type_check_decorator

class CalcBudget:

    @type_check_decorator
    def calc_load(
        self,
        ds:     xr.Dataset,
    ):
        """Calculate the load of the species in the dataset.
        The load is calculated by integrating the species over the vertical dimension.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset containing the species to be integrated.

        Returns
        -------
        ds : xr.Dataset
            The input dataset with the load of the species added as a new variable. e.g. cb_OM, cb_BC, etc.
        """
        for var in self.load_species:
            # create integrater
            integrater = make_columnintegrater.create_columnintegrater(
                var=var,
                unit='kg kg-1',
                field=ds[var],
                hyai=ds['hyai'],
                hybi=ds['hybi'],
                p0=ds['P0'],
                ps=ds['PS']
            )
            # integrate and add load to dataset
            ds[f'cb_{var}'] = integrater.integrate()

        ds[f'cb_{self.species}'] = ds[[f'cb_{var}' for var in self.load_species]].to_array().sum(dim='variable')
        return ds

    @type_check_decorator
    def calc_load_delta(
        self,
        start_conditions_ds:    xr.Dataset,
        end_conditions_ds:      xr.Dataset,
        grid_cell_area:         xr.DataArray,
        verbose:                bool = False,
        plot:                   bool = False
    ):
        """Calculate the load delta of the species in the dataset.
        The load delta is calculated by integrating the species over the vertical dimension and then calculating the difference between the start and end conditions.
        The load delta is the difference between the load at the end of the simulation and the load at the start of the simulation.

        Parameters
        ----------
        start_conditions_ds : xr.Dataset
            Dataset containing the start conditions for the simulation. Must contain the concentration of the species to be integrated.
        end_conditions_ds : xr.Dataset
            Dataset containing the end conditions for the simulation. Must contain the concentration of the species to be integrated.
        verbose : bool, optional
            If True, print the load delta for the species, by default False
        plot : bool, optional
            If True, plot the load delta for the species, by default False
            The plot will show the load at the start and end conditions, as well as the difference between the two.

        Returns
        -------
        float
            The load delta of the species in the dataset. The load delta is the difference between the load at the end of the simulation and the load at the start of the simulation.
        """

        datasets = {
            'start_conditions_ds':start_conditions_ds,
            'end_conditions_ds': end_conditions_ds
        }
        # check that all variables are present
        for name, ds in datasets.items():
            for var in self.load_species:
                if var not in ds.data_vars:
                    raise ValueError(f"Variable {var} not found in {name}")
                else:
                    # check that the variable is a DataArray
                    if not isinstance(ds[var], xr.DataArray):
                        raise ValueError(f"Variable {var} is not a DataArray in {name}")
                    # check that the variable is 3D
                    if start_conditions_ds[var].ndim != 3:
                        raise ValueError(f"Variable {var} is not 3D in {name}")
            # check that the area used for the integration is the same lat/lon grid as the datasets
            if any(grid_cell_area.lat.values != ds.lat.values) or any(grid_cell_area.lon.values != ds.lon.values):
                raise ValueError(f"Area used for the integration is not the same lat/lon grid as that in {name}")

        # check that the datasets have the same dimensions
        sizes_to_check = ['lat', 'lon', 'lev', 'ilev']
        start_cond_sizes = {key: value for key, value in dict(start_conditions_ds.sizes).items() if key in sizes_to_check}
        end_cond_sizes = {key: value for key, value in dict(end_conditions_ds.sizes).items() if key in sizes_to_check}
        if start_cond_sizes != end_cond_sizes:
            raise ValueError(
                f"Datasets have different dimensions \n",
                f"end_conditions_ds.sizes: {end_cond_sizes} \n",
                f"start_conditions_ds.sizes: {start_cond_sizes}",
            )

        # check that the datasets have the same coordinates
        for coord in [coord for coord in start_conditions_ds.coords if coord != 'time']:
            if coord not in end_conditions_ds.coords:
                raise ValueError(f"Coordinate {coord} not found in end_conditions_ds")
            else:
                # check that the coordinate is the same
                if not (start_conditions_ds[coord] == end_conditions_ds[coord]).all():
                    raise ValueError(f"Coordinate {coord} is not the same in both datasets")

        # we store the loads as a new variable in the dataset
        for name, ds in datasets.items():
            ds = self.calc_load(ds)

        # calculate the delta
        glob_load_start = float(
            (start_conditions_ds[f'cb_{self.species}'] * grid_cell_area).sum(dim=['lat', 'lon'])
        )
        glob_load_end = float(
            (end_conditions_ds[f'cb_{self.species}'] * grid_cell_area).sum(dim=['lat', 'lon'])
        )
        load_delta: float = glob_load_end - glob_load_start

        if verbose:
            print(f"Load delta for {self.species} is {glob_load_end:.4e} - {glob_load_start:.4e} = {load_delta:.4e} kg")

        if plot:
            plot_datasets_side_by_side(
                end_conditions_ds[f'cb_{self.species}'],
                start_conditions_ds[f'cb_{self.species}'],
                end_conditions_ds[f'cb_{self.species}'] - start_conditions_ds[f'cb_{self.species}'],
                var_name=f'Load {self.species}',
                title=[
                    f'Load {self.species} in end conditions',
                    f'Load {self.species} in start conditions',
                    f'Load {self.species} difference'
                ]
            )

        return load_delta

    @type_check_decorator
    def calc_budget(
        self,
        ds:                     xr.Dataset,
        time_diff_in_seconds:   int,
        start_conditions_ds:    Union[xr.Dataset, None] = None,
        end_conditions_ds:      Union[xr.Dataset, None] = None,
        verbose:                bool = False,
        plot_load_delta:        bool = False
    ):
        """Calculate the budget for the species in the dataset.
        The budget is calculated by finding the difference between the source and sink of the species.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset containing the species source and sink fields as well as the 'AREA' field.
        time_diff_in_seconds : int
            The time difference between the start and end conditions in seconds.
        start_conditions_ds : Union[xr.Dataset, None], optional
            Dataset containing the start conditions for the simulation.
            Must contain the concentration of the species that make up the load_species.
            If None, the load delta will not be calculated, by default None
        end_conditions_ds : Union[xr.Dataset, None], optional
            Dataset containing the end conditions for the simulation.
            Must contain the concentration of the species that make up the load_species.
            If None, the load delta will not be calculated, by default None
        verbose : bool, optional
            Controls the verbosity of the function, by default False
        plot_load_delta : bool, optional
            If True, plot the load delta for the species, by default False
        """


        # check that all required fields are in ds
        for field in list(self.source_fields.keys()) + list(self.sink_fields.keys()) + ['AREA']:
            if field not in ds:
                raise ValueError(f'Field {field} not in dataset')
            else:
                if verbose:
                    print(f'Found field {field} in dataset')
        # check that weather or not we can account for change in load
        if start_conditions_ds is None or end_conditions_ds is None :
            warnings.warn(
                'No start or end conditions provided. Budget will be calculated without acounting for change in load.',
                UserWarning
            )
        else:
            for find_load_delta_ds in [start_conditions_ds, end_conditions_ds]:
                if 'time' in find_load_delta_ds.sizes:
                    raise ValueError('Start and end conditions datasets should not have a time dimension, please provide a single time step.')
            self.load_delta = self.calc_load_delta(
                start_conditions_ds,
                end_conditions_ds,
                grid_cell_area=ds['AREA'],
                verbose=verbose,
                plot=plot_load_delta
            )

        # update the source_fields/sink_fields dictionary with the global average values from the dataset
        # then calculate the source and sink
        self.source = 0.0
        for field in self.source_fields.keys():
            self.source_fields[field]['value'] = float(
                (ds[field] * ds['AREA'] * time_diff_in_seconds).sum(dim=['lat', 'lon'])
            )
            self.source_fields[field]['unit'] = self.unit_after_integration
            self.source += self.source_fields[field]['value']

        self.sink = 0.0
        for field in self.sink_fields.keys():
            self.sink_fields[field]['value'] = float(
                (ds[field] * ds['AREA'] * time_diff_in_seconds).sum(dim=['lat', 'lon'])
            )
            self.sink_fields[field]['unit'] = self.unit_after_integration
            self.sink += self.sink_fields[field]['value']

        # calculate the balance
        self.balance = self.source - self.sink

        if self.load_delta is not np.nan:
            self.unexplained_error = self.source -  self.sink - self.load_delta

        self.update_overview()

    @type_check_decorator
    def update_overview(self):
        """update the overview dataframe with the source and sink fields.
        The overview dataframe is a pandas dataframe that contains the source and sink fields for the species.
        """

        if self.overview is None:
            self.overview = pd.DataFrame(columns=['Description', 'field', 'value', 'unit'])

        # add/update source fields
        for field_category in ["source_fields", "sink_fields"]:
            for field in getattr(self, field_category).keys():
                if field not in self.overview['field'].values:
                    new_row = {
                        'Description': field_category.replace('_fields', ''),
                        'field': field,
                        'value': getattr(self, field_category)[field]['value'],
                        'unit': getattr(self, field_category)[field]['unit']
                    }
                    self.overview = pd.concat([self.overview, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    self.overview.loc[self.overview['field'] == field, 'value'] = getattr(self, field_category)[field]['value']
                    self.overview.loc[self.overview['field'] == field, 'unit'] = getattr(self, field_category)[field]['unit']

        # sort overview by Description
        self.overview = self.overview.sort_values(by=['Description'])
        self.overview = self.overview.reset_index(drop=True)

    @type_check_decorator
    def print_budget(
        self,
        print_precision: int = 2
    ):
        """Print the budget for the species.

        Parameters
        ----------
        print_precision : int, optional
            The number of decimal places to print, by default 2
        """
        self.update_overview()

        width = 25
        print(f'{"Species".ljust(width)}: {self.species}')
        if not np.isnan(self.load_delta):
            print(f'{"Load change".ljust(width)}: {self.load_delta:.{print_precision}e} {self.unit_after_integration}')
        print(f'{"Source".ljust(width)}: {self.source:.{print_precision}e} {self.unit_after_integration}')
        print(f'{"Sink".ljust(width)}: {self.sink:.{print_precision}e} {self.unit_after_integration}')
        print(f'{"Balance".ljust(width)}: {self.balance:.{print_precision}e} {self.unit_after_integration}')
        if self.unexplained_error is not np.nan:
            print(f'{"Unexplained error (uxe)".ljust(width)}: {self.unexplained_error:.{print_precision}e} {self.unit_after_integration}')
            ratio_uxe_source = abs(self.unexplained_error) / self.source
            ratio_uxe_sink = abs(self.unexplained_error) / self.sink
            print(f'{"|uxe| / Source".ljust(width)}: {ratio_uxe_source:.{print_precision}e}')
            print(f'{"|uxe| / Sink".ljust(width)}: {ratio_uxe_sink:.{print_precision}e}')

        print(
            tabulate(self.overview, headers="keys", tablefmt="fancy_outline", floatfmt=".4e", showindex=False),
            "\n"
        )

class BlackCarbon(CalcBudget):
    def __init__(self):
        self.species = 'BC'

        self.unit_after_integration = 'kg'

        self.source = np.nan
        self.source_fields = {
            'emis_BC': {'value': np.nan, 'unit': 'kg m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_BC': {'value': np.nan, 'unit': 'kg m-2 s-1','new_unit': self.unit_after_integration},
            'dry_BC': {'value': np.nan, 'unit': 'kg m-2 s-1'},
        }
        self.unexplained_error = np.nan
        self.balance = np.nan
        self.load_delta = np.nan
        self.load_species = [
            'BC_A', 'BC_AC', 'BC_AI', 'BC_AX', 'BC_N', 'BC_NI',
            'BC_A_OCW', 'BC_AI_OCW', 'BC_AC_OCW', 'BC_N_OCW', 'BC_NI_OCW'
        ]
        self.overview = pd.DataFrame(columns=['Description', 'field', 'value', 'unit'])
        self.update_overview()

class Dust(CalcBudget):
    def __init__(self):
        self.species = 'DST'

        self.unit_after_integration = 'kg'

        self.source = np.nan
        self.source_fields = {
            'emis_DUST': {'value': np.nan, 'unit': 'kg m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_DUST': {'value': np.nan, 'unit': 'kg m-2 s-1'},
            'dry_DUST': {'value': np.nan, 'unit': 'kg m-2 s-1'},
        }
        self.unexplained_error = np.nan
        self.balance = np.nan
        self.load_delta = np.nan
        self.load_species = [
            'DST_A2', 'DST_A3',
            'DST_A2_OCW', 'DST_A3_OCW'
        ]
        self.overview = pd.DataFrame(columns=['Description', 'field', 'value', 'unit'])
        self.update_overview()

class OrganicMatter(CalcBudget):
    def __init__(self):
        self.species = 'OM'

        self.unit_after_integration = 'kg'

        self.source = np.nan
        self.source_fields = {
            'sour_OM': {'value': np.nan, 'unit': 'kg m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_OM': {'value': np.nan, 'unit': 'kg m-2 s-1'},
            'dry_OM': {'value': np.nan, 'unit': 'kg m-2 s-1'},
        }
        self.unexplained_error = np.nan
        self.balance = np.nan
        self.load_delta = np.nan
        self.load_species = [
            'OM_AC', 'OM_AI', 'OM_NI', 'SOA_A1', 'SOA_NA',
            'OM_AC_OCW', 'OM_AI_OCW', 'OM_NI_OCW', 'SOA_A1_OCW', 'SOA_NA_OCW'
        ]
        self.overview = pd.DataFrame(columns=['Description', 'field', 'value', 'unit'])
        self.update_overview()


class Salt(CalcBudget):
    def __init__(self):
        self.species = 'SS'

        self.unit_after_integration = 'kg'

        self.source = np.nan
        self.source_fields = {
            'emis_SALT': {'value': np.nan, 'unit': 'kg m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_SALT': {'value': np.nan, 'unit': 'kg m-2 s-1'},
            'dry_SALT': {'value': np.nan, 'unit': 'kg m-2 s-1'},
        }
        self.unexplained_error = np.nan
        self.balance = np.nan
        self.load_delta = np.nan
        self.load_species = [
            'SS_A1', 'SS_A2', 'SS_A3',
            'SS_A1_OCW', 'SS_A2_OCW', 'SS_A3_OCW'
        ]
        self.overview = pd.DataFrame(columns=['Description', 'field', 'value', 'unit'])
        self.update_overview()

class Sulfate(CalcBudget):
    def __init__(self):
        self.species = 'SULFATE'

        self.unit_after_integration = 'kg*S'
        self.source = np.nan
        self.source_fields = {
            'sour_SULFATE_S': {'value': np.nan, 'unit': 'kg*S m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_SULFATE_S': {'value': np.nan, 'unit': 'kg*S m-2 s-1'},
            'dry_SULFATE_S': {'value': np.nan, 'unit': 'kg*S m-2 s-1'},
        }
        self.unexplained_error = np.nan
        self.balance = np.nan
        self.load_delta = np.nan
        self.load_species = [
            'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_NA', 'SO4_PR',
            'SO4_A1_OCW', 'SO4_A2_OCW', 'SO4_AC_OCW', 'SO4_NA_OCW', 'SO4_PR_OCW'
        ]
        self.overview = pd.DataFrame(columns=['Description', 'field', 'value', 'unit'])
        self.update_overview()

        # sulfur mass fraction
        self.sulfur_mass_fractions = {
            'SO4_A1': 1/3.06,
            'SO4_A2': 1/3.59,
            'SO4_AC': 1/3.06,
            'SO4_NA': 1/3.06,
            'SO4_PR': 1/3.06,
        }


    @type_check_decorator
    def calc_load(
        self,
        ds:                     xr.Dataset,
    ):
        """Calculate the load of the species in the dataset.
        The load is calculated by integrating the species over the vertical dimension.
        Here we also convert the load to sulfur mass by multiplying with the sulfur mass fraction.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset containing the species to be integrated.

        Returns
        -------
        ds : xr.Dataset
            The input dataset with the load of the species added as a new variable, i.e. cb_SULFATE.
        """
        for var in self.load_species:
            # create integrater
            integrater = make_columnintegrater.create_columnintegrater(
                var=var,
                unit='kg kg-1',
                field=ds[var],
                hyai=ds['hyai'],
                hybi=ds['hybi'],
                p0=ds['P0'],
                ps=ds['PS']
            )
            # integrate and add load to dataset
            cb = integrater.integrate()
            # convert to sulfur mass
            sulfur_mass_fraction = self.sulfur_mass_fractions[var.replace('_OCW', '')]
            ds[f'cb_{var}'] = cb * sulfur_mass_fraction

        ds[f'cb_{self.species}'] = ds[[f'cb_{var}' for var in self.load_species]].to_array().sum(dim='variable')
        return ds

valid_species_dict = {
        'BC': BlackCarbon,
        'DST': Dust,
        'OM': OrganicMatter,
        'SS': Salt,
        'SULFATE': Sulfate,
    }

class create_budget_calculator:
    """Factory class to create budget calculators for different species."""

    @type_check_decorator
    @staticmethod
    def create(
        species: str
    ) -> Union[BlackCarbon, Dust, OrganicMatter, Salt, Sulfate]:
        """Create a budget calculator for a given species.

        Parameters
        ----------
        species : str
            Species to create a budget calculator for. Must be one of 'BC', 'DST', 'OM', 'SS', 'SULFATE'.

        Returns
        -------
        Union[BlackCarbon, Dust, OrganicMatter, Salt, Sulfate]
            Budget calculator for the given species.

        Raises
        ------
        ValueError
            If species is not one of 'BC', 'DST', 'OM', 'SS', 'SULFATE'.
        """

        species_upper = species.upper()
        if species_upper not in valid_species_dict:
            raise ValueError(f'Invalid species {species}. Must be one of {list(valid_species_dict.keys())}')
        return valid_species_dict[species_upper]()

def check_budget(
    history_file:           Union[str, Path],
    start_conditions:       Union[str, Path, None]  = None,
    end_conditions:         Union[str, Path, None]  = None,
    time_diff_in_seconds:   Union[int, None]        = None,
    species:                Union[str, list, None]  = None,
    plot_load_delta:        bool                    = False
):
    """Main function to calculate the budget for a given species or set of species.

    Parameters
    ----------
    history_file : str
        Path to history file.
    start_conditions : Union[str, Path, None], optional
        Path to restart file used to calculcate the load of the species at start of simulation conditions, by default None.
        If None, the load change will not be calculated.
    end_conditions : Union[str, Path, None], optional
        Path to restart file used to calculcate the load of the species at end of simulation conditions, by default None.
        If None, the load change will not be calculated.
    time_diff_in_seconds : Union[int, None], optional
        Time difference between start and end conditions in seconds, by default None.
        If None, it will atempt to calculate the time difference from the start and end conditions file paths
        assuming they are collected from <case-archive>/rest/<time_stamp>/<case_name>.cam.r.<time_stamp>.nc.
        If the time difference cannot be calculated a ValueError will be raised.
    species : Union[str, list, None], optional
        Species to calculate the budget for, by default None. If None, the budget will be calculated for all species.
        If a list is provided, the budget will be calculated for each species in the list.
    plot_load_delta : bool, optional
        Whether to plot the load changes for each species, by default False.
    """

    # ------------------
    # check arguments
    # ------------------
    # --- history_file and start/end_conditions
    # check that start_conditions is a string or None
    for condition, name in zip([history_file, start_conditions, end_conditions],
                               ['history_file', 'start_conditions', 'end_conditions']):
        if condition is not None:
            if not isinstance(condition, str):
                raise ValueError(f'{name} must be a string or None')
            # check that start_conditions is a valid path
            condition = Path(condition).resolve()
            if not condition.exists():
                raise ValueError(f'File {condition} does not exist')
            if condition.suffix != '.nc':
                raise ValueError(f'File {condition} is not a NetCDF file')
        else:
            print(f'No {name} provided, load delta will not be calculated')
    # find the time difference between the start and end conditions
    if start_conditions is not None and end_conditions is not None:
        produce_load_delta = True
        if time_diff_in_seconds is None:
            try:
                time_stamp_start = start_conditions.split('rest/')[1].split('/')[0]
                time_stamp_end = end_conditions.split('rest/')[1].split('/')[0]
                time_diff_in_seconds = calculate_seconds_diff(
                    time_stamp_start,
                    time_stamp_end,
                    date_format='%Y-%m-%d',
                    time_format='sssss',
                )
            except Exception as error:
                raise ValueError('Could not calculate time difference from start and end conditions file paths. Please provide a time_diff_in_seconds argument') from error
    else:
        produce_load_delta = False
        time_diff_in_seconds = None
    # --- species
    # assign default species if not provided
    if species is None:
        species = ['BC', 'DST', 'OM', 'SS', 'SULFATE']
    else:
        # check that species is a string or a list of strings
        if isinstance(species, str):
            species = [species]
        # check that all species are valid
        for spcs in species:
            if spcs not in valid_species_dict:
                raise ValueError(f'Invalid species {spcs}. Must be one of {list(valid_species_dict.keys())}')
    # --- plot_load_delta
    # check that plot_load_delta is a boolean
    if not isinstance(plot_load_delta, bool):
        raise ValueError('plot_load_delta must be a boolean')

    # ---------------
    # Handle paths
    # ---------------
    # make sure we are operating in the correct directory
    os.chdir(Path(__file__).resolve().parent)

    # ---------------
    # Main program
    # ---------------
    # Load history file
    hist_file = xr.open_dataset(Path(history_file).resolve())
    # Load start and end conditions files
    start_conditions_ds = None
    end_conditions_ds = None
    if produce_load_delta:
        start_conditions_path = Path(start_conditions).resolve()
        end_conditions_path = Path(end_conditions).resolve()
        if start_conditions_path == end_conditions_path:
            raise ValueError('Start and end conditions files are the same')
        start_conditions_ds = xr.open_dataset(start_conditions_path)
        end_conditions_ds = xr.open_dataset(end_conditions_path)

    # calculate the budget for each species
    for aerosolspecies in ['BC', 'DST', 'OM', 'SS', 'SULFATE']:
        calculator = create_budget_calculator.create(aerosolspecies)
        calculator.calc_budget(
            hist_file,
            time_diff_in_seconds=time_diff_in_seconds,
            start_conditions_ds=start_conditions_ds,
            end_conditions_ds=end_conditions_ds,
            plot_load_delta=plot_load_delta,
        )
        calculator.print_budget(print_precision=4)


if __name__ == '__main__':

    import argparse

    # -------------------------------
    # Parse command line arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'history_file',
        type=str,
        help='Absolute path to history file'
    )
    parser.add_argument(
        '--start_conditions',
        type=str,
        default=None,
        help='Absolute path to start conditions file to calculate load delta for. Default is None, which means no load delta is calculated.'
    )
    parser.add_argument(
        '--end_conditions',
        type=str,
        default=None,
        help='Absolute path to end conditions file to calculate load delta for. Default is None, which means no load delta is calculated.'
    )
    parser.add_argument(
        '--species',
        type=Union[str, list],
        default=['BC', 'DST', 'OM', 'SS', 'SULFATE'],
        help=f'Species to calculate budget for, default is all species. Must be one of {list(valid_species_dict.keys())}.'
    )
    parser.add_argument(
        '--plot_load_delta',
        type=bool,
        default=False,
        help='Whether to plot the load delta. Default is False.'
    )
    args = parser.parse_args()

    check_budget(
        history_file=args.history_file,
        start_conditions=args.start_conditions,
        end_conditions=args.end_conditions,
        species=args.species,
        plot_load_delta=args.plot_load_delta
    )
