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
from type_check_decorator import type_check_decorator
from descriptive_stat_utils import calculate_sum
from plot import plot_datasets_side_by_side
from columnintegration import make_columnintegrater

class CalcBudget:

    @type_check_decorator
    def calc_load(
        self,
        ds:                     xr.Dataset,
    ):
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
    def find_load_delta(
        self,
        start_conditions_ds:    xr.Dataset,
        end_conditions_ds:      xr.Dataset,
        verbose:                bool = False,
        plot:                   bool = False
    ):

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

        # check that the datasets have the same dimensions
        if end_conditions_ds.sizes != start_conditions_ds.sizes:
            raise ValueError("Datasets have different dimensions \n end_conditions_ds.sizes: {end_conditions_ds.sizes} \n start_conditions_ds.sizes: {start_conditions_ds.sizes}")

        # check that the datasets have the same coordinates
        for coord in [coord for coord in start_conditions_ds.coords if coord != 'time']:
            if coord not in end_conditions_ds.coords:
                raise ValueError(f"Coordinate {coord} not found in end_conditions_ds")
            else:
                # check that the coordinate is the same
                if not (start_conditions_ds[coord] == end_conditions_ds[coord]).all():
                    raise ValueError(f"Coordinate {coord} is not the same in both datasets")

        # integrate the variables over the vertical dimension to find the load
        # we store the loads as a new variable in the dataset
        for name, ds in datasets.items():
            ds = self.calc_load(ds)

        # calculate the delta
        glob_load_start = calculate_sum(start_conditions_ds[f'cb_{self.species}'], ['lat', 'lon']).values
        glob_load_end = calculate_sum(end_conditions_ds[f'cb_{self.species}'], ['lat', 'lon']).values
        load_delta = glob_load_end - glob_load_start

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
        start_conditions_ds:    Union[xr.Dataset, None] = None,
        end_conditions_ds:      Union[xr.Dataset, None] = None,
        time_diff_in_seconds:   Union[int, None] = None,
        verbose:                bool = False,
        plot_load_delta:        bool = False
    ):


        # check that all required fields are in ds
        for field in list(self.source_fields.keys()) + list(self.sink_fields.keys()):
            if field not in ds:
                raise ValueError(f'Source field {field} not in dataset')
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
            self.load_delta = self.find_load_delta(
                start_conditions_ds,
                end_conditions_ds,
                verbose=verbose,
                plot=plot_load_delta
            )
        # set time_diff_in_seconds to default if not provided
        time_diff_in_seconds = time_diff_in_seconds if time_diff_in_seconds is not None else 60*60*24*30

        # update the source_fields/sink_fields dictionary with the global average values from the dataset
        # then calculate the source and sink
        self.source = 0.0
        for field in self.source_fields.keys():
            self.source_fields[field]['value'] = (
                calculate_sum(ds[field], ['lat', 'lon']).values[0] *
                time_diff_in_seconds
            )
            self.source_fields[field]['unit'] = 'kg'
            self.source += self.source_fields[field]['value']
        self.sink = 0.0
        for field in self.sink_fields.keys():
            self.sink_fields[field]['value'] = (
                calculate_sum(ds[field], ['lat', 'lon']).values[0] *
                time_diff_in_seconds
            )
            if 'wet' in field:
                self.sink_fields[field]['value'] *= -1.0
            self.sink_fields[field]['unit'] = 'kg'
            self.sink += self.sink_fields[field]['value']

        # calculate the balance
        self.balance = self.source - self.sink

        if self.load_delta is not np.nan:
            self.adjust_balance = self.source -  self.sink - self.load_delta

        self.update_overview()

    @type_check_decorator
    def update_overview(self):

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

        # sort overview by Description
        self.overview = self.overview.sort_values(by=['Description'])
        self.overview = self.overview.reset_index(drop=True)

    @type_check_decorator
    def print_budget(
        self,
        print_precision: int = 2
    ):

        self.update_overview()

        width = 20
        print(f'{"Species".ljust(width)}: {self.species}')
        if not np.isnan(self.load_delta):
            print(f'{"Load change".ljust(width)}: {self.load_delta:.{print_precision}e} kg')
        print(f'{"Source".ljust(width)}: {self.source:.{print_precision}e} kg')
        print(f'{"Sink".ljust(width)}: {self.sink:.{print_precision}e} kg')
        print(f'{"Balance".ljust(width)}: {self.balance:.{print_precision}e} kg')
        if self.adjust_balance is not np.nan:
            print(f'{"Adjusted balance".ljust(width)}: {self.adjust_balance:.{print_precision}e} kg')

        print(
            tabulate(self.overview, headers="keys", tablefmt="fancy_outline", floatfmt=".4e", showindex=False),
            "\n"
        )

class BlackCarbon(CalcBudget):
    def __init__(self):
        self.species = 'BC'

        self.source = np.nan
        self.source_fields = {
            'emis_BC': {'value': np.nan, 'unit': 'kg m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_BC': {'value': np.nan, 'unit': 'kg m-2 s-1'},
            'dry_BC': {'value': np.nan, 'unit': 'kg m-2 s-1'},
        }
        self.adjust_balance = np.nan
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

        self.source = np.nan
        self.source_fields = {
            'emis_DUST': {'value': np.nan, 'unit': 'kg m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_DUST': {'value': np.nan, 'unit': 'kg m-2 s-1'},
            'dry_DUST': {'value': np.nan, 'unit': 'kg m-2 s-1'},
        }
        self.adjust_balance = np.nan
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

        self.source = np.nan
        self.source_fields = {
            'emis_OM': {'value': np.nan, 'unit': 'kg m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_OM': {'value': np.nan, 'unit': 'kg m-2 s-1'},
            'dry_OM': {'value': np.nan, 'unit': 'kg m-2 s-1'},
        }
        self.adjust_balance = np.nan
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

        self.source = np.nan
        self.source_fields = {
            'emis_SALT': {'value': np.nan, 'unit': 'kg m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_SALT': {'value': np.nan, 'unit': 'kg m-2 s-1'},
            'dry_SALT': {'value': np.nan, 'unit': 'kg m-2 s-1'},
        }
        self.adjust_balance = np.nan
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

        self.source = np.nan
        self.source_fields = {
            'sour_SULFATE_S': {'value': np.nan, 'unit': 'kg*S m-2 s-1'}
        }
        self.sink = np.nan
        self.sink_fields = {
            'wet_SULFATE_S': {'value': np.nan, 'unit': 'kg*S m-2 s-1'},
            'dry_SULFATE_S': {'value': np.nan, 'unit': 'kg*S m-2 s-1'},
        }
        self.adjust_balance = np.nan
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

    @staticmethod

    def create(species):

        species_upper = species.upper()
        if species_upper not in valid_species_dict:
            raise ValueError(f'Invalid species {species}. Must be one of {list(valid_species_dict.keys())}')
        return valid_species_dict[species_upper]()

def __main__():

    # -------------------------------
    # Import the necessary modules
    # -------------------------------
    import argparse
    from time_utils import calculate_seconds_diff

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

    # -------------------------------
    # check command line arguments
    # -------------------------------
    # Make sure that all paths are valid are NetCDF files
    for file_path in [args.history_file, args.start_conditions, args.end_conditions]:
        if file_path is not None:
            file_path = Path(file_path).resolve()
            if not file_path.exists():
                raise ValueError(f'File {file_path} does not exist')
            if file_path.suffix != '.nc':
                raise ValueError(f'File {file_path} is not a NetCDF file')
    if args.start_conditions is not None and args.end_conditions is not None:
        time_stamp_start = args.start_conditions.split('rest/')[1].split('/')[0]
        time_stamp_end = args.end_conditions.split('rest/')[1].split('/')[0]
        time_diff_in_seconds = calculate_seconds_diff(
            time_stamp_start,
            time_stamp_end,
            date_format='%Y-%m-%d',
            time_format='sssss',
        )
        produce_load_delta = True
    else:
        produce_load_delta = False
        time_diff_in_seconds = None
    # check that species is a list
    if isinstance(args.species, str):
        args.species = [args.species]

    # ---------------
    # Handle paths
    # ---------------
    # make sure we are operating in the correct directory
    os.chdir(Path(__file__).resolve().parent)

    # ---------------
    # Main program
    # ---------------
    # Load history file
    hist_file = xr.open_dataset(Path(args.history_file).resolve())
    # Load start and end conditions files
    if produce_load_delta:
        start_conditions_path = Path(args.start_conditions).resolve()
        end_conditions_path = Path(args.end_conditions).resolve()
        if start_conditions_path == end_conditions_path:
            raise ValueError('Start and end conditions files are the same')
        start_conditions_ds = xr.open_dataset(start_conditions_path)
        end_conditions_ds = xr.open_dataset(end_conditions_path)
    else:
        start_conditions_ds = None
        end_conditions_ds = None

    # calculate the budget for each species
    for aerosolspecies in ['BC', 'DST', 'OM', 'SS', 'SULFATE']:
        calculator = create_budget_calculator.create(aerosolspecies)
        calculator.calc_budget(
            hist_file,
            start_conditions_ds,
            end_conditions_ds,
            time_diff_in_seconds=time_diff_in_seconds,
            plot_load_delta=args.plot_load_delta,
        )
        calculator.print_budget(print_precision=4)

if __name__ == '__main__':
    __main__()
