#
# By Dirk Olivie up until 2025-05
# By Johannes Fjeldså from 2025-05

import numpy as np
import xarray as xr
from getlenintervaloverlap import getlenintervaloverlap

class columnintegration:

    def __init__(
        self,
        var:    str,
        unit:   str,
        field:  xr.DataArray,
        hyai:   xr.DataArray,
        hybi:   xr.DataArray,
        p0:     xr.DataArray,
        ps:     xr.DataArray,
        ta:     xr.DataArray=None,
        hyam:   xr.DataArray=None,
        hybm:   xr.DataArray=None,
        limit:  str='',
        ptrop:  xr.DataArray=None,
    ):
        """Initializes the column integration class.
        This class is used to perform column integration of atmospheric fields.

        Parameters
        ----------
        var : str
            The variable name of the field to be integrated (e.g., 'ozone').
        unit : str
            The unit of the field to be integrated (e.g., 'kg kg-1').
        field : xr.DataArray
            The 3D array of the field to be integrated.
        hyai : xr.DataArray
            _description_
        hybi : xr.DataArray
            _description_
            _description_
        p0 : xr.DataArray
            _description_
        ps : xr.DataArray
            _description_
        ta : xr.DataArray, optional
            _description_, by default None
        hyam : xr.DataArray
            _description_, by default None
        hybm : xr.DataArray
            _description_, by default None
        limit : str, optional
            _description_, by default ''
        ptrop : xr.DataArray, optional
            _description_, by default None

        Returns
        -------
        columnintegration of the specified field

        Raises
        ------
        ValueError
            If the limit is not recognized or if ptrop is not provided when using 'belowtropopause' or 'abovetropopause' limit.

        MODIFICATIONS
        -------------
        (2021-04-16) possibility for concentrations (m-3 or cm-3)
            needs temperature field to calculate density
        (2022-01-10) added extra argument function call : var
            needed to distinguish Molar weights
        (2022-02-28) extra optional argument : limit
            limit : above100hPa
            (2022-07-05) added possibility to go from extinction -> [AOD]
            (2022-07-11) added possibility to integrate aerosol volume [m m-3]
            (2022-11-17) added possibility to integrate aerosol surface density [cm2 cm-3]
            (2024-06-13) added possibility to integrate
                tropcolumn
                stratcolumn
        (2025-02-05) refactored the code to use classes and subclasses. Made it xarray compatible. - Johannes Fjeldså
        """
        # Define molecular weight database
        self.Mw_dict = {
            'ozone': 47.9982,
            'ozone_extclim': 47.9982,
            'O3': 47.9982,
            'OH': 17.0068,
            'HO2': 33.0062,
            'LNO_PROD': 14.0067,
            'r_jchbr3': 252.7304,      # CHBR3
            'r_CHBR3_OH': 252.7304,    # CHBR3
            'r_DMS_OHa': 62.1324,      # DMS
            'r_DMS_OH': 62.1324,       # DMS
            'r_DMS_NO3': 62.1324,      # DMS
            'r_jch4_a': 16.0406,       # CH4
            'r_jch4_b': 16.0406,       # CH4
            'r_CH4_OH': 16.0406,       # CH4
            'r_CL_CH4': 16.0406,       # CH4
            'r_F_CH4': 16.0406,        # CH4
            'r_O1D_CH4a': 16.0406,     # CH4
            'r_O1D_CH4b': 16.0406,     # CH4
            'r_O1D_CH4c': 16.0406,     # CH4
        }
        implemented_limits = ['','above100hPa','belowtropopause','abovetropopause']
        # Initialize the attributes of the class with the provided parameters
        self.var = var
        self.unit = unit
        self.field = field
        self.hyai = hyai
        self.hybi = hybi
        self.p0 = p0
        self.ps = ps
        self.ta = ta
        self.hyam = hyam
        self.hybm = hybm
        self.limit = limit
        self.ptrop = ptrop
        # check that limit is valid
        if limit not in implemented_limits:
            raise ValueError(f"Limit '{limit}' is not recognized. Valid options are: {', '.join(implemented_limits)}.")
        if limit == 'belowtropopause' or limit == 'abovetropopause':
            if ptrop is None:
                raise ValueError("ptrop must be provided when using 'belowtropopause' or 'abovetropopause' limit.")
        # Initialize constants
        self.Rair = 287.058
        self.Mwair = 28.97
        self.Nav = 6.022e23
        self.Mw = self.Mw_dict.get(var, np.nan)  # Get molecular weight from dictionary or set to NaN
        self.nlev, self.nlat, self.nlon = field.shape  # Get the shape of the field
        # Initialize the column array
        self.column = np.zeros((self.nlat, self.nlon)) # Initialize the column array

    def describe(self):
        return f'Integration for {self.var} in {self.unit} units.'

    def integrate(self):
        #   standard : whole column
        for ilev in range(self.nlev) :

            # estimate level airmass dm = dp / g
            dm = (abs(
                ( self.hyai[ilev+1] - self.hyai[ilev] ) * self.p0 +
                ( self.hybi[ilev+1] - self.hybi[ilev] ) * self.ps[:,:]
            ) / 9.81
            )

            # how much does level contributes
            w = np.zeros( ( self.nlat, self.nlon ) )
            if ( self.limit == '' ):
                w[:,:] = 1.
            elif ( self.limit == 'above100hPa' ) :
                plim = 10000.
                w = np.zeros( ( self.nlat, self.nlon ) )
                for ilat in range(self.nlat) :
                    for ilon in range(self.nlon) :
                        pa   = self.hyai[ilev] * self.p0  + self.hybi[ilev  ] * self.ps[ilat,ilon] # pressure at ilev interface
                        pb   = self.hyai[ilev+1] * self.p0  + self.hybi[ilev+1] * self.ps[ilat,ilon] # pressure at ilev+1 interface
                        pmin = np.amin([pa,pb]) # minimum of boundary pressures
                        pmax = np.amax([pa,pb]) # maximum of boundary pressures
                        w[ilat,ilon] = np.amin([1.,np.amax([0., (pmax-plim) / (pmax-pmin) ])])
            elif ( self.limit == 'belowtropopause' or self.limit == 'abovetropopause' ) :
                phigh = np.zeros( ( self.nlat, self.nlon ) )
                plow  = np.zeros( ( self.nlat, self.nlon ) )
                if (   self.limit == 'belowtropopause' ) :
                    phigh[:,:] = 120000.
                    plow[:,:] = self.ptrop[:,:]
                elif ( self.limit == 'abovetropopause' ) :
                    phigh[:,:] = self.ptrop[:,:]
                    plow[:,:] = 0.


                for ilat in range(self.nlat) :
                    for ilon in range(self.nlon) :
                        # level boundaries
                        pa   = self.hyai[ilev]   * self.p0  + self.hybi[ilev  ] * self.ps[ilat,ilon] # pressure at ilev interface
                        pb   = self.hyai[ilev+1] * self.p0  + self.hybi[ilev+1] * self.ps[ilat,ilon] # pressure at ilev+1 interface
                        # external limits
                        w[ilat,ilon] = getlenintervaloverlap.getlenintervaloverlap(x=[pa,pb],y=[plow[ilat,ilon],phigh[ilat,ilon]]) / abs(pb-pa)

            # update the airmass with the weight
            dm = dm * w

            # update the column with the level contribution
            self.add_level_contribution(dm, ilev)

        return self.column

    def add_level_contribution(self):
        raise NotImplementedError("The add_level_contribution method is implemented in subclasses only.")

class mol_per_mol(columnintegration):
    """Subclass for 'mol mol-1' unit."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = 'kg m-2'
        # check that Mw is not NaN
        if np.isnan(self.Mw):
            raise ValueError(f"Molecular weight for {self.var} is not defined in Mw_dict which is required for {self.unit}.")

    def describe(self):
        return f'{super().describe()} Requires molecular weight for {self.var} described in Mw_dict.'

    def add_level_contribution(self, dm, ilev):
        self.column += (
            dm * self.field[ilev,:,:] * self.Mw / self.Mwair
        )


class kg_per_kg(columnintegration):
    """Subclass for 'kg kg-1' unit."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = 'kg m-2'

    def describe(self):
        return super().describe()

    def add_level_contribution(self, dm, ilev):
        self.column += (
            dm * self.field[ilev,:,:]
        )

class per_kg(columnintegration):
    """Subclass for 'kg-1' unit."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = 'm-2'

    def describe(self):
        return super().describe()

    def add_level_contribution(self, dm, ilev):
        self.column += (
            dm * self.field[ilev,:,:]
        )

class per_m3(columnintegration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = 'm-2'
        # check that ta, hyam and hybm are provided
        if self.ta or self.hyam or self.hybm is None:
            raise ValueError(f"Air temperature (ta), hyam and hybm must be provided when calculating with unit {self.unit}.")

    def describe(self):
        return f'{super().describe()} Requires air temperature (ta), hyam and hybm to be a 3D xr.DataArray.'

    def add_level_contribution(self, dm, ilev):
        pressure = self.hyam[ilev] * self.p0 + self.hybm[ilev] * self.ps[:,:]  # mid-level pressure
        density = pressure / self.Rair / self.ta[ilev,:,:]  # density
        self.column += (
            dm * self.field[ilev,:,:] / density
        )

class per_cm3(columnintegration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = 'm-2'
        # check that ta, hyam and hybm are provided
        if self.ta or self.hyam or self.hybm is None:
            raise ValueError(f"Air temperature (ta), hyam and hybm must be provided when calculating with unit {self.unit}.")

    def describe(self):
        return f'{super().describe()} Requires air temperature (ta), hyam and hybm to be a 3D xr.DataArray.'

    def add_level_contribution(self, dm, ilev):
        pressure = self.hyam[ilev] * self.p0 + self.hybm[ilev] * self.ps[:,:]  # mid-level pressure
        density = pressure / self.Rair / self.ta[ilev,:,:]  # density
        self.column += (
            dm * self.field[ilev,:,:] / density * 1e6
        )

class per_cm2_cm3(columnintegration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = 'cm-2 m-2'
        # check that ta, hyam and hybm are provided
        if self.ta or self.hyam or self.hybm is None:
            raise ValueError(f"Air temperature (ta), hyam and hybm must be provided when calculating with unit {self.unit}.")

    def describe(self):
        return f'{super().describe()} Requires air temperature (ta), hyam and hybm to be a 3D xr.DataArray.'

    def add_level_contribution(self, dm, ilev):
        pressure = self.hyam[ilev] * self.p0 + self.hybm[ilev] * self.ps[:,:]
        density = pressure / self.Rair / self.ta[ilev,:,:]  # density
        self.column += (
            dm * self.field[ilev,:,:] / density * 1e6
        )

class mulecules_per_cm3_s(columnintegration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = 'kg m-2 s-1'
        # check that ta, hyam and hybm are provided
        if self.ta or self.hyam or self.hybm is None:
            raise ValueError(f"Air temperature (ta), hyam and hybm must be provided when calculating with unit {self.unit}.")
        # check that Mw are not NaN
        if np.isnan(self.Mw):
            raise ValueError(f"Molecular weight for {self.var} is not defined in Mw_dict which is required for {self.unit}.")

    def describe(self):
        return f'{super().describe()} Requires air temperature (ta), hyam and hybm to be a 3D xr.DataArray and molecular weight for {self.var} defined in Mw_dict.'

    def add_level_contribution(self, dm, ilev):
        pressure = self.hyam[ilev] * self.p0 + self.hybm[ilev] * self.ps[:,:]
        density = pressure / self.Rair / self.ta[ilev,:,:]  # density
        self.column += (
            dm * self.field[ilev,:,:] / density * 1e6 * self.Mw * 1e-3 / self.Nav
        )

class per_m(columnintegration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = ''
        # check that ta, hyam and hybm are provided
        if self.ta or self.hyam or self.hybm is None:
            raise ValueError(f"Air temperature (ta), hyam and hybm must be provided when calculating with unit {self.unit}.")

    def describe(self):
        return f'{super().describe()} Requires air temperature (ta), hyam and hybm to be a 3D xr.DataArray.'

    def add_level_contribution(self, dm, ilev):
        pressure = self.hyam[ilev] * self.p0 + self.hybm[ilev] * self.ps[:,:]
        density = pressure / self.Rair / self.ta[ilev,:,:]
        self.column += (
            dm * self.field[ilev,:,:] / density
        )

class per_km(columnintegration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = ''
        # check that ta, hyam and hybm are provided
        if self.ta or self.hyam or self.hybm is None:
            raise ValueError(f"Air temperature (ta), hyam and hybm must be provided when calculating with unit {self.unit}.")

    def describe(self):
        return f'{super().describe()} Requires air temperature (ta), hyam and hybm to be a 3D xr.DataArray.'

    def add_level_contribution(self, dm, ilev):
        pressure = self.hyam[ilev] * self.p0 + self.hybm[ilev] * self.ps[:,:]
        density = pressure / self.Rair / self.ta[ilev,:,:]
        self.column += (
            dm * self.field[ilev,:,:] / density * 1e-3
        )

class m3_per_m3(columnintegration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization for this subclass
        # new unit
        self.new_unit = 'm3 m-2'
        # check that ta, hyam and hybm are provided
        if self.ta or self.hyam or self.hybm is None:
            raise ValueError(f"Air temperature (ta), hyam and hybm must be provided when calculating with unit {self.unit}.")

    def describe(self):
        return f'{super().describe()} Requires air temperature (ta), hyam and hybm to be a 3D xr.DataArray.'

    def add_level_contribution(self, dm, ilev):
        pressure = self.hyam[ilev] * self.p0 + self.hybm[ilev] * self.ps[:,:]
        density = pressure / self.Rair / self.ta[ilev,:,:]
        self.column += (
            dm * self.field[ilev,:,:] / density
        )

class make_columnintegrater:
    """Factory function to create a column integrator based on the variable name.
    """
    @staticmethod
    def create_columnintegrater(
        var:    str,
        unit:   str,
        field:  xr.DataArray,
        hyai:   xr.DataArray,
        hybi:   xr.DataArray,
        p0:     xr.DataArray,
        ps:     xr.DataArray,
        ta:     xr.DataArray=None,
        hyam:   xr.DataArray=None,
        hybm:   xr.DataArray=None,
        limit:  str='',
        ptrop:  xr.DataArray=None,
    ):
        """Creates a column integrator based on the unit"""
        arguments = {
            'var': var,
            'unit': unit,
            'field': field,
            'hyai': hyai,
            'hybi': hybi,
            'p0': p0,
            'ps': ps,
            'ta': ta,
            'hyam': hyam,
            'hybm': hybm,
            'limit': limit,
            'ptrop': ptrop,
        }

        if unit == 'mol mol-1':
            return mol_per_mol(**arguments)
        elif unit == 'kg kg-1':
            return kg_per_kg(**arguments)
        elif unit == 'kg-1':
            return per_kg(**arguments)
        elif unit == 'm-3':
            return per_m3(**arguments)
        elif unit == 'cm-3':
            return per_cm3(**arguments)
        elif unit == 'cm-2 cm-3':
            return per_cm2_cm3(**arguments)
        elif unit == 'molecules cm-3 s-1':
            return mulecules_per_cm3_s(**arguments)
        elif unit == 'm-1':
            return per_m(**arguments)
        elif unit == 'km-1':
            return per_km(**arguments)
        elif unit == 'm3 m-3':
            return m3_per_m3(**arguments)
        else:
            raise ValueError(f"Unit {unit} is not supported. Supported units are: \n",
                             f"mol mol-1, kg kg-1, kg-1, m-3, cm-3, cm-2 cm-3, molecules cm-3 s-1, m-1, km-1, m3 m-3.")
