def columnintegration(var, field, unit, hyai, hybi, p0, ps, ta, hyam, hybm, limit='', ptrop=None) :
#--------------------
    '''
    PARAMETERS
    ----------
	var : original name of the (unintegrated) species [e.g., ozone]
    field : 3d array (NOT a structure)
        hyai
        hybi
        p0
        ps
	ta
	hyam
	hybm
        limit [optional] : can be above100hPa | belowtropopause | abovetropopause
        ptrop [optional] : tropopause pressure field

    RETURNS
    -------
	columnintegrated field

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
    '''

    print('+++ column integration +++')

    Mw_dict = {
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
    unit_dict = {
        'mol mol-1': {'new_unit': 'kg m-2', 'needs_density': False},
        'kg kg-1': {'new_unit': 'kg m-2', 'needs_density': False},
        'kg-1': {'new_unit': 'm-2', 'needs_density': False},
        'm-3': {'new_unit': 'm-2', 'needs_density': True},
        'cm-3': {'new_unit': 'm-2', 'needs_density': True},
        'cm-2 cm-3': {'new_unit': 'cm-2 m-2', 'needs_density': True},
        'molecules cm-3 s-1': {'new_unit': 'kg m-2 s-1', 'needs_density': True},
        'm-1': {'new_unit': '', 'needs_density': True},
        'km-1': {'new_unit': '', 'needs_density': True},
        'm3 m-3': {'new_unit': 'm3 m-2', 'needs_density': True}
    }

    # Define standard values
    Rair  = 287.058
    Mwair =  28.97
    Nav   =   6.022e23
    Mw = Mw_dict[var] if var in Mw_dict.keys() else np.nan
    #
    nlev, nlat, nlon = field.shape
    #
    column = np.zeros( ( nlat,nlon ) )  # new  array
    #
    # check if ta is provided if needed
    if unit_dict[unit]['needs_density'] and ta == None:
        raise ValueError(f'Air temperature (ta) must provided when calculating with unit {unit}.')

#   standard : whole column
    for ilev in range(nlev) :

        pressure = hyam[ilev] * p0 + hybm[ilev] * ps[:,:] # mid-level pressure

        if unit_dict[unit]['needs_density']:
            density  = pressure / Rair / ta[ilev,:,:]         # density

#       airmass
        dm = abs( ( hyai[ilev+1] - hyai[ilev] ) * p0 + ( hybi[ilev+1] -hybi[ilev] )  * ps[:,:] ) / 9.81    # dm = dp / g

#       how much does level contributes
        w = np.zeros( ( nlat, nlon ) )
        if ( limit == '' ) : w[:,:] = 1.
        elif ( limit == 'above100hPa' ) :
            plim = 10000.
            w = np.zeros( ( nlat, nlon ) )
            for ilat in range(nlat) :
                for ilon in range(nlon) :
                   pa   = hyai[ilev]   * p0  + hybi[ilev  ] * ps[ilat,ilon] # pressure at ilev interface
                   pb   = hyai[ilev+1] * p0  + hybi[ilev+1] * ps[ilat,ilon] # pressure at ilev+1 interface
                   pmin = np.amin([pa,pb])                                  # minimum of boundary pressures
                   pmax = np.amax([pa,pb])                                  # maximum of boundary pressures
                   w[ilat,ilon] = np.amin([1.,np.amax([0., (pmax-plim) / (pmax-pmin) ])])
        elif ( limit == 'belowtropopause' \
            or limit == 'abovetropopause' ) :
            phigh = np.zeros( ( nlat, nlon ) )
            plow  = np.zeros( ( nlat, nlon ) )
            if (   limit == 'belowtropopause' ) : phigh[:,:] = 120000.    ; plow[:,:] = ptrop[:,:]
            elif ( limit == 'abovetropopause' ) : phigh[:,:] = ptrop[:,:] ; plow[:,:] = 0.
            else :
                print('No limit recognized : ', limit)
                sys.quit()

            for ilat in range(nlat) :
                for ilon in range(nlon) :
#                  level boundaries
                   pa   = hyai[ilev]   * p0  + hybi[ilev  ] * ps[ilat,ilon] # pressure at ilev interface
                   pb   = hyai[ilev+1] * p0  + hybi[ilev+1] * ps[ilat,ilon] # pressure at ilev+1 interface
#                  external limits
                   w[ilat,ilon] = getlenintervaloverlap.getlenintervaloverlap(x=[pa,pb],y=[plow[ilat,ilon],phigh[ilat,ilon]]) / abs(pb-pa)

        else : sys.exit()

#       combination
        dm = dm * w

        if   ( unit=='mol mol-1'          ) : column = column + dm * field[ilev,:,:] * Mw / Mwair                         ; newunit = 'kg m-2'
        elif ( unit=='kg kg-1'            ) : column = column + dm * field[ilev,:,:]                                      ; newunit = 'kg m-2'
        elif ( unit=='kg-1'               ) : column = column + dm * field[ilev,:,:]                                      ; newunit = 'm-2'
        elif ( unit=='m-3'                ) : column = column + dm * field[ilev,:,:] / density                            ; newunit = 'm-2'
        elif ( unit=='cm-3'               ) : column = column + dm * field[ilev,:,:] / density    *1.e6                   ; newunit = 'm-2'
        elif ( unit=='cm-2 cm-3'          ) : column = column + dm * field[ilev,:,:] / density    *1.e6                   ; newunit = 'cm-2 m-2'
        elif ( unit=='molecules cm-3 s-1' ) : column = column + dm * field[ilev,:,:] / density    *1.e6 * Mw *1.e-3 / Nav ; newunit = 'kg m-2 s-1'
        elif ( unit=='m-1'                ) : column = column + dm * field[ilev,:,:] / density                            ; newunit = ''
        elif ( unit=='km-1'               ) : column = column + dm * field[ilev,:,:] / density * 1.e-3                    ; newunit = ''
        elif ( unit=='m3 m-3'             ) : column = column + dm * field[ilev,:,:] / density                            ; newunit = 'm3 m-2'
        else :
            print('columnintegration : Unit not recognized : '+unit)
            sys.exit()

    print('--- column integration ---')

    return column, newunit
#