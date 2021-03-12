#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWIFT automated synoptic charts

This module was developed by CEMAC as part of the GCRF African Swift
Project. This script allows automated plotting of synoptic features
across African domains.

.. module:: chart
   :synopsis: SWIFT automated synoptic charts

.. moduleauthor:: Tamora D. James <t.d.james1@leeds.ac.uk>, CEMAC (UoL)

:copyright: © 2020 University of Leeds.
:license: GPL 3.0 (see LICENSE)

Example::
        ./chart.py <domain> <forecast_timestamp> <forecast_hour> [<chart_type>]

        <domain> Domain specified as standardised domain name (WA or
                 EA)

        <timestamp> Timestamp for chart data in the format YYYYmmddHH

        <forecast_hour> Forecast hour as non-negative integer multiple
                 of 3 (max 72)

        <chart_type> Chart type (low, jets, conv or synth)

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)

import os
import sys
import datetime as dt
import argparse
import math
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as mcm
import matplotlib.colors as mc
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.offsetbox import AnchoredText
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import iris
import iris.analysis.calculus
import numpy as np
import skimage.measure
import shapely.geometry as sgeom

#from * import gfs_utils  # works for tests but not when calling module code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import gfs_utils  # works when calling module code

class SynopticChart:
    """
    Synoptic chart base class

    This class provides functionality for setting forecast domain,
    date and time, building charts, and plotting.

    """

    # Coords for standard domains from GFS_plotting/controls/domains
    # (could read in programatically from similar file)
    # Values are (min_lat, min_lon, max_lat, max_lon)
    # (could set up with value as {lat_min: -16.0, etc})

    # Domain specification is a tuple specifying (lat_min, lon_min,
    # lat_max, lon_max).
    DOMAINS = {
        'WA': (-2.5, -25.0, 36.0, 35.0),
        'EA': (-16.0, 20.0, 12.0, 50.0)
    }

    def __init__(self, domain, fct_timestamp, fct_hour, data_dir=None):
        self.set_domain(domain)
        self.set_timestamp(fct_timestamp)
        self.fct_hour = fct_hour
        self.set_data_dir(data_dir)
        self.components = []

        # Options for plotting backend
        self.mpl_options = {
            'figsize': None,
            'subplot_kw': { 'projection': ccrs.PlateCarree() },
        }

    def __str__(self):
        if self.chart_type is None:
            self.chart_type = "Synoptic"
        if self.domain_name is None:
            self.domain_name = "Custom"
        return (
            f'{self.chart_type.capitalize()} chart for {self.get_date_str()} '
            f'+ {self.fct_hour} hours ({self.domain_name} domain)'
        )

    def get_date_str(self):
        return dt.datetime.strptime(self.fct_timestamp, '%Y%m%d%H').strftime("%Y%m%d %H:%M")

    def set_domain(self, arg):
        arg_type = type(arg)
        if arg_type is str:
            if arg == 'WA' or arg == 'WestAfrica':
                self.domain = SynopticChart.DOMAINS['WA']
            elif arg == 'EA' or arg == 'EastAfrica':
                self.domain = SynopticChart.DOMAINS['EA']
            else:
                raise ValueError("Unrecognised domain name")
            self.domain_name = arg
        elif arg_type is dict:
            keys = ['lat_min', 'lon_min', 'lat_max', 'lon_max']
            if all([key in arg.keys() for key in keys]):
                self.domain = tuple([arg.get(key) for key in keys])
            else:
                raise ValueError("Domain does not contain required keys")
        elif arg_type is list or arg_type is tuple:
            if len(arg) == 4:
                self.domain = tuple(arg)
            else:
                raise ValueError("Domain does not contain the right number of values")
        else:
            raise ValueError("Unrecognised domain type")

    def set_timestamp(self, arg):
        try:
            if arg != dt.datetime.strptime(arg, '%Y%m%d%H').strftime('%Y%m%d%H'):
                raise ValueError('Unexpected timestamp format')
        except ValueError as e:
            raise(e)
        self.fct_timestamp = arg

    def set_data_dir(self, data_dir):
        if data_dir is None:
            # Use $SWIFT_GFS environment variable
            data_dir = os.getenv("SWIFT_GFS")
            if data_dir is None:
                raise RuntimeError('''Data directory not specified and
                $SWIFT_GFS environment variable is not set''')
        if not os.path.isdir(data_dir):
            raise ValueError("{:s} is not a directory".format(data_dir))
        self.data_dir = data_dir

    def date(self):
        return dt.datetime.strptime(self.fct_timestamp, '%Y%m%d%H')

    def get_file_path(self, timestamp):
        ts = dt.datetime.strptime(timestamp, '%Y%m%d%H')
        if self.fct_hour > 0:
            file_path = os.path.join(self.data_dir, "GFS_NWP", timestamp,
                                     'GFS_forecast_{:%Y%m%d_%H}.nc'.format(ts))
        else:
            file_path = os.path.join(self.data_dir, "GFS_NWP", timestamp,
                                     'analysis_gfs_4_{:%Y%m%d_%H}00_000.nc'.format(ts))
        if not os.path.isfile(file_path):
            raise ValueError("{:s} is not a file", file_path)
        return file_path

    def get_data(self, gfs_vars, units=None, delta=None):

        if delta is None:
            timestamp = self.fct_timestamp
        else:
            if not isinstance(delta, dt.timedelta):
                raise ValueError("Expecting datetime.timedelta object")

            ts = dt.datetime.strptime(self.fct_timestamp, "%Y%m%d%H") + delta
            timestamp = ts.strftime("%Y%m%d%H")

        # Load data
        with warnings.catch_warnings():
            # Suppress warnings
            warnings.simplefilter("ignore")
            cubes = iris.load_cubes(self.get_file_path(timestamp), gfs_vars)

        rtn = iris.cube.CubeList()
        for c in cubes:
            # Rewrap longitude
            c = c.intersection(longitude = (-180, 180))

            if units is not None:
                # Convert units
                c.convert_units(units)

            # Constrain to specified forecast hour for this chart
            fct_date = dt.datetime.strptime(c.attributes['initial_time'], '%m/%d/%Y (%H:%M)')
            time_constraint = gfs_utils.get_time_constraint(fct_date, self.fct_hour)
            c = c.extract(time_constraint)

            # Constrain to specified domain
            domain_constraint = gfs_utils.get_domain_constraint(self.domain)
            c = c.extract(domain_constraint)

            rtn.append(c)

        return rtn if len(rtn) > 1 else rtn[0]

    def add_component(self, component):
        self.components.append(component)

    def build(self, dir_path = None, scale_factor = 10):
        """Build chart from components"""
        if self.components:
            sample_data = self.components[0].data
            if isinstance(sample_data, iris.cube.CubeList):
                sample_data = sample_data[0]
            # Set sensible size in inches based on coordinate array extent
            coords = ('longitude', 'latitude')
            shape = np.array([sample_data.coord(x).points.size for x in coords])
            self.mpl_options['figsize'] = shape/scale_factor

        # set up chart - this might be configurable to use different
        # plotting back ends
        fig, ax = self.setup_plot()

        for c in self.components:
            # may need some concept of "order"?
            # add each component to plot
            c.plot(ax)

        if dir_path is not None:
            file_name = '{:%Y%m%d%H%M}_{:03d}.png'.format(self.date(), self.fct_hour)
            file_path = os.path.join(dir_path, file_name)
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()

    def setup_plot(self):

        # Set up figure
        fig = plt.figure(figsize=self.mpl_options['figsize'])

        # Set up axis
        ax = fig.add_subplot(**self.mpl_options['subplot_kw'])
        ax.set_position([0., 0., 1., 1.])  # Fill frame
        ax.set_axis_off()

        # Add gridlines
        grid_col = 'lightsteelblue'
        gl = ax.gridlines(draw_labels=True, x_inline=True, y_inline=True, zorder=1.5, color=grid_col)
        gl.xlocator = mticker.FixedLocator(np.arange(-30, 40, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 40, 10))
        gl.xlabel_style = { 'color': grid_col}
        gl.ylabel_style = { 'color': grid_col}

        # Add coastlines and borders
        map_alpha = 0.9
        ax.coastlines(color='black', alpha=map_alpha)
        ax.add_feature(cfeature.BORDERS, color='black', alpha=map_alpha)

        # Add formatted date string
        date_str = '{:%Y%m%d %H:%M} + {:d} hours'.format(self.date(), self.fct_hour)
        ax.text(0, 0, date_str,
                horizontalalignment='left',
                verticalalignment='bottom',
                size='large',
                backgroundcolor='#ffffff',
                transform=ax.transAxes)

        DATA_SOURCE = 'NCEP/NOAA'
        DATA_LICENSE = 'public domain'

        # Add a text annotation for the license information
        text = AnchoredText(f'Data: {DATA_SOURCE} ({DATA_LICENSE})',
                            loc='lower right', borderpad=0, pad=0,
                            prop=dict(size=9, backgroundcolor='#ffffff'),
                            frameon=False)
        ax.add_artist(text)

        return fig, ax

class LowLevelChart(SynopticChart):
    """Low-level chart displaying windspeed, streamlines, ITD based on
    dewpoint temperature, mean sea level pressure and pressure
    tendency.

    Note that this chart has been developed for West Africa. Extension
    to other domains may require some adjustments.

    """

    def __init__(self, domain, fct_timestamp, fct_hour, data_dir=None):
        super().__init__(domain, fct_timestamp, fct_hour, data_dir)

        self.chart_type = "low-level"

        # Mean sea level pressure
        self.mslp = MeanSeaLevelPressure(self)

        # Inter-tropical discontinuity
        self.itd = ITD(self)

        # Windspeed and streamlines at 925 hPa
        self.wc_925 = WindPressureLevel(self, 925)

        # 24 hour change in mean sea level pressure
        self.mslp_24 = MeanSeaLevelPressureChange(self)

        # Windspeed at 10m, 15 m/s contour
        self.wc_10m = WindHeightLevel(self, 10)

class WAJetsWaves(SynopticChart):
    """Chart displaying jets and waves for West Africa.

    Features to be plotted:
    * AEJ
    * STJ
    * TEJ
    * AEW troughs and ridges
    * PW* or Monsoon depth (to be renamed moisture depth) MD as filled contours
    * Monsoon trough [but this might be too difficult in practice]
    * Cyclonic centres at 850 hPa and 700/600 hPa level.
    * Possibly, dry air boundaries.

    """

    def __init__(self, domain, fct_timestamp, fct_hour, data_dir=None):
        super().__init__(domain, fct_timestamp, fct_hour, data_dir)

        self.chart_type = "WA-jets-waves"

        self.aej = AfricanEasterlyJet(self)

class SynthesisChart(SynopticChart):
    """Synthesis chart displaying key features for analysis.

    - a contour plot of 925-650 hPa wind shear
    - low level streamlines, no shading
    - overplot ITD and AEJ (and AEW if possible)
    - overplot contours of RH700 = 60% and 75%
    """

    def __init__(self, domain, fct_timestamp, fct_hour, data_dir=None):
        super().__init__(domain, fct_timestamp, fct_hour, data_dir)

        self.chart_type = "synthesis"

        # Inter-tropical discontinuity
        self.itd = ITD(self)

        # African Easterly Jet
        self.aej = AfricanEasterlyJet(self, 600)
        self.aej.plot_ws = False

        # Windspeed and streamlines at 925 hPa
        self.wc_925 = WindPressureLevel(self, 925)
        self.wc_925.plot_ws = False
        self.wc_925.strm_options['color'] = 'black'
        self.wc_925.strm_options['linewidth'] = 0.7

        # Mid-level dry intrusion - relative humidity contours at 700 hPA
        self.mdi_700 = MidlevelDryIntrusion(self, 700)

        # 925-650hPa wind shear
        self.ws_925_650 = WindShear(self, 925, 650)

#-----------------------------------------

# Synoptic components

#-----------------------------------------

class SynopticComponent:
    """ Synoptic chart component """

    # GFS variables lookup
    GFS_VARS = {
        # dewpoint temperature at 2m
        'dpt_2m': 'DPT_P0_L103_GLL0',
        # precipitable water
        'pwat': 'PWAT_P0_L200_GLL0',
        # mean sea level pressure
        'prmsl': 'PRMSL_P0_L101_GLL0',
        # relative humidity
        'rh': 'RH_P0_L100_GLL0',
        # specific humidity, specified height level above ground ('lv_HTGL4')
        'spfh': 'SPFH_P0_L103_GLL0',
        # temperature on pressure levels
        'tmp': 'TMP_P0_L100_GLL0',
        # wind components on pressure levels
        'u_lvp': 'UGRD_P0_L100_GLL0',
        'v_lvp': 'VGRD_P0_L100_GLL0',
        # wind components at specific altitude (m) above MSL
        'u_lva': 'UGRD_P0_L102_GLL0',
        'v_lva': 'VGRD_P0_L102_GLL0',
        # wind components at specific height (m) above ground
        'u_lvh': 'UGRD_P0_L103_GLL0',
        'v_lvh': 'VGRD_P0_L103_GLL0',
        # surface pressure??
        'pres': 'PRES_P0_L1_GLL0',
    }

    def __init__(self, chart, level=None):
        self.init()
        self.chart = chart
        self.data = chart.get_data(self.gfs_vars, units=self.units)
        if level is not None:
            self.level = level
            # Constrain data to specified level
            lv_coord = gfs_utils.get_level_coord(self.data, self.level_units)
            cc = gfs_utils.get_coord_constraint(lv_coord.name(), self.level)
            self.data = self.data.extract(cc)
        #self.backend = chart.get_backend()
        coords = ('latitude', 'longitude')
        if isinstance(self.data, iris.cube.CubeList):
            self.lat, self.lon = [self.data[0].coord(x).points for x in coords]
        else:
            self.lat, self.lon = [self.data.coord(x).points for x in coords]
        chart.add_component(self)

    def init():
        """
        Derived classes should implement this method to initialise
        self.gfs_vars and self.units, together with any class specific
        initialisation (e.g. formatting options).
        """
        raise RuntimeError("Not implemented")

    def plot(self, ax):
        # might change this to pass in chart then have a method in
        # chart that returns the plotting apparatus
        raise RuntimeError("Not implemented")

    def get_masked_colormap(self, name=None, cm_range=[0,1], res=256, alpha=None,
                            thres_min=None, thres_max=None,
                            val_min=0, val_max=1, mask_color=[1, 1, 1, 0]):
        """
        Get colour map with transparency-based masking for values outside
        a specified range.

        Using this method means that the colours in the colour map are
        assigned across the range of values in the data rather than
        being compressed to the range of unmasked values as would be
        the case when applying a colour mask to masked data.

        """
        cm_min, cm_max = cm_range

        # Create a Colormap object which can be used to get a
        # hi-res RGBA array of values for the named colour map
        cmap_hi = mcm.get_cmap(self.cm_name if name is None else name, 2*res)

        # Create a colour map based on specified range
        cmap_sampled = mc.ListedColormap(cmap_hi(np.linspace(cm_min, cm_max, res)))
        cmap = cmap_sampled(np.linspace(0, 1, res))

        if alpha is not None:
            # Set alpha transparency
            cmap[:, -1] = alpha

        mask_val = np.array(mask_color)

        if self.cm_thres is not None:
            thres_min, thres_max = self.cm_thres

        if thres_min is not None:
            # Mask colour map for values below thres_min
            cmap[:round(thres_min/(val_max - val_min)*res), :] = mask_val

        if thres_max is not None:
            # Mask colour map for values above thres_max
            cmap[round(thres_max/(val_max - val_min)*res):, :] = mask_val

        return mc.ListedColormap(cmap)

class MeanSeaLevelPressure(SynopticComponent):
    """
    Mean sea level pressure.
    """

    def __init__(self, chart):
        super().__init__(chart)

    def init(self):
        self.name = "Mean Sea Level Pressure"
        self.gfs_vars = SynopticComponent.GFS_VARS['prmsl']
        self.units = 'hPa'

        # Default to plot contour lines at 5 hPa intervals, 1015 bold
        self.step = 5
        self.highlight = 1015

        # Formatting options
        self.options = {
            'alpha': 0.6,
            'colors': 'purple',
        }

    def plot(self, ax):

        # Data to plot
        mslp = self.data.data

        pmin = np.amin(mslp)
        pmax = np.amax(mslp)
        levels = np.arange(pmin - pmin % self.step,
                           pmax + self.step - pmax % self.step,
                           self.step)
        lw = np.array([1.6 if x == self.highlight else 0.8 for x in levels])

        # Plot mean sea level pressure contours
        ctr = ax.contour(self.lon, self.lat, mslp,
                         levels=levels,
                         linewidths=lw,
                         **self.options)

class MeanSeaLevelPressureChange(MeanSeaLevelPressure):
    """
    Change in mean sea level pressure over 24 hours.
    """
    def __init__(self, chart):
        super().__init__(chart)

    def init(self):
        self.name = "Mean Sea Level Pressure Change"
        self.gfs_vars = SynopticComponent.GFS_VARS['prmsl']
        self.units = 'hPa'

        # Inspect MSLP change over past 24 hours
        self.delta = -24

        # Plot lines at 0.5 hPa intervals, skip contours in [-0.5, 0.5] interval
        self.step = 0.5
        self.thres = 0.5

        # Formatting options
        self.options = {
            'colors': 'orange',
        }

    def plot(self, ax):

        try:
            mslp_m24 = self.chart.get_data(self.gfs_vars, units=self.units,
                                           delta=dt.timedelta(hours=self.delta))
        except ValueError:
            msg = "{:d} hour data not available, cannot plot mean sea level pressure change".format(self.delta)
            warnings.warn(msg)
            return

        delta_mslp = self.data.data - mslp_m24.data

        dp_min = np.amin(delta_mslp)
        dp_max = np.amax(delta_mslp)
        levels = np.arange(dp_min - dp_min % self.step,
                           dp_max + self.step - dp_max % self.step,
                           self.step)
        levels = np.delete(levels, np.argwhere(np.abs(levels) <= self.thres))
        linestyles = np.where(levels < 0, '--', '-')

        ctr = ax.contour(self.lon, self.lat, delta_mslp,
                         levels=levels,
                         linestyles=linestyles,
                         **self.options)
        ax.clabel(ctr, fmt = '%1.1f')

class ITD(SynopticComponent):
    """
    Inter-tropical discontinuity (ITD), based on 15 deg C dewpoint
    temperature contour.

    """

    # could include options for type, masking etc.
    def __init__(self, chart):
        super().__init__(chart)

    def init(self):
        self.name = "Inter-tropical Discontinuity"
        self.gfs_vars = SynopticComponent.GFS_VARS['dpt_2m']
        self.units = 'Celsius'

        # Tolerance for gradient-based masking
        self.tol = 0.9
        self.dewpoint_level = [ 15.0 ]

        # Minimum number of vertices to include when plotting ITD contours
        self.min_vertices = 10

        # Formatting options
        self.options = {
        }

        # Plotting options
        self.lw = 2.0
        self.col = 'black'
        self.col2 = 'white'

    def plot(self, ax):

        # Calculate gradient of dewpoint temp with latitude
        gradient_dpt_wrt_lat = iris.analysis.calculus.differentiate(self.data, 'latitude')

        # Regrid dewpoint temperature to match gradient data
        dpt_2m_regridded = self.data.regrid(gradient_dpt_wrt_lat,
                                            iris.analysis.Linear(extrapolation_mode='extrapolate'))

        # Mask dpt_2m_regridded if gradient exceeds tolerance
        dpt_2m_masked = iris.util.mask_cube(dpt_2m_regridded,
                                            gradient_dpt_wrt_lat.data > self.tol)

        dewpoint = dpt_2m_masked.data
        self.lat = dpt_2m_masked.coord('latitude').points

        # Plot masked dewpoint temperature contour (ITD)
        itd1 = ax.contour(self.lon, self.lat, dewpoint,
                          levels = self.dewpoint_level,
                          colors = self.col, linewidths = 3*self.lw)
        itd2 = ax.contour(self.lon, self.lat, dewpoint,
                          levels = self.dewpoint_level,
                          colors = self.col2, linewidths = self.lw,
                          linestyles = 'dashed')

        # Clean up contours
        self.clean_contour(itd1)
        self.clean_contour(itd2)

    def clean_contour(self, ctr):
        """Clean up contour by getting rid of short paths"""
        for lc in ctr.collections:
            paths = lc.get_paths()
            new_paths = [path for path in paths if len(path.vertices) >= self.min_vertices]
            del paths[:]

            # set lc._paths directly because lc.set_paths() is buggy
            lc._paths = new_paths

class WindComponent(SynopticComponent):
    """
    Base class for wind-related synoptic components, providing utility
    method to extract U/V wind components and windspeed for a given level
    in the data.

    """

    def __init__(self, chart, level):
        super().__init__(chart)
        self.level = level

    def get_wind_components(self, **kwargs):
        """
        Extract U/V wind components and windspeed for given level(s)
        """
        level = kwargs.get("level", self.level)

        # Get U/V wind components
        u, v = self.data.extract(self.gfs_vars)

        # Get level coordinates for U/V wind components
        u_lv_coord = gfs_utils.get_level_coord(u, self.level_units)
        v_lv_coord = gfs_utils.get_level_coord(v, self.level_units)

        # Constrain to specified level(s)
        uc = gfs_utils.get_coord_constraint(u_lv_coord.name(), level)
        u = u.extract(uc)

        vc = gfs_utils.get_coord_constraint(v_lv_coord.name(), level)
        v = v.extract(vc)

        if type(level) is list:
            U = []
            V = []
            windspeed = []
            for lvl in level:
                ui = u.extract(gfs_utils.get_coord_constraint(u_lv_coord.name(), lvl))
                vi = v.extract(gfs_utils.get_coord_constraint(v_lv_coord.name(), lvl))
                U.append(ui.data)
                V.append(vi.data)
                windspeed.append(np.sqrt(ui.data**2 + vi.data**2))
        else:
            U = u.data
            V = v.data
            windspeed = np.sqrt(U**2 + V**2)

        return (U, V, windspeed)

class WindPressureLevel(WindComponent):
    """
    Wind at specified pressure level

    """

    def __init__(self, chart, level):
        super().__init__(chart, level)

    def init(self):
        self.name = "Wind on Pressure Level"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x) for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ws = True
        self.plot_strm = True

        self.min_ws = 0
        self.max_ws = 50
        self.thres_ws = 10

        self.cm_name = 'Blues'
        self.cm_thres = [ self.thres_ws, None ]

        # Formatting options
        self.ws_options = {
            'cmap': self.cm_name,
        }

        self.strm_options = {
            'density': (2.5, 1.66),
            # 'color': 'black',
            # 'color': speed,
            # 'cmap': strm_cmap,
            'linewidth': 0.4,
            'arrowsize': 0.9,
            'arrowstyle': '->',
            # 'norm': cnorm,
        }

    def plot(self, ax):

        U, V, windspeed = self.get_wind_components()

        if self.plot_ws:
            # Plot windspeed at 925 hPa
            max_ws = np.amax(windspeed)
            self.ws_options['cmap'] = self.get_masked_colormap(val_max=max_ws, alpha=0.4)
            ctr = ax.contourf(self.lon, self.lat, windspeed,
                              **self.ws_options)

        if self.plot_strm:
            # Plot streamlines
            strm = ax.streamplot(self.lon, self.lat, U, V,
                                 **self.strm_options)

class WindHeightLevel(WindComponent):
    """
    Wind at specified height level

    """
    def __init__(self, chart, level):
        super().__init__(chart, level)

    def init(self):
        self.name = "Windspeed at Height Level"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x) for x in ('u_lvh', 'v_lvh')]
        self.units = None
        self.level_units = 'm'

        self.ws_level = [ 15.0 ]

        self.options = {
            'alpha': 0.6,
            'linewidths': 1.6,
            'colors': 'darkgreen',
        }

    def plot(self, ax):

        # Get U/V wind components
        _, _, windspeed = self.get_wind_components()

        # Plot 10m windspeed 15 m/s contour
        ctr = ax.contour(self.lon, self.lat, windspeed,
                         levels = self.ws_level,
                         **self.options)

class AfricanEasterlyJet(WindComponent):

    def __init__(self, chart, level=650):
        super().__init__(chart, level)

    def init(self):
        self.name = "African Easterly Jet"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x) for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ws = True
        self.plot_core = True
        self.plot_strm = True

        self.min_ws = 0
        self.max_ws = 25
        self.thres_axis = 10
        self.thres_core = 23
        self.thres_windshear = 15

        self.cm_name = 'Greens'
        self.cm_thres = [ self.thres_axis, None ]

        # Formatting options
        self.ws_options = {
            'cmap': self.cm_name,
        }

        self.core_options = {
            'edgecolor': 'black',
            'fill': None,
            'linewidth': 1.0,
        }
        self.core_label_fontsize = 8.0
        self.core_label_units = f'ms\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}'
        self.core_label_options = {
            'xytext': np.array([-2, -1.25])*self.core_label_fontsize,
            'textcoords': 'offset points',
            'fontsize': self.core_label_fontsize,
        }

        self.strm_options = {
            'color': 'green',
            # 'cmap': cmap,
            'linewidth': 2.0,
            'arrowsize': 3.0,
            'arrowstyle': '-',
            # 'norm': cnorm,
        }

        self.marker_interval = 3.2

    def plot(self, ax):

        U, V, windspeed = self.get_wind_components()

        if self.plot_ws:
            # Plot windspeed contours
            max_ws = np.amax(windspeed)
            self.ws_options['cmap'] = self.get_masked_colormap(val_max=max_ws, alpha=0.4)
            ctr = ax.contourf(self.lon, self.lat, windspeed,
                              **self.ws_options)

        if self.plot_core:
            # Mask windspeed to relevant region i.e. between 10-15 deg N
            lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
            mask = (lat_grid < 5) | (lat_grid > 20)
            ws_masked = np.ma.masked_where(mask, windspeed)
            ws_masked[mask] = np.nan

            # Get contours for core threshold windspeed
            ctr_list = skimage.measure.find_contours(ws_masked, level=self.thres_core)
            for c in ctr_list:
                shp = sgeom.Polygon(c)

                # Translate centroid coordinates to lon/lat
                centroid = shp.centroid.coords[0]
                xy = [self.lon[np.rint(centroid[1]).astype(int)],
                      self.lat[np.rint(centroid[0]).astype(int)]]

                # Use contour bounds to get dimensions for core
                it = iter(shp.bounds)
                bounds = np.array([[self.lon[np.rint(next(it)).astype(int)],
                                    self.lat[np.rint(x).astype(int)]] for x in it])
                dx, dy = np.abs(bounds[1] - bounds[0])

                # Draw core as ellipse around centroid
                p = mpatches.Ellipse(xy=xy,
                                     width=dx,
                                     height=dy,
                                     angle=0,
                                     **self.core_options)
                ax.add_artist(p)

                # Add label to indicate core threshold windspeed
                loc = xy + np.array([0, dy/2])
                ax.annotate(f'{self.thres_core:1.0f} {self.core_label_units}',
                            xy=loc,
                            **self.core_label_options)

        if self.plot_strm:
            # Plot streamlines connecting maximum wind location

            # mask to relevant region i.e. between 10-15 deg N
            lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
            mask = (lat_grid < 5) | (lat_grid > 20)

            # find max windspeed for the relevant region
            ws_masked = np.ma.masked_where(mask, windspeed)
            max_ws = np.amax(ws_masked)

            # select seed points
            seed_index = np.argwhere(ws_masked > max_ws*0.85)
            seed_points = np.array([[self.lon[x[1]], self.lat[x[0]]] for x in seed_index])

            # Define mask for jet axis
            mask1 = (ws_masked < self.thres_axis)

            # Get wind shear components
            Us, Vs, _ = self.get_wind_components(level=[650, 925])

            UU = Us[1] - Us[0]
            VV = Vs[1] - Vs[0]

            wind_shear = np.sqrt(UU**2 + VV**2)

            # Define mask for wind shear
            mask2 = wind_shear < self.thres_windshear

            # Mask U, V to mask streamlines below thresholds for
            # windspeed and wind shear
            mask = mask1 & mask2

            # U = np.ma.array(U, mask=mask)
            # V = np.ma.array(V, mask=mask)
            U[mask] = np.nan
            V[mask] = np.nan

            strm = ax.streamplot(self.lon, self.lat, U, V,
                                 start_points=seed_points,
                                 **self.strm_options)

            arrow_kw = self.strm_options
            arrow_kw['mutation_scale'] = 10 * self.strm_options["arrowsize"]
            arrow_kw.pop('arrowsize')
            arrow_kw['arrowstyle'] = '->'

            # Get line segments and place arrows
            current_point = None
            segments = strm.lines.get_segments()
            for seg in segments:
                for i, s in enumerate(seg[:-1]):
                    if not all(s == current_point):
                        # Reset distance sum
                        dist_sum = 0
                    # Check length of segment and add patches at
                    # suitable intervals
                    dx, dy = seg[i+1] - s
                    seg_len = np.hypot(dx, dy)
                    dist_sum = dist_sum + seg_len
                    while dist_sum > self.marker_interval:
                        dist_sum -= self.marker_interval
                        # Find start and end points for arrow patch
                        v = np.array([dx, dy])
                        loc = (seg_len - dist_sum)/seg_len
                        start = s + loc*v
                        end = start + 0.1*v/seg_len
                        # Draw patch
                        p = mpatches.FancyArrowPatch(
                            start, end, transform=ax.transData,
                            **arrow_kw
                        )
                        ax.add_patch(p)
                    current_point = seg[i+1]

class MidlevelDryIntrusion(SynopticComponent):
    """
    Mid-level dry intrusion

    Relative humidity at 700hPa, 60% and 75% contours
    """
    def __init__(self, chart, level):
        super().__init__(chart, level)

    def init(self):
        self.name = "Mid-level dry intrusion"
        self.gfs_vars = SynopticComponent.GFS_VARS['rh']
        self.units = 'percent'
        self.level_units = 'hPa'

        self.levels = [ 60 ]

        self.lw = 3.0

        self.marker_thres = 0.8

        # Formatting options
        self.options = {
            'linewidths': [ self.lw ],
            'linestyles': [ 'solid' ],
            'colors': '#198dd5'
        }

    def plot(self, ax):

        # Data to plot
        rh = self.data.data

        # Plot relative humidity contours
        ctr = ax.contour(self.lon, self.lat, rh,
                         levels=self.levels,
                         **self.options)
        #ax.clabel(ctr, fmt = '%1.0f')

        # Draw short line at right angles to line segments to indicate
        # dry side of contour
        for lc in ctr.collections:
            segments = lc.get_segments()
            for seg in segments:
                dist_sum = self.marker_thres/2
                for i, s in enumerate(seg[:-1]):
                    # Check length of segment and add patches at
                    # suitable intervals
                    dx, dy = seg[i+1] - s
                    seg_len = np.linalg.norm([dx, dy])
                    dist_sum = dist_sum + seg_len
                    while dist_sum > self.marker_thres:
                        dist_sum -= self.marker_thres
                        # Find start point for marker
                        loc = (seg_len - dist_sum)/seg_len
                        start = s + loc*np.array([dx, dy])
                        # Find end point for marker
                        v = np.array([-dy, dx])
                        end = start + 0.33*v/np.linalg.norm(v)
                        # Draw path
                        pth = mpath.Path([start, end])
                        line = mpatches.PathPatch(pth,
                                                  color=self.options["colors"],
                                                  linewidth=1.4*self.lw,
                                                  capstyle='butt')
                        ax.add_patch(line)

class WindShear(WindComponent):

    def __init__(self, chart, level1, level2):
        super().__init__(chart, [level1, level2])

    def init(self):
        self.name = "Wind shear"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x) for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ctr = True
        self.plot_qv = True

        # Formatting options
        self.qv_skip = 2
        self.qv_options = {
            'color': 'black',
            'alpha': 0.2,
            'width': 0.002,  # width relative to selected units (default = axis width)
        }

        self.ws_thres = 25

        self.cm_name = 'Reds'
        self.cm_thres = [ self.ws_thres, None ]

        self.ws_options = {
            'cmap': self.cm_name,
        }

    def plot(self, ax):

        U, V, _ = self.get_wind_components()

        U_diff = U[1] - U[0]
        V_diff = V[1] - V[0]

        ws_diff = np.sqrt(U_diff**2 + V_diff**2)

        if self.plot_ctr:
            self.ws_options['cmap'] = self.get_masked_colormap(val_max=np.amax(ws_diff), alpha=0.4)
            ctr = ax.contourf(self.lon, self.lat, ws_diff,
                              **self.ws_options)

        if self.plot_qv:
            # Mask values below threshold
            U_diff = np.ma.masked_where(ws_diff < self.ws_thres, U_diff)
            V_diff = np.ma.masked_where(ws_diff < self.ws_thres, V_diff)
            qv = ax.quiver(self.lon[::self.qv_skip], self.lat[::self.qv_skip],
                           U_diff[::self.qv_skip, ::self.qv_skip],
                           V_diff[::self.qv_skip, ::self.qv_skip],
                           **self.qv_options)


#---------------------------------------------------------------

# class Template(SynopticComponent):

#     def __init__(self, chart):
#         super().__init__(chart)

#     def init(self):
#         self.name = "Name"
#         self.gfs_vars = SynopticComponent.GFS_VARS['code']
#         self.units = None

#     def plot(self, ax):
#         pass

#---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Plot synoptic chart')

    parser.add_argument('domain', type=str,
                        metavar='domain', choices=['WA', 'EA'],
                        help='''Domain specified as standardised
                        domain name (WA or EA)''')

    parser.add_argument('timestamp', type=str,
                        help='''Timestamp for chart data in format
                        \"YYYYmmddHH\"''')

    parser.add_argument('forecast_hour', type=int,
                        metavar='forecast_hour', choices=range(0,73,3),
                        help='''Forecast hour as non-negative integer
                        multiple of 3 (max 72)''')

    parser.add_argument('chart_type', nargs='?', type=str,
                        metavar="chart_type", choices=["low", "jets", "conv", "synth"],
                        default="low",
                        help='''Chart type (low, jets, conv or synth)''')

    parser.add_argument('-o', '--output-dir', nargs='?', type=str,
                        dest='output_dir', default=None,
                        help="Path to output directory")

    pa = parser.parse_args()

    # Check if output directory exists
    if pa.output_dir and not os.path.exists(pa.output_dir):
        err_msg = "Output directory {0} does not exist\n"
        err_msg = err_msg.format(pa.output_dir)
        raise ValueError(err_msg)

    return (pa.domain, pa.timestamp, pa.forecast_hour, pa.chart_type, pa.output_dir)

def main():

    domain, timestamp, hour, chart_type, out_dir = parse_args()

    if chart_type == "low":
        chart = LowLevelChart(domain, timestamp, hour)
    elif chart_type == "jets":
        chart = WAJetsWaves(domain, timestamp, hour)
    elif chart_type == "conv":
        chart = ConvectiveChart(domain, timestamp, hour)
    elif chart_type == "synth":
        chart = SynthesisChart(domain, timestamp, hour)
    else:
        raise ValueError("Unrecognised chart type")

    print(chart)

    chart.build(out_dir)

    #end main()

if __name__ == '__main__':
    main()
