#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWIFT automated synoptic charts

Plot automatically-generated charts of synoptic weather features such
as jets, convergence lines, troughs and waves across African domains.

This module was developed by CEMAC as part of the GCRF African Swift
Project.

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

        <chart_type> Chart type (low, jets, conv or synth) (default: low)

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)

import os
import sys
import datetime as dt
import argparse
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import iris
import numpy as np

# from * import gfs_utils  # works for tests but not when calling module code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '.')))
import gfs_utils  # works when calling module code
from component import *


class SynopticChart:
    """
    Synoptic chart base class

    This class provides functionality for setting forecast domain,
    date and time, building charts, and plotting.

    """

    # Coords for standard domains from GFS_plotting/controls/domains
    # (could read in programatically from similar file)
    # Values are (lat_min, lon_min, lat_max, lon_max)
    # (could set up with value as {lat_min: -16.0, etc})

    # Domain specification is a tuple specifying (lat_min, lon_min,
    # lat_max, lon_max).
    DOMAINS = {
        'WA': (0.0, -25.0, 40.0, 35.0),
        'EA': (-16.0, 18.0, 21.0, 52.0),
        # Pan Africa: 60 S to 60 N, 60 W to 90 E.
        'PA': (-60.0, -60.0, 60.0, 90.0),
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
            'subplot_kw': {'projection': ccrs.PlateCarree()},
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
        return dt.datetime.strptime(self.fct_timestamp,
                                    '%Y%m%d%H').strftime("%Y%m%d %H:%M")

    def set_domain(self, arg):
        arg_type = type(arg)
        if arg_type is str:
            if arg == 'WA' or arg == 'WestAfrica':
                self.domain = SynopticChart.DOMAINS['WA']
            elif arg == 'EA' or arg == 'EastAfrica':
                self.domain = SynopticChart.DOMAINS['EA']
            elif arg == 'PA' or arg == 'PanAfrica':
                self.domain = SynopticChart.DOMAINS['PA']
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
                raise ValueError("Domain does not contain the right number "
                                 "of values")
        else:
            raise ValueError("Unrecognised domain type")

    def get_domain_extent(self):
        """
        Get longitudinal and latitudinal extent of this chart's domain
        """
        domain = np.array(self.domain)
        lon_lat = domain.reshape((2, 2), order='F')[::-1]
        return np.diff(lon_lat).flatten()

    def set_timestamp(self, arg):
        try:
            if arg != (dt.datetime.strptime(arg, '%Y%m%d%H')
                       .strftime('%Y%m%d%H')):
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
        prefix, suffix = (('GFS_forecast_', '.nc')
                          if (self.fct_hour > 0)
                          else ('analysis_gfs_4_', '00_000.nc'))
        file_name = '{}{:%Y%m%d_%H}{}'.format(prefix, ts, suffix)
        file_path = os.path.join(self.data_dir, "GFS_NWP",
                                 timestamp, file_name)
        if not os.path.isfile(file_path):
            raise ValueError("{:s} is not a file", file_path)
        return file_path

    def get_data(self, gfs_vars, units=None, delta=None, apply_domain=True):

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
            c = c.intersection(longitude=(-180, 180))

            if units is not None:
                # Convert units
                c.convert_units(units)

            # Constrain to specified forecast hour for this chart
            fct_date = dt.datetime.strptime(c.attributes['initial_time'],
                                            '%m/%d/%Y (%H:%M)')
            time_constraint = gfs_utils.get_time_constraint(fct_date,
                                                            self.fct_hour)
            c = c.extract(time_constraint)

            if apply_domain:
                # Constrain to specified domain
                domain_constraint = gfs_utils.get_domain_constraint(self.domain)
                c = c.extract(domain_constraint)

            rtn.append(c)

        return rtn if len(rtn) > 1 else rtn[0]

    def add_component(self, component):
        self.components.append(component)

    def build(self, dir_path=None, scale_factor=10):
        """Build chart from components"""
        if dir_path is not None:
            # Use non-interactive backend
            mpl.use('agg')
            # Adjust output scaling according to domain extent
            _, dlat = self.get_domain_extent()
            scale_factor = scale_factor*dlat/40

        if self.components:
            sample_data = self.components[0].data
            if isinstance(sample_data, iris.cube.CubeList):
                sample_data = sample_data[0]
            # Set sensible size in inches based on coordinate array extent
            coords = ('longitude', 'latitude')
            shape = np.array([sample_data.coord(x).points.size
                              for x in coords])
            self.mpl_options['figsize'] = shape/scale_factor

        # set up chart - this might be configurable to use different
        # plotting back ends
        fig, ax = self.setup_plot()

        for c in self.components:
            # may need some concept of "order"?
            # add each component to plot
            c.plot(ax)

        lat_min, lon_min, lat_max, lon_max = self.domain
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        if dir_path is not None:
            file_name = ('{:%Y%m%d_%H%M}_{:03d}_{}_{}.png'
                         .format(self.date(), self.fct_hour,
                                 self.domain_name, self.chart_type.lower()))
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
        grid_col = '#53606d'
        grid_label_style = {'color': grid_col}
        grid_step = 20 if self.domain_name == 'PA' else 10
        gl = ax.gridlines(draw_labels=True, x_inline=True, y_inline=True,
                          zorder=1.5, color=grid_col)
        gl.xlocator = mticker.FixedLocator(np.arange(-60, 90, grid_step))
        gl.ylocator = mticker.FixedLocator(np.arange(-60, 60, grid_step))
        gl.xlabel_style = grid_label_style
        gl.ylabel_style = grid_label_style

        # Add coastlines and borders
        map_alpha = 0.9
        ax.coastlines(color='black', alpha=map_alpha)
        ax.add_feature(cfeature.BORDERS, color='#333333', alpha=map_alpha, linewidth=1.5)

        ax.add_feature(cfeature.LAND, alpha=0.5)
        ax.add_feature(cfeature.LAKES, alpha=0.5, edgecolor='#6e9ee1')
        ax.add_feature(cfeature.OCEAN, alpha=0.5)

        # Add formatted date string
        date_str = '{:%Y%m%d %H:%M} + {:d} hours'.format(self.date(),
                                                         self.fct_hour)
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
    """
    Chart displaying domain specific low-level features

    Low-level chart displaying combination of windspeed, streamlines,
    ITD based on dewpoint temperature, mean sea level pressure and
    pressure tendency. Features are selected as appropriate for the
    specified domain.

    """

    def __init__(self, domain, fct_timestamp, fct_hour, data_dir=None):
        super().__init__(domain, fct_timestamp, fct_hour, data_dir)

        self.chart_type = "low-level"

        if self.domain_name == 'WA':
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

        elif self.domain_name == 'EA':
            # Mean sea level pressure
            self.mslp = MeanSeaLevelPressure(self)
            self.mslp.lw = 2.0
            self.mslp.highlight = 1012
            self.mslp.step = 4

            # 24 hour change in mean sea level pressure
            self.mslp_24 = MeanSeaLevelPressureChange(self)

            # Convergence of 10m winds
            self.wc_10m = WindHeightLevel(self, 10)
            self.wc_10m.plot_strm = True
            self.wc_10m.strm_options['color'] = '#00619e'

            # Windspeed at 10m, 25kt = 12.86 m/s contour
            self.wc_10m.plot_ws = True
            self.wc_10m.ws_level = [12.86]
            self.wc_10m.highlight = 12.86
            self.wc_10m.lw = 1.5

            # Streamlines at 700 hPa
            self.wc_700 = WindPressureLevel(self, 700)
            self.wc_700.plot_ws = False
            self.wc_700.strm_options['color'] = 'black'

            # Relative humidity at 700 hPa above 80%
            self.rh_700 = MidlevelDryIntrusion(self, [700], [80])
            self.rh_700.options['colors'] = '#00c7ff'
            self.rh_700.options['linewidths'] = [2.0]
            self.rh_700.label_contours = True

            # Dewpoint temperature at 2m
            self.dpt = DPT(self)

            # Mid level dry intrusion
            self.mdi = MidlevelDryIntrusion(self)

        elif self.domain_name == 'PA':
            # Mean sea level pressure
            self.mslp = MeanSeaLevelPressure(self)
            self.mslp.lw = 1.0

            # Plot MSLP at specified levels
            self.mslp.levels = [960, 980, 990, 1000, 1004, 1008, 1012,
                                1016, 1020, 1024, 1028, 1032, 1036, 1040]
            self.mslp.highlight = 1020

            # 24 hour change in mean sea level pressure
            self.mslp_24 = MeanSeaLevelPressureChange(self)
            self.mslp_24.levels = [-12.0, -8.0, -4.0, -2.0,
                                   2.0, 4.0, 8.0, 12.0, 16.0, 20.0]

            # Windspeed and streamlines at 925 hPa
            self.wc_925 = WindPressureLevel(self, 925)

            # Mid level dry intrusion
            self.mdi = MidlevelDryIntrusion(self)
            self.mdi.marker_thres = 2.0
            self.mdi.marker_scale = 1.0


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

        if self.domain_name == 'WA':

            self.chart_type = "WA-jets-waves"

            self.aej = AfricanEasterlyJet(self, 600)
            #self.aej.plot_ws = False
            self.aej.ws_options['alpha'] = 0.01

            # Windspeed and streamlines at 600 hPa for diagnosis of AEWs
            # self.wc_600 = WindPressureLevel(self, 600)
            # self.wc_600.plot_ws = False
            # self.wc_600.strm_options['color'] = 'black'
            # self.wc_600.strm_options['linewidth'] = 0.7

            # Tropical Easterly Jet
            self.tej100 = TropicalEasterlyJet(self, 100)
            self.tej200 = TropicalEasterlyJet(self, 200)

            # Subtropical Jet
            self.stj = SubtropicalJet(self)

            # African Easterly Waves
            self.aew = AfricanEasterlyWaves(self)

            # Moisture depth
            self.md = MoistureDepth(self)
            self.md.cm_alpha = 0.6

            # Monsoon Trough
            self.mt = MonsoonTrough(self)

        elif self.domain_name == 'EA':
            self.chart_type = "winds"

            # Convergence of 10m winds
            self.wc_10m = WindHeightLevel(self, 10)
            self.wc_10m.plot_strm = True
            self.wc_10m.strm_options['color'] = '#00619e'

            # Streamlines at 300 hPa
            self.wc_300 = WindPressureLevel(self, 300)
            self.wc_300.plot_ws = False
            self.wc_300.strm_options['color'] = '#aa0000'

            # Streamlines at 700 hPa
            self.wc_700 = WindPressureLevel(self, 700)
            self.wc_700.plot_ws = False
            self.wc_700.strm_options['color'] = 'black'

            # Upper level convergence/divergence
            self.div = Divergence(self, 300)

        elif self.domain_name == 'PA':
            self.chart_type = "winds"

            # Streamlines at 925 hPa
            self.wc_925 = WindPressureLevel(self, 925)
            self.wc_925.plot_ws = False

            # Streamlines at 300 hPa
            self.wc_300 = WindPressureLevel(self, 300)
            self.wc_300.strm_options['color'] = '#aa0000'
            self.wc_300.thres_ws = 51.44  # = 100 kt
            self.wc_300.cm_name = 'Reds'

            # Upper level convergence/divergence
            self.div = Divergence(self, 300)

            # Sub-tropical Jet at 200 hPa
            self.stj = SubtropicalJet(self)
            self.stj.arrow_size = 0.6
            self.stj.arrow_interval = 6.0


class ConvectiveChart(SynopticChart):
    """Chart displaying convection for West Africa.

    Features to be plotted:
    * Measures of convectively favourable conditions. Consider a choice or
    combination of
      - PW/Moisture depth.
      - CAPE or K Index
      - CIN
    * Dynamical features favouring convection:
      - Low level convergence
      - regions of significant 24h pressure difference (from low-level chart)
      - midlevel (maybe 850 hPa) vortices?

    * Features from the other charts:
      - topography;
      - dry intrusions;
      - AEJ and wind-shear.

    """

    def __init__(self, domain, fct_timestamp, fct_hour, data_dir=None):
        super().__init__(domain, fct_timestamp, fct_hour, data_dir)

        self.chart_type = "convective"

        # Low pressure levels suitable to domain
        if self.domain_name == 'WA':
            plvl = 925
            mdlvl = 850
        elif self.domain_name == 'EA':
            plvl = 700
            mdlvl = 800

        #self.pwat = PWAT(self)
        self.md = MoistureDepth(self, mdlvl)
        self.cape = CAPE(self)
        if self.domain_name == 'WA':
            self.cape.plot_fill = True
            self.cape.cm_alpha = 0.7
            self.cape.cm_thres = [900, None]
            self.cape.cm_range = [0, 0.5]
            self.cape.cm_name = 'twilight'
            self.cape.label_fill = True
            self.cape.label_col = '#5a3397'
            #self.cape.levels = [900, 1400, 1900]
        elif self.domain_name == 'EA':
            self.cape.levels = [1000, 2000, 3000]
        self.cin = CIN(self)

        if self.domain_name == 'WA':
            # Inter-tropical discontinuity
            self.itd = ITD(self)

        # Streamlines at level suitable to domain
        self.wind = WindPressureLevel(self, plvl)
        self.wind.plot_ws = False
        self.wind.strm_options['color'] = 'black'
        self.wind.strm_options['linewidth'] = 0.7

        # 24 hour change in mean sea level pressure
        self.mslp_24 = MeanSeaLevelPressureChange(self)

        # Mid level (850 hPa) vortices

        # Mid-level dry intrusion - defaults to 60% contour of
        # min(RH700, RH600, RH500)
        self.mdi = MidlevelDryIntrusion(self)

        if self.domain_name == 'WA':
            # African Easterly Jet
            self.aej = AfricanEasterlyJet(self, 600)
            self.aej.ws_options['alpha'] = 0.01

        if self.domain_name == 'WA':
            # 925-650hPa wind shear
            self.ws_925_650 = WindShear(self, 925, 650)
            self.ws_925_650.ws_thres = 15
            self.ws_925_650.qv_options['alpha'] = 0.3
        elif self.domain_name == 'EA':
            # 10m windspeed - 650 hPa windspeed divided by pressure
            # difference
            pass


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

        # Low pressure level suitable to domain
        if self.domain_name == 'WA':
            plvl = 925
            mdlvl = 850
        elif self.domain_name == 'EA':
            plvl = 700
            mdlvl = 800

        if self.domain_name == 'WA':

            # Moisture Depth
            self.md = MoistureDepth(self, mdlvl)

            # Inter-tropical discontinuity
            self.itd = ITD(self)

            # African Easterly Jet
            self.aej = AfricanEasterlyJet(self, 600)
            self.aej.plot_ws = False

            # African Easterly Waves
            self.aew = AfricanEasterlyWaves(self)

        elif self.domain_name == 'EA':

            # Convergence of 10m winds
            self.wc_10m = WindHeightLevel(self, 10)
            self.wc_10m.plot_strm = True
            self.wc_10m.strm_options['color'] = '#00619e'

        # Streamlines at level suitable to domain
        self.wind = WindPressureLevel(self, plvl)
        self.wind.plot_ws = False
        self.wind.strm_options['color'] = 'black'
        self.wind.strm_options['linewidth'] = 0.7

        # 24 hour change in mean sea level pressure
        self.mslp_24 = MeanSeaLevelPressureChange(self)

        # Mid-level dry intrusion - defaults to 60% contour of
        # min(RH700, RH600, RH500)
        self.mdi = MidlevelDryIntrusion(self)

        if self.domain_name == 'WA':

            # 925-650hPa wind shear
            self.ws_925_650 = WindShear(self, 925, 650)

            # Tropical Easterly Jet
            self.tej = TropicalEasterlyJet(self)

            # Subtropical Jet
            self.stj = SubtropicalJet(self)

            # Monsoon Trough
            #self.mt = MonsoonTrough(self)

        elif self.domain_name == 'EA':

            # Relative humidity at 700 hPa above 80%
            self.rh_700 = MidlevelDryIntrusion(self, [700], [80])
            self.rh_700.options['colors'] = '#00c7ff'
            self.rh_700.options['linewidths'] = [2.0]
            self.rh_700.label_contours = True

            self.cape = CAPE(self)
            self.cape.levels = [1000, 2000, 3000]


# ---------------------------------------------------------------


def parse_args():
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)

    parser.add_argument('domain', type=str,
                        metavar='domain', choices=['WA', 'EA', 'PA'],
                        help='''Domain specified as standardised
                        domain name (WA, EA or PA)''')

    parser.add_argument('timestamp', type=str,
                        help='''Timestamp for chart data in format
                        \"YYYYmmddHH\"''')

    parser.add_argument('forecast_hour', type=int,
                        metavar='forecast_hour', choices=range(0, 73, 3),
                        help='''Forecast hour as non-negative integer
                        multiple of 3 (max 72)''')

    parser.add_argument('chart_type', nargs='?', type=str,
                        metavar="chart_type",
                        choices=["low", "jets", "conv", "synth"],
                        default="low",
                        help='Chart type (low, jets, conv or synth) '
                        '(default: low)')

    parser.add_argument('-o', '--output-dir', nargs='?', type=str,
                        dest='output_dir', default=None,
                        help="Path to output directory")

    pa = parser.parse_args()

    # Check if output directory exists
    if pa.output_dir and not os.path.exists(pa.output_dir):
        err_msg = "Output directory {0} does not exist\n"
        err_msg = err_msg.format(pa.output_dir)
        raise ValueError(err_msg)

    return (pa.domain, pa.timestamp, pa.forecast_hour, pa.chart_type,
            pa.output_dir)


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

    # end main()


if __name__ == '__main__':
    main()
