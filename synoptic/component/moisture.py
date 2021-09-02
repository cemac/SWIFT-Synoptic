import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import iris.analysis
from wrf import smooth2d as wrf_smooth2d

import gfs_utils
from .component import SynopticComponent


class MidlevelDryIntrusion(SynopticComponent):
    """
    Mid-level dry intrusion

    Relative humidity, plotted as contours at specified level(s) and
    threshold(s). Defaults to min(RH700, RH600, RH500), 60% threshold.
    """
    def __init__(self, chart, level=[700, 600, 500], thres=[60]):
        super().__init__(chart, level)
        self.levels = thres

    def init(self):
        self.name = "Mid-level dry intrusion"
        self.gfs_vars = SynopticComponent.GFS_VARS['rh']
        self.units = 'percent'
        self.level_units = 'hPa'

        self.lw = 3.0

        self.marker_thres = 0.8  # spacing between markers
        self.marker_scale = 0.33  # length of marker

        self.label_contours = False
        self.label_units = '%%'

        # Formatting options
        self.options = {
            'linewidths': [self.lw],
            'linestyles': ['solid'],
            'colors': '#198dd5'
        }

    def plot(self, ax):

        if self.data.coord(self.level_coord).points.size > 1:
            # Take minimum over specified pressure levels
            rh_min = self.data.collapsed(self.level_coord, iris.analysis.MIN)
            rh = rh_min.data
        else:
            rh = self.data.data

        # Apply smoothing
        rh = wrf_smooth2d(rh, 4)

        # Mask data outside bounding box for Africa
        self.set_coord_mask(lon_min=-25, lon_max=56, lat_min=-40, lat_max=40)
        rh_masked = np.ma.masked_where(self.coord_mask, rh)

        # Plot relative humidity contours
        ctr = ax.contour(self.lon, self.lat, rh_masked,
                         levels=self.levels,
                         **self.options)
        if self.label_contours:
            ax.clabel(ctr, fmt='%1.0f'+self.label_units)

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
                    seg_len = np.hypot(dx, dy)
                    dist_sum = dist_sum + seg_len
                    while dist_sum > self.marker_thres:
                        dist_sum -= self.marker_thres
                        # Find start point for marker
                        loc = (seg_len - dist_sum)/seg_len
                        start = s + loc*np.array([dx, dy])
                        # Find end point for marker
                        v = np.array([-dy, dx])
                        end = start + self.marker_scale*v/np.linalg.norm(v)
                        # Draw path
                        pth = mpath.Path([start, end])
                        line = mpatches.PathPatch(pth,
                                                  color=self.options["colors"],
                                                  linewidth=1.4*self.lw,
                                                  capstyle='butt')
                        ax.add_patch(line)


class MoistureDepth(SynopticComponent):
    """
    Moisture depth

    """

    def __init__(self, chart):
        super().__init__(chart)

    def init(self):
        self.name = "Moisture depth"
        self.gfs_vars = SynopticComponent.GFS_VARS['pwat']
        self.units = 'kg m-2'
        self.level_units = ''

        self.lw = 1.0

        # Formatting options
        self.options = {
            'linewidths': [self.lw],
            'linestyles': ['solid'],
        }

        # Colour map settings
        self.cm_name = 'BuGn'
        self.cm_thres = [2000, None]
        # self.cm_alpha = None
        # self.cm_range = [0, 0.75]
        self.cm_alpha = 0.6
        self.cm_range = [0, 1]

    def plot(self, ax):

        # Precipitable water
        pw = self.data

        temp = self.chart.get_data(self.GFS_VARS['temp'])

        # Constrain data to specified level(s)
        lv_coord = gfs_utils.get_level_coord(temp, 'hPa')
        cc = gfs_utils.get_coord_constraint(lv_coord.name(), 850)

        # Temp in K
        temp_850 = temp.extract(cc)

        # Temp in Celsius for use in Teten's formula
        temp_850_C = temp.extract(cc)
        temp_850_C.convert_units('Celsius')

        # Geopotential height (at surface?)
        sh = self.chart.get_data(self.GFS_VARS['hgt'])

        # Using MD calculation from Alex's code

        # SVP calculated using Teten's formulae as specified on AMS
        # Glossary of Meteorology (ref Tetens, 0. 1930 Uber einige
        # meteorologische Begriffe. Z. Geophys.. 6. 297–309.)
        # https://glossary.ametsoc.org/wiki/Tetens%27s_formula

        # NB Correction to Alex's version: use temperature in Celsius
        # in denominator of SVP term

        svp = (6.11*10.0**((7.5*temp_850_C.data)/(237.3+temp_850_C.data)))*100

        # SVD = SVP / (R * T) # R is gas constant for 1kg water
        # vapour, T is temperature in Kelvin
        svd = svp/(461.5*temp_850.data)

        # MD = Z + PW/SVD  # Z is geopotential height (at surface?)
        md = sh.data + pw.data/svd

        # Set up colour map
        self.options['cmap'] = self.get_masked_colormap(val_max=np.amax(md),
                                                        alpha=self.cm_alpha,
                                                        cm_range=self.cm_range)

        # Plot moisture depth
        ax.contourf(self.lon, self.lat, md.data, **self.options)

        ctr = ax.contour(self.lon, self.lat, md.data,
                         **self.options)

        label_levels = ctr.levels[ctr.levels > self.cm_thres[0]]
        ax.clabel(ctr, fmt='%1.0f', colors='#02682b', levels=label_levels)
