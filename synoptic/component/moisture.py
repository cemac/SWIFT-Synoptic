import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import iris.analysis
from wrf import smooth2d as wrf_smooth2d

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

        self.marker_thres = 0.8

        # Formatting options
        self.options = {
            'linewidths': [ self.lw ],
            'linestyles': [ 'solid' ],
            'colors': '#198dd5'
        }

    def plot(self, ax):

        if self.data.coord(self.level_coord).points.size > 1:
            # take minimum over specified pressure levels
            rh_min = self.data.collapsed(self.level_coord, iris.analysis.MIN)
            rh = rh_min.data
        else:
            rh = self.data.data

        # Apply smoothing
        rh = wrf_smooth2d(rh, 4)

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
                    seg_len = np.hypot(dx, dy)
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


