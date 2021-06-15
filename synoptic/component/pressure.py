import datetime as dt
import warnings

import numpy as np

from .component import SynopticComponent


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
        self.label_contours = True
        self.label_size = 10
        self.lw = 0.8

        # Formatting options
        self.options = {
            'alpha': 0.6,
            'colors': 'purple',
        }

    def plot(self, ax):

        # Data to plot
        mslp = self.data.data

        try:
            levels = self.levels
        except AttributeError:
            pmin = np.amin(mslp)
            try:
                pmin = max(pmin, self.pmin)
            except AttributeError:
                pass

            pmax = np.amax(mslp)
            try:
                pmax = min(pmax, self.pmax)
            except AttributeError:
                pass

            levels = np.arange(pmin - pmin % self.step,
                               pmax + self.step - pmax % self.step,
                               self.step)

        lw = np.array([2.0*self.lw if x == self.highlight else
                       1.0*self.lw for x in levels])

        # Plot mean sea level pressure contours
        ctr = ax.contour(self.lon, self.lat, mslp,
                         levels=levels,
                         linewidths=lw,
                         **self.options)

        if self.label_contours:
            ax.clabel(ctr, fmt='%1.0f', fontsize=self.label_size)


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

        # Plot lines at 0.5 hPa intervals, skip contours in [-0.5, 0.5]
        # interval
        self.step = 0.5
        self.thres = 0.5

        # Formatting options
        self.options = {
            'colors': '#d49048',
        }

    def plot(self, ax):

        try:
            td = dt.timedelta(hours=self.delta)
            mslp_m24 = self.chart.get_data(self.gfs_vars, units=self.units,
                                           delta=td)
        except ValueError:
            msg = """{:d} hour data not available, cannot plot mean sea level
            pressure change""".format(self.delta)
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
        ax.clabel(ctr, fmt='%1.1f')
