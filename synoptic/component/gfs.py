import numpy as np

from wrf import smooth2d as wrf_smooth2d

from .component import SynopticComponent

# -----------------------------------------

# GFS components

# -----------------------------------------


class GFSComponent(SynopticComponent):
    """
    Base class for GFS components, providing standardised
    initialisation and plotting methods for GFS data.

    """

    def __init__(self, chart, level):
        self.smooth = None
        self.plot_fill = False
        self.label_fill = False
        self.levels = None
        self.lw = 1.0
        self.ls = 'solid'
        self.color_var = 'colors'
        self.label_col = 'black'
        self.cm_thres = None
        self.cm_alpha = None
        self.cm_range = [0, 1]
        self.label_units = ''

        super().__init__(chart)
        self.level = level

        if self.plot_fill or self.color_var == 'cmap':
            self.color_var = 'cmap'
            self.cm_name = self.color_val

        # Formatting options
        self.options = {
            'linewidths': [self.lw],
            'linestyles': [self.ls],
        }
        try:
            self.options[self.color_var] = self.color_val
        except AttributeError:
            pass

    def plot(self, ax):

        z = self.data.data

        if self.smooth:
            # Apply smoothing
            z = wrf_smooth2d(z, 4)

        try:
            max_z = np.amax(z)
            self.options['cmap'] = self.get_masked_colormap(val_max=max_z,
                                                            cm_range=self.cm_range,
                                                            alpha=self.cm_alpha)
        except AttributeError:
            pass

        if self.plot_fill:
            ax.contourf(self.lon, self.lat, z,
                        **self.options)

            if self.label_fill:
                # Draw contours and label levels
                ctr = ax.contour(self.lon, self.lat, z,
                                 **self.options)
                lvl = ctr.levels[ctr.levels > self.cm_thres[0]]
                ax.clabel(ctr, fmt='%1.0f', colors=self.label_col, levels=lvl)

        else:
            if self.levels is not None:
                ctr = ax.contour(self.lon, self.lat, z,
                                 levels=self.levels,
                                 **self.options)
            else:
                ctr = ax.contour(self.lon, self.lat, z,
                                 **self.options)

            ax.clabel(ctr, fmt='%1.0f'+self.label_units)


class CAPE(GFSComponent):
    """
    CAPE - convective available potential energy

    """

    def __init__(self, chart):
        super().__init__(chart, level=None)

    def init(self):
        self.name = "CAPE"
        self.gfs_vars = SynopticComponent.GFS_VARS['cape']
        self.units = 'J kg-1'
        self.level_units = ''

        self.levels = [1200, 1600, 2000]

        self.plot_fill = False
        self.thres_min = 900
        self.thres_max = 2000

        # Formatting options
        self.lw = 2.0

        self.color_var = 'cmap'
        self.color_val = 'BrBG'

        #self.cm_thres = [self.thres_min, self.thres_max]
        #self.cm_range = [0.57, 0.8]
        self.cm_range = [0.85, 1.0]
        #self.cm_alpha = 0.7


class CIN(GFSComponent):
    """
    CIN - convective inhibition

    """

    def __init__(self, chart):
        super().__init__(chart, level=None)

    def init(self):
        self.name = "CIN"
        self.gfs_vars = SynopticComponent.GFS_VARS['cin']
        self.units = 'J kg-1'
        self.level_units = ''

        self.smooth = True

        self.levels = [-250.0, -100.0, -50.0]

        # Formatting options
        self.lw = 2.0

        self.color_var = 'cmap'
        self.color_val = 'RdPu'
        self.cm_range = [0.9, 0.5]


class PWAT(GFSComponent):
    """
    Precipitable water

    """

    def __init__(self, chart):
        super().__init__(chart, level=None)

    def init(self):
        self.name = "Precipitable water"
        self.gfs_vars = SynopticComponent.GFS_VARS['pwat']
        self.units = 'kg m-2'
        self.level_units = ''

        self.plot_fill = True

        # Formatting options
        self.color_val = 'Purples'

        self.cm_thres = [None, None]


class DPT(GFSComponent):
    """
    Dewpoint temperature 2m
    """

    def __init__(self, chart):
        super().__init__(chart, level=None)

    def init(self):
        self.name = "Dewpoint temperature"
        self.gfs_vars = SynopticComponent.GFS_VARS['dpt_2m']
        self.units = 'Celsius'
        self.level_units = ''

        self.plot_fill = False

        self.smooth = True

        self.levels = [15.0]
        self.label_units = '\N{DEGREE SIGN}C'

        # Formatting options
        self.color_val = 'firebrick'
        self.lw = 2.0
