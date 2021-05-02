import numpy as np
import iris
import matplotlib.cm as mcm
import matplotlib.colors as mc

import gfs_utils

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
        # geopotential height
        'geo': 'HGT_P0_L100_GLL0',
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
        self.chart = chart
        self.init()
        self.data = chart.get_data(self.gfs_vars, units=self.units)
        if level is not None:
            self.level = level
            # Constrain data to specified level(s)
            lv_coord = gfs_utils.get_level_coord(self.data, self.level_units)
            self.level_coord = lv_coord.name()
            cc = gfs_utils.get_coord_constraint(self.level_coord, self.level)
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
        the case when applying a colour map to masked data.

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
