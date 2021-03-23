import numpy as np
import iris.util
import iris.analysis
import iris.analysis.calculus
import skimage.measure
import shapely.geometry as sgeom
from wrf import smooth2d as wrf_smooth2d

import gfs_utils
from .component import SynopticComponent

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
        self.tol = 0.2
        self.dewpoint_level = [ 15.0 ]

        # Minimum number of vertices to include when plotting ITD contours
        self.min_vertices = 10

        # Smoothing
        self.apply_smooth = True
        self.smooth_pass = 4

        # Formatting options
        self.options = {
        }

        # Plotting options
        self.lw = 2.0
        self.col = 'black'
        self.col2 = 'white'

    def plot(self, ax):

        if self.apply_smooth:
            # Apply smoothing
            self.data.data = wrf_smooth2d(self.data.data, self.smooth_pass)

        # Calculate gradient of dewpoint temp with latitude
        gradient_dpt_wrt_lat = iris.analysis.calculus.differentiate(self.data, 'latitude')

        # Regrid dewpoint temperature to match gradient data
        dpt_2m_regridded = self.data.regrid(gradient_dpt_wrt_lat,
                                            iris.analysis.Linear(extrapolation_mode='extrapolate'))

        # Mask dpt_2m_regridded if gradient exceeds tolerance
        dpt_2m_masked = iris.util.mask_cube(dpt_2m_regridded,
                                            gradient_dpt_wrt_lat.data > self.tol)

        # Adjust self.lat to account for regridding
        self.lat = dpt_2m_masked.coord('latitude').points

        # Create an additional mask based on geopotential height
        # difference
        geo = self.chart.get_data(SynopticComponent.GFS_VARS['geo'])
        levels = [ 700, 925 ]
        lv_coord = gfs_utils.get_level_coord(geo, 'hPa')
        cc = gfs_utils.get_coord_constraint(lv_coord.name(), levels)
        geo_data = geo.extract(cc)

        # Calculate 700-925hPa geopotential height difference
        geo_diff = geo_data[0, :, :] -  geo_data[1, :, :]

        # Get threshold value for locating area of highest
        # geopotential height difference
        thresh = np.percentile(geo_diff.data, 90)

        # Find maximum latitude of area of geopotential height
        # difference above threshold
        max_lat = []
        ctr_list = skimage.measure.find_contours(geo_diff.data, level=thresh)
        for c in ctr_list:
            shp = sgeom.Polygon(c)
            lat_index = [np.rint(x).astype(int) for x in shp.bounds[::2]]
            lat = [self.lat[x] for x in lat_index]
            max_lat.append(max(lat))

        max_lat = max(max_lat)

        # Mask dewpoint temperature north of area of highest
        # geopotential height difference
        _, lat_grid = np.meshgrid(self.lon, self.lat)

        dpt_2m_masked = iris.util.mask_cube(dpt_2m_masked, lat_grid > max_lat)
        dp_masked = dpt_2m_masked.data

        # Plot masked dewpoint temperature contour (ITD)
        itd1 = ax.contour(self.lon, self.lat, dp_masked,
                         levels = self.dewpoint_level,
                          colors = self.col, linewidths = 3*self.lw)
        itd2 = ax.contour(self.lon, self.lat, dp_masked,
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
