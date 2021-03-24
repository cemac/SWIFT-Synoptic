import numpy as np
import skimage.measure
import shapely.geometry as sgeom
import matplotlib.patches as mpatches

import gfs_utils
from .component import SynopticComponent

#-----------------------------------------

# Wind components

#-----------------------------------------

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

        # Set streamline density based on longitudinal and latitudinal
        # extent
        domain = np.array(self.chart.domain)
        lon_lat = domain.reshape((2,2), order='F')[::-1]
        delta = np.diff(lon_lat).flatten()

        self.strm_options = {
            'density': tuple(delta*0.04),
            'linewidth': 0.4,
            'arrowsize': 0.9,
            'arrowstyle': '->',
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

        self.cm_name = 'Oranges'
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

class AfricanEasterlyJet(WindComponent):
    """
    African Easterly Jet

    """

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

        self.lw = 2.0
        self.color = 'green'
        self.arrow_size = 3.0
        self.arrow_interval = 3.2

        self.strm_options = {
            'color': self.color,
            'linewidth': self.lw,
            'arrowstyle': '-',
        }

        self.arrow_options = {
            'color': self.color,
            'linewidth': self.lw,
            'arrowstyle': '->',
            # set mutation_scale as in matplotlib.streamplot():
            'mutation_scale': 10 * self.arrow_size,
        }

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
                    while dist_sum > self.arrow_interval:
                        dist_sum -= self.arrow_interval
                        # Find start and end points for arrow patch
                        v = np.array([dx, dy])
                        loc = (seg_len - dist_sum)/seg_len
                        start = s + loc*v
                        end = start + 0.1*v/seg_len
                        # Draw patch
                        p = mpatches.FancyArrowPatch(
                            start, end, transform=ax.transData,
                            **self.arrow_options
                        )
                        ax.add_patch(p)
                    current_point = seg[i+1]
