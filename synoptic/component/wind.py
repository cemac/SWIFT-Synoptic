import math

import numpy as np
import skimage.measure
import shapely.geometry as sgeom
import matplotlib.patches as mpatches
from matplotlib.path import Path

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

        self.plot_ws = True
        self.plot_strm = False

        self.ws_level = [ 15.0 ]

        self.options = {
            'alpha': 0.6,
            'linewidths': 1.6,
            'colors': 'darkgreen',
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

        # Get U/V wind components
        U, V, windspeed = self.get_wind_components()

        if self.plot_ws:
            # Plot 10m windspeed 15 m/s contour
            ctr = ax.contour(self.lon, self.lat, windspeed,
                             levels = self.ws_level,
                             **self.options)

        if self.plot_strm:
            # Plot streamlines
            strm = ax.streamplot(self.lon, self.lat, U, V,
                                 **self.strm_options)


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

class TropicalEasterlyJet(WindComponent):
    """
    Tropical Easterly Jet

    Two cores above 35 kt, at 100 and 200 hPa, around 15°N and 8°N respectively.

    """

    def __init__(self, chart, level=200):
        super().__init__(chart, level)

    def init(self):
        self.name = "Tropical Easterly Jet"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x) for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ws = False
        self.plot_core = True
        self.plot_jet = True

        self.thres_axis = 18 # FCHB: 35 kt = 18 m/s
        self.thres_core = 25 # FCHB: 40-50 kt = 20-25 m/s
        self.thres_core_percentile = 95 # Eniola @ NiMET says no set
                                        # threshold for core

        # Formatting options
        self.lw = 2.0
        # TODO select preferred colour
        # self.color = '#b34c2e' # Colour value derived from FCHB - Table 11.1
        # self.color = '#c3704b' # Colour value derived from FCHB - section 11.7
        # self.color = '#b99356' # Colour value derived from FCHB - Figure 11.1
        self.color = '#be933a' # Colour value derived from FCHB - Figure 11.2

        self.cm_name = 'Oranges'
        self.cm_thres = [ self.thres_axis, None ]

        self.ws_options = {
            'cmap': self.cm_name,
        }

        self.core_options = {
            'edgecolor': self.color,
            'fill': None,
            'linewidth': self.lw/2,
        }

        self.core_label_fontsize = 8.0
        self.core_label_units = 'hPa'
        self.core_label_options = {
            'xytext': np.array([-2, -1.25])*self.core_label_fontsize,
            'textcoords': 'offset points',
            'fontsize': self.core_label_fontsize,
        }

        self.arrow_size = 2.6
        self.arrow_interval = 2.0
        self.arrow_options = {
            'color': self.color,
            'linewidth': self.lw,
            'arrowstyle': '->,head_length=0.5,head_width=0.3',
            # set mutation_scale as in matplotlib.streamplot():
            'mutation_scale': 10 * self.arrow_size,
        }

        self.path_options = {
            'color': self.color,
            'linewidth': self.lw,
            'fill': False,
        }

    def plot(self, ax):

        U, V, windspeed = self.get_wind_components()

        angle = 0

        # Set mask to analyse windspeed in relevant region i.e. around
        # 15N for core at 100hPa, around 8N for core at 200hPa
        _, lat_grid = np.meshgrid(self.lon, self.lat)
        lat_min, lat_max = (12, 18) if self.level == 100 else (5, 11)
        lat_mask = (lat_grid < lat_min) | (lat_grid > lat_max)

        # Mask windspeed below specified threshold
        ws_mask = (windspeed < self.thres_axis)

        if self.plot_ws:
            # Plot windspeed contours
            max_ws = np.amax(windspeed)
            self.ws_options['cmap'] = self.get_masked_colormap(val_max=max_ws, alpha=0.4)
            ctr = ax.contourf(self.lon, self.lat, windspeed,
                              **self.ws_options)

        if self.plot_jet:
            # Define mask for jet axis
            mask = lat_mask | ws_mask

            # Mask wind components using NaNs because streamplot
            # doesn't seem to respect masked arrays
            U[mask] = np.nan
            V[mask] = np.nan

            # Mask windspeed
            ws_masked = np.ma.masked_where(mask, windspeed)

            # Select seed point(s) corresponding to maximum windspeed
            seed_index = np.unravel_index(np.argmax(ws_masked, axis=None), ws_masked.shape)
            seed_index = np.array(seed_index, ndmin=2)
            seed_points = np.array([[self.lon[x[1]], self.lat[x[0]]] for x in seed_index])

            # Get streamline through specified seed point
            strm = ax.streamplot(self.lon, self.lat, U, V,
                                 start_points=seed_points,
                                 arrowstyle='-',
                                 linewidth=0)

            segments = strm.lines.get_segments()
            num_seg = len(segments)
            if num_seg == 0:
                return

            # Get line segments, place arrows and get vertices for tramlines
            # TODO add jet entrance/exit markers
            verts = []
            current_point = None
            for j, seg in enumerate(segments):
                if j == num_seg//2:
                    # Get angle of central line segment
                    tdx, tdy = seg[-1] - seg[0]
                    angle = math.tan(tdy/tdx) * 180/math.pi
                    print("angle across seg:", angle)
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

                    n = np.array([-dy, dx])
                    n = 0.25*n/np.linalg.norm(n)
                    verts.append([s+n, s-n])
                    current_point = seg[i+1]
                verts.append([current_point+n, current_point-n])

            # Rearrange array to get vertices for parallel tramlines
            verts = np.stack(verts, axis=1)
            for v in verts:
                path = Path(v)
                patch = mpatches.PathPatch(path, **self.path_options)
                ax.add_patch(patch)

        if self.plot_core:

            # Mask windspeed to relevant region
            ws_masked = np.ma.masked_where(lat_mask, windspeed)

            # Use maximum of specified core threshold and specified percentile
            ws_thres = np.percentile(ws_masked[~ws_masked.mask], self.thres_core_percentile)
            ws_thres = max(self.thres_core, ws_thres)
            print("Using threshold for core:", ws_thres)

            # Get contours for core threshold windspeed
            ctr_list = skimage.measure.find_contours(ws_masked, level=ws_thres)
            for c in ctr_list:
                shp = sgeom.Polygon(c)

                # Does this contain a point of maximum windspeed?
                # FIXME what if not self.plot_jet (i.e. seed_index not defined)?
                contains_seed = False
                for x in seed_index:
                    if shp.intersects(sgeom.Point((x))):
                        contains_seed = True
                if not contains_seed:
                    continue

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
                                     angle=angle,
                                     **self.core_options)
                ax.add_artist(p)

                # Add label to indicate core pressure level
                loc = xy + np.array([0, dy/2])
                ax.annotate(f'{self.level} {self.level_units}',
                            xy=loc,
                            **self.core_label_options)

