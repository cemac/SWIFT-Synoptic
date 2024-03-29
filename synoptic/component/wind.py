import math
import warnings

import numpy as np
import iris.util
import skimage.measure
import shapely.geometry as sgeom
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from matplotlib.path import Path
from windspharm.iris import VectorWind
from wrf import smooth2d as wrf_smooth2d
from scipy.ndimage import label

import gfs_utils
from .component import SynopticComponent

# -----------------------------------------

# Wind components

# -----------------------------------------


class WindComponent(SynopticComponent):
    """
    Base class for wind-related synoptic components, providing utility
    method to extract U/V wind components and windspeed for a given level
    in the data.

    """

    def __init__(self, chart, level):
        super().__init__(chart, level)
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
        try:
            uc = gfs_utils.get_coord_constraint(u_lv_coord.name(), level)
            u = u.extract(uc)
        except AttributeError:
            pass

        try:
            vc = gfs_utils.get_coord_constraint(v_lv_coord.name(), level)
            v = v.extract(vc)
        except AttributeError:
            pass

        def handle_error(lvl, units):
            err_msg = f'''Could not extract wind component data for
            specified level: {lvl} {units}'''
            warnings.warn(err_msg)

        if type(level) is list:
            U = []
            V = []
            windspeed = []
            for lvl in level:
                try:
                    ucc = gfs_utils.get_coord_constraint(u_lv_coord.name(), lvl)
                    vcc = gfs_utils.get_coord_constraint(v_lv_coord.name(), lvl)
                    ui = u.extract(ucc)
                    vi = v.extract(vcc)
                except AttributeError:
                    ui = u
                    vi = v

                try:
                    Ui = ui.data.astype(np.float64)
                    Vi = vi.data.astype(np.float64)
                    U.append(Ui)
                    V.append(Vi)
                    windspeed.append(np.sqrt(Ui**2 + Vi**2))
                except AttributeError:
                    handle_error(lvl, self.level_units)
                    raise
        else:
            try:
                U = u.data.astype(np.float64)
                V = v.data.astype(np.float64)
                windspeed = np.sqrt(U**2 + V**2)
            except AttributeError:
                handle_error(level, self.level_units)
                raise

        return (U, V, windspeed)

    def plot_jet_axis(self, ax, mask=None, plot_core=True):

        U, V, windspeed = self.get_wind_components()

        lat_min, lon_min, lat_max, lon_max = self.chart.domain
        box = sgeom.box(lon_min, lat_min, lon_max, lat_max)

        if mask is not None:
            # Mask wind components using NaNs because streamplot
            # doesn't seem to respect masked arrays
            U[mask] = np.nan
            V[mask] = np.nan

            # Mask windspeed
            ws = np.ma.masked_where(mask, windspeed)
        else:
            ws = windspeed

        # Select seed point(s) corresponding to maximum windspeed
        seed_index = np.unravel_index(np.argmax(ws, axis=None), ws.shape)
        seed_index = np.array(seed_index, ndmin=2)
        seed_points = np.array([[self.lon[x[1]], self.lat[x[0]]]
                                for x in seed_index])

        # Get streamline(s) through specified seed point(s)
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
        angle = 0
        for j, seg in enumerate(segments):
            if np.allclose(seg[0], seed_points[0]):
                # Found matching seed point
                if not all(seg[0] == seg[-1]):
                    # Get angle of segment
                    tdx, tdy = seg[-1] - seg[0]
                    angle = math.atan(tdy/tdx) * 180/math.pi
            for i, s in enumerate(seg[:-1]):
                if not all(s == current_point):
                    # Reset distance sum
                    dist_sum = 0
                # Check length of segment and add patches at
                # suitable intervals
                if np.allclose(s, seg[i+1]):
                    continue
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
                sp = s+n
                sm = s-n
                if not box.contains(sgeom.Point((sp))):
                    line = sgeom.LineString([sp, sp + np.array([dx, dy])])
                    sp = np.array(box.exterior.intersection(line))
                if not box.contains(sgeom.Point((sm))):
                    line = sgeom.LineString([sm, sm + np.array([dx, dy])])
                    sm = np.array(box.exterior.intersection(line))
                if len(sp) and len(sm):
                    verts.append([sp, sm])
                current_point = seg[i+1]
            if current_point is not None:
                sp = current_point+n
                sm = current_point-n
                if (box.contains(sgeom.Point((sp))) and
                    box.contains(sgeom.Point((sm)))):
                    verts.append([sp, sm])

        # Rearrange array to get vertices for parallel tramlines
        verts = np.stack(verts, axis=1)
        for v in verts:
            path = Path(v)
            patch = mpatches.PathPatch(path, **self.path_options)
            ax.add_patch(patch)

        if plot_core:

            # Mask windspeed to relevant region
            ws_masked = np.ma.masked_where(self.coord_mask, windspeed)

            # Use maximum of specified core threshold and specified percentile
            ws_thres = np.percentile(ws_masked[~ws_masked.mask],
                                     self.thres_core_percentile)
            ws_thres = max(self.thres_core, ws_thres)

            # Get contours for core threshold windspeed
            ctr_list = skimage.measure.find_contours(ws_masked, level=ws_thres)
            for c in ctr_list:
                shp = sgeom.Polygon(c)

                # Does this contain a point of maximum windspeed?
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
                                    self.lat[np.rint(x).astype(int)]]
                                   for x in it])
                dx, dy = np.abs(bounds[1] - bounds[0])

                # Draw core as ellipse around centroid
                p = mpatches.Ellipse(xy=xy,
                                     width=dx,
                                     height=dy,
                                     angle=angle,
                                     **self.core_options)
                ax.add_patch(p)

                # Add label to indicate core pressure level
                loc = xy + np.array([0, dy/2])
                ax.annotate(f'{self.level} {self.level_units}',
                            xy=loc,
                            **self.core_label_options)


class WindPressureLevel(WindComponent):
    """
    Wind at specified pressure level

    """

    def __init__(self, chart, level):
        super().__init__(chart, level)

    def init(self):
        self.name = "Wind on Pressure Level"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x)
                         for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ws = True
        self.plot_strm = True

        self.density = 0.7 if self.chart.domain_name == 'EA' else 2.4

        self.min_ws = 0
        self.max_ws = 50
        self.thres_ws = 10

        self.cm_name = 'Blues'
        self.cm_thres = [self.thres_ws, None]

        # Formatting options
        self.ws_options = {
            'cmap': self.cm_name,
        }

        self.strm_options = {
            'linewidth': 0.8,
            'arrowsize': 1.2,
            'arrowstyle': '->',
        }

    def plot(self, ax):

        U, V, windspeed = self.get_wind_components()

        if self.plot_ws:
            # Plot windspeed at 925 hPa
            max_ws = np.amax(windspeed)
            self.cm_thres = [self.thres_ws, None]
            self.ws_options['cmap'] = self.get_masked_colormap(val_max=max_ws,
                                                               alpha=0.4)
            ax.contourf(self.lon, self.lat, windspeed,
                        **self.ws_options)

        if self.plot_strm:
            # Set streamline density based on longitudinal and
            # latitudinal extent
            delta = self.chart.get_domain_extent()
            self.strm_options['density'] = tuple(delta*self.density/delta[0])

            # Plot streamlines
            ax.streamplot(self.lon, self.lat, U, V,
                          **self.strm_options)


class WindHeightLevel(WindComponent):
    """
    Wind at specified height level

    """
    def __init__(self, chart, level):
        super().__init__(chart, level)

    def init(self):
        self.name = "Windspeed at Height Level"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x)
                         for x in ('u_lvh', 'v_lvh')]
        self.units = None
        self.level_units = 'm'

        self.plot_ws = True
        self.plot_strm = False

        self.density = 0.7 if self.chart.domain_name == 'EA' else 2.4

        self.ws_level = [15.0]
        self.step = 5.0
        self.lw = 1.0
        self.label_contours = True
        self.label_size = 10
        self.label_units = 'm/s'  # 'ms\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}'

        self.options = {
            'alpha': 1.0,
            'colors': 'darkgreen',
        }

        self.strm_options = {
            'linewidth': 0.8,
            'arrowsize': 1.2,
            'arrowstyle': '->',
        }

    def plot(self, ax):

        # Get U/V wind components
        U, V, windspeed = self.get_wind_components()

        if self.plot_ws:

            try:
                levels = self.ws_level
            except AttributeError:
                vmin = np.amin(windspeed)
                vmax = np.amax(windspeed)
                levels = np.arange(vmin - vmin % self.step,
                                   vmax + self.step - vmax % self.step,
                                   self.step)

            try:
                lw = np.array([2.0*self.lw if x == self.highlight else
                               1.0*self.lw for x in levels])
            except AttributeError:
                lw = self.lw

            # Plot 10m windspeed contour at specified level(s)
            ctr = ax.contour(self.lon, self.lat, windspeed,
                             levels=levels,
                             linewidths=lw,
                             **self.options)

            if self.label_contours:
                ax.clabel(ctr, fmt='%1.0f'+self.label_units,
                          fontsize=self.label_size)

        if self.plot_strm:

            # Set streamline density based on longitudinal and
            # latitudinal extent
            delta = self.chart.get_domain_extent()
            self.strm_options['density'] = tuple(delta*self.density/delta[0])

            # Plot streamlines
            ax.streamplot(self.lon, self.lat, U, V,
                          **self.strm_options)


class Divergence(WindComponent):
    """
    Divergence

    """

    def __init__(self, chart, level=650):
        super().__init__(chart, level)

    def init(self):
        self.name = "Divergence"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x)
                         for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        # Threshold absolute value for masking
        self.thres = 1e-05

        self.cm_name = 'PRGn'
        self.cm_thres = [None, None]

        self.options = {
            'alpha': 0.6,
            'cmap': self.cm_name,
        }

    def plot(self, ax):

        # Get global u/v wind data
        ug, vg = self.chart.get_data(self.gfs_vars, apply_domain=False)

        # Get level coordinates for U/V wind components
        ug_lv_coord = gfs_utils.get_level_coord(ug, self.level_units)
        vg_lv_coord = gfs_utils.get_level_coord(vg, self.level_units)

        # Constrain to specified level(s)
        ugc = gfs_utils.get_coord_constraint(ug_lv_coord.name(), self.level)
        ug = ug.extract(ugc)

        vgc = gfs_utils.get_coord_constraint(vg_lv_coord.name(), self.level)
        vg = vg.extract(vgc)

        # Apply smoothing
        ug_data = wrf_smooth2d(ug.data, 6)
        ug.data = ug_data.data
        vg_data = wrf_smooth2d(vg.data, 6)
        vg.data = vg_data.data

        # Set up windspharm wind vector object
        w = VectorWind(ug, vg)

        # Get divergence
        div = w.divergence()

        # Constrain to specified domain
        domain_constraint = gfs_utils.get_domain_constraint(self.chart.domain)
        div = div.extract(domain_constraint)

        # Mask absolute values below threshold
        div_masked = np.ma.masked_where(np.abs(div.data) < self.thres, div.data)

        # Set norm to match centre of colour scale to zero value
        self.options['norm'] = mcolors.TwoSlopeNorm(vmin=np.min(div_masked),
                                                    vcenter=0, vmax=np.max(div_masked))

        ax.contourf(self.lon, self.lat, div_masked, **self.options)


class WindShear(WindComponent):

    def __init__(self, chart, level1, level2):
        super().__init__(chart, [level1, level2])

    def init(self):
        self.name = "Wind shear"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x)
                         for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ctr = True
        self.plot_qv = True

        # Formatting options
        self.qv_skip = 2
        self.qv_options = {
            'color': 'black',
            'alpha': 0.2,
            # width relative to selected units (default = axis width)
            'width': 0.002,
        }

        self.ws_thres = 25

        self.cm_name = 'Wistia'
        self.cm_thres = [self.ws_thres, None]

        self.ws_options = {
            'cmap': self.cm_name,
        }

    def plot(self, ax):

        U, V, _ = self.get_wind_components()

        U_diff = U[1] - U[0]
        V_diff = V[1] - V[0]

        ws_diff = np.sqrt(U_diff**2 + V_diff**2)

        if self.plot_ctr:
            cm = self.get_masked_colormap(val_max=np.amax(ws_diff), alpha=0.4)
            self.ws_options['cmap'] = cm
            ax.contourf(self.lon, self.lat, ws_diff,
                        **self.ws_options)

        if self.plot_qv:
            # Mask values below threshold
            U_diff = np.ma.masked_where(ws_diff < self.ws_thres, U_diff)
            V_diff = np.ma.masked_where(ws_diff < self.ws_thres, V_diff)
            ax.quiver(self.lon[::self.qv_skip], self.lat[::self.qv_skip],
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
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x)
                         for x in ('u_lvp', 'v_lvp')]
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
        self.cm_thres = [self.thres_axis, None]

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
        self.core_label_units = 'ms\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}'
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
            self.ws_options['cmap'] = self.get_masked_colormap(val_max=max_ws,
                                                               alpha=0.4)
            ax.contourf(self.lon, self.lat, windspeed,
                        **self.ws_options)

        if self.plot_core:
            # Mask windspeed to relevant region i.e. between 10-15 deg N
            lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
            mask = (lat_grid < 5) | (lat_grid > 20)
            ws_masked = np.ma.masked_where(mask, windspeed)
            ws_masked[mask] = np.nan

            # Get contours for core threshold windspeed
            ctr_list = skimage.measure.find_contours(ws_masked,
                                                     level=self.thres_core)
            for c in ctr_list:
                shp = sgeom.Polygon(c)

                # Translate centroid coordinates to lon/lat
                centroid = shp.centroid.coords[0]
                xy = [self.lon[np.rint(centroid[1]).astype(int)],
                      self.lat[np.rint(centroid[0]).astype(int)]]

                # Use contour bounds to get dimensions for core
                it = iter(shp.bounds)
                bounds = np.array([[self.lon[np.rint(next(it)).astype(int)],
                                    self.lat[np.rint(x).astype(int)]]
                                   for x in it])
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
            seed_points = np.array([[self.lon[x[1]], self.lat[x[0]]]
                                    for x in seed_index])

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

    Two cores above 35 kt, at 100 and 200 hPa, around 15°N and 8°N
    respectively.

    """

    def __init__(self, chart, level=200):
        super().__init__(chart, level)

    def init(self):
        self.name = "Tropical Easterly Jet"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x)
                         for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ws = False
        self.plot_core = True
        self.plot_jet = True

        self.thres_axis = 18  # FCHB: 35 kt = 18 m/s
        self.thres_core = 25  # FCHB: 40-50 kt = 20-25 m/s
        # Eniola @ NiMET says there is no set threshold for core
        self.thres_core_percentile = 95

        # Formatting options
        self.lw = 2.0
        # TODO select preferred colour
        # self.color = '#b34c2e'  # Colour value from FCHB - Table 11.1
        # self.color = '#c3704b'  # Colour value from FCHB - section 11.7
        # self.color = '#b99356'  # Colour value from FCHB - Figure 11.1
        self.color = '#be933a'  # Colour value from FCHB - Figure 11.2

        self.cm_name = 'Oranges'
        self.cm_thres = [self.thres_axis, None]

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

        _, _, windspeed = self.get_wind_components()

        # Set mask to analyse windspeed in relevant region i.e. around
        # 15N for core at 100hPa, around 8N for core at 200hPa
        lat_min, lat_max = (12, 18) if self.level == 100 else (5, 11)
        self.set_coord_mask(lat_min=lat_min, lat_max=lat_max)

        # Mask windspeed below specified threshold
        ws_mask = (windspeed < self.thres_axis)

        if self.plot_ws:
            # Plot windspeed contours
            max_ws = np.amax(windspeed)
            self.ws_options['cmap'] = self.get_masked_colormap(val_max=max_ws,
                                                               alpha=0.4)
            ax.contourf(self.lon, self.lat, windspeed,
                        **self.ws_options)

        if self.plot_jet:
            # Define mask for jet axis
            mask = self.coord_mask | ws_mask

            self.plot_jet_axis(ax, mask, self.plot_core)


class SubtropicalJet(WindComponent):
    """
    Subtropical Jet

    Marks boundary between deep tropical troposphere and less deep
    mid-latitude troposphere.

    Indicated by windspeed above threshold on PVU isosurfaces (0.7 or
    2 PVU). Otherwise can use 200hPa winds.

    Windspeed threshold is seasonally dependent: during monsoon
    (May-Sept): 45kt, winter: 60kt.

    """

    def __init__(self, chart, level=200):
        super().__init__(chart, level)

    def init(self):
        self.name = "Sub Tropical Jet"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x)
                         for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ws = False
        self.plot_core = True
        self.plot_jet = True

        month = self.chart.date().month

        # FIXME: which months to use for winter and what about outside
        # monsoon + winter?
        self.thres_axis = 23.15  # FCHB: 45 kt = 23.15 m/s (during monsoon)
        if month >= 10 or month <= 2:
            self.thres_axis = 30.87  # FCHB: 60 kt = 30.87 m/s (during winter)
        self.thres_core = 25
        self.thres_core_percentile = 95

        # Formatting options
        self.lw = 2.0
        self.color = '#be933a'  # Colour value derived from FCHB - Figure 11.2

        self.cm_name = 'Oranges'
        self.cm_thres = [self.thres_axis, None]

        self.ws_options = {
            'cmap': self.cm_name,
        }

        self.core_options = {
            'edgecolor': self.color,
            'fill': None,
            'linewidth': self.lw/2,
        }

        self.core_label_fontsize = 8.0
        self.core_label_units = 'ms\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}'
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

        _, _, windspeed = self.get_wind_components()

        # Set mask to analyse windspeed in relevant region i.e. > 20N
        self.set_coord_mask(lat_min=20)

        # Mask windspeed below specified threshold
        ws_mask = (windspeed < self.thres_axis)

        if self.plot_ws:
            # Plot windspeed contours
            max_ws = np.amax(windspeed)
            self.ws_options['cmap'] = self.get_masked_colormap(val_max=max_ws,
                                                               alpha=0.4)
            ax.contourf(self.lon, self.lat, windspeed,
                        **self.ws_options)

        if self.plot_jet:
            # Define mask for jet axis
            mask = self.coord_mask | ws_mask

            self.plot_jet_axis(ax, mask, self.plot_core)


class AfricanEasterlyWaves(WindComponent):
    """
    African Easterly Waves

    """

    def __init__(self, chart, level=600):
        super().__init__(chart, level)

    def init(self):
        self.name = "African Easterly Waves"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x)
                         for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.mask_min_size = 12  # need to relate to grid size

        self.options = {
            'linewidths': 3.0,
            'colors': 'black',
        }

    def plot(self, ax):

        # Get global u/v wind data
        ug, vg = self.chart.get_data(self.gfs_vars, apply_domain=False)

        # Get level coordinates for U/V wind components
        ug_lv_coord = gfs_utils.get_level_coord(ug, self.level_units)
        vg_lv_coord = gfs_utils.get_level_coord(vg, self.level_units)

        # Constrain to specified level(s)
        ugc = gfs_utils.get_coord_constraint(ug_lv_coord.name(), self.level)
        ug = ug.extract(ugc)

        vgc = gfs_utils.get_coord_constraint(vg_lv_coord.name(), self.level)
        vg = vg.extract(vgc)

        # Apply smoothing
        ug_data = wrf_smooth2d(ug.data, 6)
        ug.data = ug_data.data
        vg_data = wrf_smooth2d(vg.data, 6)
        vg.data = vg_data.data

        # Set up windspharm wind vector object
        w = VectorWind(ug, vg)

        # Get nondivergent component of wind
        upsi, vpsi = w.nondivergentcomponent()
        Vpsi = VectorWind(upsi, vpsi)

        # Partition streamfunction vorticity into curvature and shear
        # components

        # Bell and Keyser 1993 Appendix equation A.3b
        # Relative curvature vorticity:
        # V dalpha/ds = 1/V^2[u^2 dv/dx - v^2 du/dy - uv(du/dx - dv/dy)]

        dupsi_dx, dupsi_dy = Vpsi.gradient(upsi)
        dvpsi_dx, dvpsi_dy = Vpsi.gradient(vpsi)
        ws = Vpsi.magnitude()

        vrt = (upsi**2 * dvpsi_dx - vpsi**2 * dupsi_dy -
               upsi*vpsi*(dupsi_dx - dvpsi_dy))/ws**2

        # Calculate advection of non-divergent curvature vorticity
        dvrt_dx, dvrt_dy = Vpsi.gradient(vrt)
        advec = -1*(upsi * dvrt_dx + vpsi * dvrt_dy)

        # Second order advection term needed for masking
        dadv_dx, dadv_dy = Vpsi.gradient(advec)
        advec2 = upsi * dadv_dx + vpsi * dadv_dy

        # Apply domain constraint
        domain_constraint = gfs_utils.get_domain_constraint(self.chart.domain)
        upsi = upsi.extract(domain_constraint)
        vrt = vrt.extract(domain_constraint)
        advec = advec.extract(domain_constraint)
        advec2 = advec2.extract(domain_constraint)

        # Masking rules from Berry & Thorncroft 2007, Table 1

        # A1. Mask streamfunction curvature vorticity to exclude AEW
        # ridges axes or weak systems (BT2007: K1T = 0.5 * 10^-5 s^-1)
        K1T = 1.5 * 10**-5  # s^-1
        m1t = vrt.data <= K1T  # mask for troughs
        m1r = vrt.data >= -K1T  # mask for ridges

        # A2. Remove "pseudoridge" axes in nondivergent flow that is
        # highly cyclonically curved
        K2T = 0  # m s^-3
        m2 = advec2.data <= K2T

        # A3. Removes trough axes in westerly flow
        K3T = 0  # m s^-1
        m3 = upsi.data >= K3T

        trough_mask = m1t | m2 | m3
        ridge_mask = m1r | ~m2 | m3

        troughs = self.apply_size_mask(trough_mask, advec.data)
        ridges = self.apply_size_mask(ridge_mask, advec.data)

        # Plot troughs and ridges
        ax.contour(self.lon, self.lat, troughs, levels=0, **self.options)
        ax.contour(self.lon, self.lat, ridges, levels=0, **self.options,
                   linestyles="dotted")

    def apply_size_mask(self, mask, data):

        # Convert mask to integer for subsequent labelling
        m = (~mask).astype(int)
        m_label, m_label_count = label(m)
        # loop through labels
        for j in np.arange(0, m_label_count + 1):
            # get the indices for this label
            m_iy, m_ix = np.where(m_label == j)
            # if less than specified number of cells in latitudinal
            # direction, mask this area
            if (m_iy.ptp() < (self.mask_min_size - 1)):
                mask[m_iy, m_ix] = True
        return np.ma.masked_where(mask, data)


class MonsoonTrough(WindComponent):
    """
    Monsoon trough

    Line of maximum vorticity at 850 hPa
    """

    def __init__(self, chart, level=850):
        super().__init__(chart, level)

    def init(self):
        self.name = "Monsoon Trough"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x)
                         for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'
        self.thres = 6*10**-5

        self.cm_name = 'Reds'
        self.cm_thres = [self.thres, None]

        self.options = {
            # 'alpha': 0.6,
            # 'linewidths': 1.6,
            # 'colors': 'darkred',
            'cmap': self.cm_name,
        }

    def plot(self, ax):

        # Get global u/v wind data
        ug, vg = self.chart.get_data(self.gfs_vars, apply_domain=False)

        # Get level coordinates for U/V wind components
        ug_lv_coord = gfs_utils.get_level_coord(ug, self.level_units)
        vg_lv_coord = gfs_utils.get_level_coord(vg, self.level_units)

        # Constrain to specified level(s)
        ugc = gfs_utils.get_coord_constraint(ug_lv_coord.name(), self.level)
        ug = ug.extract(ugc)

        vgc = gfs_utils.get_coord_constraint(vg_lv_coord.name(), self.level)
        vg = vg.extract(vgc)

        # Apply smoothing
        ug_data = wrf_smooth2d(ug.data, 6)
        ug.data = ug_data.data
        vg_data = wrf_smooth2d(vg.data, 6)
        vg.data = vg_data.data

        # Set up windspharm wind vector object
        w = VectorWind(ug, vg)
        xi = w.vorticity()
        div = w.divergence()

        # Constrain to specified domain
        domain_constraint = gfs_utils.get_domain_constraint(self.chart.domain)
        xi = xi.extract(domain_constraint)
        div = div.extract(domain_constraint)

        # Calculate strain S following Schielicke et al 2016
        u_x, u_y = w.gradient(ug)
        v_x, v_y = w.gradient(vg)

        D_h = u_x + v_y  # horizontal divergence
        Def = u_x - v_y  # stretching deformation
        Def_s = u_y + v_x  # shearing deformation

        ss = D_h**2 + Def**2 + Def_s**2
        ss = ss.extract(domain_constraint)

        S = np.sqrt(ss.data)/math.sqrt(2)

        # Okubo-Weiss parameter
        # vorticity - (div + strain)
        okw = (div + S) - xi

        # Set mask to inspect O-W parameter in relevant region i.e. < 20N
        self.set_coord_mask(lat_max=20)

        # Define mask for O-W parameter
        mask = self.coord_mask | (okw.data < self.thres)

        # Apply mask
        okw_masked = iris.util.mask_cube(okw, mask)

        ax.contourf(self.lon, self.lat, okw_masked.data, **self.options)
