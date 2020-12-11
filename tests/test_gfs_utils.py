import unittest

import iris
from datetime import datetime as dt

from .context import synoptic
from synoptic import gfs_utils

class TestGFSUtils(unittest.TestCase):

    def setUp(self):
        self.test_cube = iris.load_cube(iris.sample_data_path('pre-industrial.pp'))
        self.gfs_cube = iris.load_cube("tests/test_data/gfs_spfh.nc")

    def test_get_level_coord(self):
        self.assertEqual(gfs_utils.get_level_coord(self.test_cube), None)
        self.assertEqual(gfs_utils.get_level_coord(self.gfs_cube).var_name, "lv_HTGL4")

        with self.assertRaises(TypeError):
            gfs_utils.get_level_coord(iris.cube.CubeList())

    def test_get_domain_constraint(self):
        """
        Test that get_domain_constraint() creates an iris.Constraint object
        that constrains latitude and longitude to the specified ranges
        """
        lat_min, lon_min, lat_max, lon_max = [-12.0, -10.0, 20.0, 30.0]
        domain_spec = [
            (lat_min, lon_min, lat_max, lon_max),
            [lat_min, lon_min, lat_max, lon_max],
            {'lat_min': lat_min, 'lat_max': lat_max,
             'lon_min': lon_min, 'lon_max': lon_max}
        ]
        for d in domain_spec:
            dc = gfs_utils.get_domain_constraint(d)
            self.assertIs(type(dc), iris.Constraint)

            test = dc.extract(self.test_cube)
            self.assertGreaterEqual(test.coord('latitude').points.min(), lat_min)
            self.assertLessEqual(test.coord('latitude').points.max(), lat_max)
            self.assertGreaterEqual(test.coord('longitude').points.min(), lon_min)
            self.assertLessEqual(test.coord('longitude').points.max(), lon_max)

        # check that get_domain_constraint fails when the input is not
        # in an expected form
        with self.assertRaises(TypeError):
            gfs_utils.get_domain_constraint(-12.0)

    def test_get_time_constraint(self):
        """
        Test that get_time_constraint() creates an iris.Constraint object
        that constrains time to the specified time.
        """
        fct_date = dt.strptime(self.gfs_cube.attributes['initial_time'], '%m/%d/%Y (%H:%M)')
        fct_time = 12

        fct_date_str = fct_date.strftime('%Y%m%dT%H%MZ')

        for date in [fct_date, fct_date_str]:
            tc = gfs_utils.get_time_constraint(date, fct_time)
            self.assertIs(type(tc), iris.Constraint)

            test = tc.extract(self.gfs_cube).coord('time')

            # Does time coord match given constraint?
            self.assertEqual(test.points.size, 1)
            self.assertEqual(test.points[0], fct_time)
            n, m = divmod(fct_date.hour+fct_time, 24)
            self.assertEqual(test.cell(0).point.year, fct_date.year)
            self.assertEqual(test.cell(0).point.month, fct_date.month)
            self.assertEqual(test.cell(0).point.day, fct_date.day+n)
            self.assertEqual(test.cell(0).point.hour, m)

        # check that get_time_constraint fails when the input is not
        # in an expected form
        with self.assertRaises(TypeError):
            gfs_utils.get_time_constraint(-12.0, 3)

        with self.assertRaises(ValueError):
            gfs_utils.get_time_constraint('20201030', 4)

if __name__ == '__main__':
    unittest.main()
