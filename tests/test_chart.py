import unittest

from .context import synoptic
from synoptic.chart import SynopticChart

class TestSynopticChart(unittest.TestCase):

    def setUp(self):
        self.domains = {
            'name': 'WA',
            'unknown_name': 'XX',
            'dict': { 'lat_min': -14, 'lat_max': 10, 'lon_min': 0, 'lon_max': 23 },
            'malformed_dict': { 'min_lat': -14, 'max_lat': 10, 'min_lon': 0, 'max_lon': 23 },
            'list': [-14, 10, 0, 23 ],
            'incomplete_list': [-14, 10, 0 ],
        }
        self.timestamp = {
            'valid': '2018062400',
            'invalid': '20180624',
        }
        self.hour = {
            'valid': [0, 3, 27],
            'invalid': [5, 100],
        }

    def testUnrecognisedDomainName(self):
        with self.assertRaises(ValueError):
            SynopticChart(self.domains['unknown_name'],
                          self.timestamp['valid'],
                          self.hour['valid'][0])

    def testDomainMalformedDict(self):
        with self.assertRaises(ValueError):
            SynopticChart(self.domains['malformed_dict'],
                          self.timestamp['valid'],
                          self.hour['valid'][0])

    def testDomainIncompleteList(self):
        with self.assertRaises(ValueError):
            SynopticChart(self.domains['incomplete_list'],
                          self.timestamp['valid'],
                          self.hour['valid'][0])

    def testInvalidTimestamp(self):
        with self.assertRaises(ValueError):
            SynopticChart(self.domains['dict'],
                          self.timestamp['invalid'],
                          self.hour['valid'][0])


if __name__ == '__main__':
    unittest.main()
