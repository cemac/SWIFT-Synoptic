from datetime import datetime as dt
from calendar import monthrange

from iris import Constraint, cube
from iris.time import PartialDateTime

def get_domain_constraint(domain):
    """
    Create an iris.Constraint for a specified domain

    Create an iris.Constraint to select a domain specified by
    minimum and maximum latitude and longitude.

    Domain specification is passed in as a list, tuple or dict
    containing lat_min, lon_min, lat_max, lon_max (specifed in that
    order for a list or tuple).
    """
    if type(domain) is tuple or type(domain) is list:
        lat_min, lon_min, lat_max, lon_max = domain
    elif type(domain) is dict:
        keys = ['lat_min', 'lon_min', 'lat_max', 'lon_max']
        lat_min, lon_min, lat_max, lon_max = [domain.get(key) for key in keys]
    else:
        raise TypeError("Unhandled domain type:", type(domain))
    return Constraint(latitude = lambda cell: lat_min <= cell <= lat_max,
                      longitude = lambda cell: lon_min <= cell <= lon_max)

def get_time_constraint(fct_date, fct_time):
    """
    Create an iris.Constraint for a specified forecast date and time

    Create an iris.Constraint to select a time specified by initial forecast
    date and forecast time in hours.

    Arguments:
    fct_date : Initial date of forecast, as datetime.datetime or string
               formatted as:
                   YYYYMMDDThhmmZ
               For example:
                   20201202T0000Z
    fct_time : Required forecast time in hours.
    """

    if isinstance(fct_date, dt):
        pass
    elif type(fct_date) is str:
        fct_date = dt.strptime(fct_date, '%Y%m%dT%H%MZ')
    else:
        raise TypeError("Unhandled date type:", type(fct_date))

    # Check for day overflow
    days, hours = divmod(fct_date.hour+fct_time, 24)

    # Check for month overflow
    months = 0
    days = fct_date.day+days
    days_in_month = monthrange(fct_date.year, fct_date.month)[-1]

    if days > days_in_month:
        months, days = divmod(days, days_in_month)

    # Constraint for specified forecast time
    return Constraint(time=PartialDateTime(month=fct_date.month+months,
                                           day=days,
                                           hour=hours))

def get_coord_constraint(name, value):
    """
    Create an iris.Constraint for coordinate key/value pair
    """
    return Constraint(coord_values = {
        name: value
    })

def get_level_coord(gfs_cube):
    """
    Get level coordinate for GFS cube

    Identify level coordinate dimension in GFS cube, these are
    coordinate dimensions whose var_name starts with "lv".

    Returns (some information about the level coordinate) or None if
    none found.
    """
    if type(gfs_cube) is not cube.Cube:
        raise TypeError("Expecting Cube object")
    level_coord = next((c for c in gfs_cube.dim_coords
                        if c.var_name is not None and c.var_name.startswith("lv")), None)
    return level_coord

def main():
    pass

if __name__ == '__main__':
    main()
