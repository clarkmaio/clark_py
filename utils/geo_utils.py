import pandas as pd
import numpy as np
import urllib
import json


def retrieve_elevation(points):
    '''
    Use Open elevation to retrieve elevation from lat lon.
    points: LIST of tuples with lat lon

    Return pandas dataframe with latitude, longitude, elevation columns.
    '''
    print('Retrieving elevation for {} points'.format(len(points)))

    # Prepare api string
    url = 'https://api.open-elevation.com/api/v1/lookup?locations='
    point_str = '|'.join([str(x) + ',' + str(y) for (x,y) in points])

    request = urllib.request.urlopen(url + point_str)
    results = json.load(request).get('results')

    elevation_df = pd.DataFrame.from_records(results)
    return elevation_df
