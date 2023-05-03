from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u

import pandas as pd

data = pd.read_csv('data.csv')

for i in range(data.shape[0]):
    location = EarthLocation(lon=data['Longitude'][i] * u.deg, lat=data['Latitude'][i] * u.deg)
    time = Time(pd.to_datetime(data['Date'][i] + " " + data['Time'][i]) - pd.Timedelta(hours=data['UTC'][i]))
    
    sun = get_sun(time)
    sun_location = sun.transform_to(AltAz(obstime=time, location=location))
    
    data.at[i, 'Solar elevation'] = sun_location.alt.degree
    data.at[i, 'Solar declination'] = sun.dec.degree
    data.at[i, 'Solar azimuth'] = sun_location.az.degree
    
data.to_csv('data.csv', index=False)