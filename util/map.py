from math import cos, sin, atan2, sqrt, radians, asin
import numpy as np

def map(value,xmin,xmax,ymin,ymax):
    result = (value-xmin)*((ymax-ymin)/(xmax-xmin))+ymin
    return result

def get_distance(lat1,lon1,lat2,lon2) -> float:

  R = 6378 # Radius of the earth in km
  delta_lat = np.radians(lat2-lat1)
  delta_lon = np.radians(lon2-lon1)
  a = (np.sin(delta_lat/2)**2 +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
        np.sin(delta_lon/2)**2)

  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
  #c = 2*asin(min(1,sqrt(a)))
  d = R * c
  return d

if __name__ == "__main__":
    print(get_distance( 10.1563889, -67.995, 10.145, -68.03472222222221))
