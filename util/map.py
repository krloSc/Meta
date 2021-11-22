from math import cos, sin, atan2, sqrt, radians, asin
import numpy as np

def map(value,xmin,xmax,ymin,ymax):
    """Re-maps a number from one range to another"""

    result = (value-xmin)*((ymax-ymin)/(xmax-xmin))+ymin
    return result

def get_distance(lat1,lon1,lat2,lon2) -> float:
    """Return the distance between two coordinates"""

    R = 6378 # Radius of the earth in km
    delta_lat = np.radians(lat2-lat1)
    delta_lon = np.radians(lon2-lon1)
    a = (np.sin(delta_lat/2)**2 +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
        np.sin(delta_lon/2)**2)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

if __name__ == "__main__":
    value = np.array([[1,1],[2,3]])
    xmin = np.array([0,0])
    xmax = np.array([10,10])
    ymax = np.array([100,1000])
    print(map(value,xmin,xmax,xmin,ymax))
