import geohash2

def geobin(lat, lon, precision=7):  # ~153m cells at p=7
    return geohash2.encode(lat, lon, precision=precision)
