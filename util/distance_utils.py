import math
import pyspark.sql.functions as F


def deg2rad(deg):
    return deg * (F.lit(math.pi) / 180)


def km(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dLat = deg2rad(lat2 - lat1)  # deg2rad below
    dLon = deg2rad(lon2 - lon1)
    a = F.sin(dLat / 2) * F.sin(dLat / 2) + F.cos(deg2rad(lat1)) * F.cos(deg2rad(lat2)) * F.sin(dLon / 2) * F.sin(
        dLon / 2)
    c = 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))
    d = R * c  # Distance in km
    return d
