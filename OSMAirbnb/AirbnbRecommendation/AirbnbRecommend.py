import pandas as pd
import numpy as np
from GPSPhoto import gpsphoto
import sys


def photo_coordinate(photo):
    data = gpsphoto.getGPSData(photo)
    try:
        return data['Latitude'], data['Longitude']
    except KeyError:
        print('Sorry, we cannot identify coordinates from this photo')
        return None


def closed_distance(lat, lon, neigh_data):
    lat2 = neigh_data.lat
    lon2 = neigh_data.lon
    r = 6371
    dLat = np.deg2rad(lat2-lat)
    dLon = np.deg2rad(lon2-lon)
    a = np.sin(dLat/2)**2+np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lat2))*(np.sin(dLon/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = c*r*1000
    return list(d[d < 3000].index)


def nearby_neighborhood(lat, lon, neigh_data):
    ls = closed_distance(lat, lon, neigh_data)
    return neigh_data.iloc[ls]['neighborhood'].values


def distance(lat1, lon1, lat2, lon2):
    r = 6371
    dLat = np.deg2rad(lat2-lat1)
    dLon = np.deg2rad(lon2-lon1)
    a = np.sin(dLat/2)**2+np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*(np.sin(dLon/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = c*r*1000
    return d


def show_info(airbnb):
    print("name:", airbnb['name'])
    print("price:", airbnb['price'])
    rate = airbnb['avg_score']*100
    print("rate:{:.2f}".format(rate))
    print("number of restaurant nearby:", airbnb['sustenance'])
    print("number of leisure nearby:", airbnb['leisure'])
    print("number of transportation nearby:", airbnb['transport'])
    d = distance(photo_lat, photo_lon, airbnb.lat, airbnb.lon)/1000
    print("distance from you: {:.2f} km".format(d))
    print("link:", airbnb['url'])
    print("\n")


def main():
    photo = sys.argv[1]
    neigh_data = pd.read_csv('neighborhood_info.csv')
    airbnb_data = pd.read_json('airbnb_info.json')
    photo_info = photo_coordinate(photo)
    if photo_info is None:
        return
    global photo_lat, photo_lon
    photo_lat = photo_info[0]
    photo_lon = photo_info[1]
    nearby_neighbor = nearby_neighborhood(photo_lat, photo_lon, neigh_data)
    airbnb_data = airbnb_data[airbnb_data.neighborhood.isin(nearby_neighbor)]

    min_price = int(input("please enter lower limit of price:"))
    max_price = int(input("please enter upper limit of price:"))
    airbnb_data = airbnb_data[(airbnb_data.price < max_price) & (airbnb_data.price > min_price)]
    if len(airbnb_data) == 0:
        print("unfortunately, no adjacent Airbnb meet the condition")
        return
    for neighbor in nearby_neighbor:
        tempdata = airbnb_data[airbnb_data.neighborhood == neighbor]
        tempdata = tempdata.sort_values(by=['avg_score'], ascending=False).head(3)
        print("Best Reviewed Airbnb in", neighbor)
        tempdata.apply(show_info, axis=1)
        print("\n")


if __name__ == '__main__':
    main()