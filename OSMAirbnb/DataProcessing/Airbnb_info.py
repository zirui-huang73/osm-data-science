# we further process listings data
# url of airbnb added
# number of nearby amenities (<300m) added

import pandas as pd
import numpy as np

listings_detail = pd.read_csv('./raw/raw_listings_detail.csv')
van_osm_data = pd.read_json('./processed/van_osm_data_refined.json')
listings_data = pd.read_json('./processed/listings_edited.json')


def num_category(place, category):
    refined_data = van_osm_data[van_osm_data['category'] == category]
    refined_data.reset_index(drop=True, inplace=True)
    lat1 = place.lat
    lon1 = place.lon
    lat2 = refined_data.lat
    lon2 = refined_data.lon
    r = 6371
    dLat = np.deg2rad(lat2-lat1)
    dLon = np.deg2rad(lon2-lon1)
    a = np.sin(dLat/2)**2+np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*(np.sin(dLon/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = c*r*1000
    return len(list(d[d<300]))


def number_of_sustenance(location):
    return num_category(location, 'sustenance')


def number_of_leisure(location):
    return num_category(location, 'leisure')


def number_of_transport(location):
    return num_category(location, 'transportation')


def number_of_arts(location):
    return num_category(location, 'arts')


listings_detail = listings_detail[listings_detail['id'].isin(listings_data['id'].values)]
listings_data['url'] = listings_detail['listing_url'].values
listings_data['sustenance'] = listings_data.apply(number_of_sustenance, axis=1)
listings_data['leisure'] = listings_data.apply(number_of_leisure, axis=1)
listings_data['transport'] = listings_data.apply(number_of_transport, axis=1)
listings_data['arts'] = listings_data.apply(number_of_arts, axis=1)

listings_data.to_json('./processed/airbnb_info.json')