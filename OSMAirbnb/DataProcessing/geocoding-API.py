import pandas as pd
from urllib.request import urlopen
import json

"""
This script help us to get address information of each location in detail
input: OSM data
output: OSM data with address information added (final_OSM.json)

warning: need google geocoding API key to run this script, and it takes one half our to execute.
"""

def get_place(place):
    lat = place.lat
    lon = place.lon
    key = "<KEY>"
    url = "https://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false&key=%s" % (lat, lon, key)
    v = urlopen(url).read()
    j = json.loads(v)
    components = j['results'][0]['address_components']
    street = route = neighborhood = city = None
    for c in components:
        if "street_number" in c['types']:
            street = c['long_name']
        if "route" in c['types']:
            route = c['long_name']
        if "neighborhood" in c['types']:
            neighborhood = c['long_name']
        if "locality" in c['types']:
            city = c['long_name']
    return [street, route, neighborhood, city]


def find_street_num(location):
    return location.address[0]


def find_street(location):
    return location.address[1]


def find_neighbor(location):
    return location.address[2]


def find_city(location):
    return location.address[3]


def data_process():
    data = pd.read_json('./raw/amenities-vancouver.json.gz', lines=True)
    drop_List = ['bench', 'trash', 'training', 'waste_disposal', 'loading_dock', 'cram_school', 'ATLAS_clean_room',
                'animal_shelter', 'chiropractor', 'driving_school', 'housing co-op', 'office|financial', 'scrapyard',
                'trolley_bay', 'vacuum_cleaner', 'water_point', 'waste_transfer_station', 'disused:restaurant',
                'veterinary', 'workshop', 'telephone', 'safety', 'luggage_locker', 'letter_box', 'atm;bank',
                'construction', 'family_centre', 'luggage_locker', 'hunting_stand', 'recycling',
                'sanitary_dump_station', 'waste_basket', 'ranger_station', 'lobby', 'shelter', 'toilets', 'EVSE', 'bbq',
                'vending_machine', 'smoking_area', 'storage', 'nursery', 'atm', 'car_rep', 'doctors',
                'parking_entrance', 'parking_space', 'post_box', 'post_depot', 'storage_rental', 'money_transfer']
    data.drop(data[data.amenity.isin(drop_List)].index, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['address'] = data.apply(get_place, axis=1)

    data['street_num'] = data.apply(find_street_num, axis=1)
    data['street'] = data.apply(find_street, axis=1)
    data['neighborhood'] = data.apply(find_neighbor, axis=1)
    data['city'] = data.apply(find_city, axis=1)
    data.drop(columns=['address'], inplace=True)
    data.to_json("./processed/final_OSM.json")


def main():
    data_process()


if __name__ == '__main__':
    main()
