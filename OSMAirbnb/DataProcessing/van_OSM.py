# clean and process raw OSM data
# category of amenity added
# missing neighborhood filled
# only keep those in neighborhoods within airbnb data
# neighborhood information processed

# output: van_osm_data.json neighborhood.info.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

osm_data = pd.read_json('./processed/finalOSM.json')
airbnb_data = pd.read_json('./processed/listings_edited.json')

def find_category(location):
    # assign category to each location of OSM data
    if location.amenity in (['bar', 'bbq', 'biergarten', 'cafe', 'fast_food', 'food_court', 'ice_cream', 'pub',
                             'restaurant']):
        return 'sustenance'
    elif location.amenity in (['bicycle_rental', 'bus_station', 'car_sharing', 'car_rental', 'taxi']):
        return 'transportation'
    elif location.amenity in (['internet_cafe', 'casino', 'cinema', 'community_centre', 'gambling',
                               'nightclub', 'theatre', 'stripclub', 'studio', 'park']):
        return 'leisure'
    elif location.amenity in (['arts_centre', 'community_centre', 'fountain', 'public_bookcase', 'clock',
                               'place_of_worship', 'monastery']):
        return 'arts'
    else:
        return np.nan


def vancouver_osm_data():
    van_osm_data = osm_data[osm_data.city == 'Vancouver']
    van_osm_data = van_osm_data.copy()
    van_osm_data['category'] = van_osm_data.apply(find_category, axis=1)
    van_osm_data.dropna(subset=['category'], inplace=True)
    van_osm_data.reset_index(drop=True, inplace=True)
    van_osm_data['neighborhood'].fillna(predict_neighborhood(van_osm_data), inplace=True)
    van_osm_data_temp = van_osm_data.copy()
    van_osm_data_temp = van_osm_data_temp[van_osm_data_temp['neighborhood'] != 'Metrotown']
    van_osm_data_temp.reset_index(drop=True, inplace=True)
    van_osm_data_temp.to_json('./processed/van_osm_data.json')
    van_osm_data = van_osm_data[van_osm_data.neighborhood.isin(airbnb_data.neighborhood.values)]
    van_osm_data.to_json('./processed/van_osm_data_refined.json')
    neighbor_osm_data = van_osm_data.groupby(['neighborhood', 'category']).size().reset_index()
    neighbor_osm_data.rename(columns={0: 'count'}, inplace=True)
    neighbor_osm_data = neighbor_osm_data.pivot(index='neighborhood', columns='category', values='count')
    neighbor_osm_data['score'] = airbnb_data.groupby('neighborhood').mean()['avg_score']
    neighbor_osm_data.fillna(0, inplace=True)
    neighbor_osm_data['total_amenity'] = neighbor_osm_data['arts'] + neighbor_osm_data['leisure'] + \
                                         neighbor_osm_data['sustenance'] + neighbor_osm_data['transportation']
    neighbor_osm_data['avg_price'] = airbnb_data.groupby('neighborhood').mean().price
    neigh_lat = np.array(
        [49.253659, 49.283439, 49.250538, 49.274353, 49.263745, 49.275107, 49.246320, 49.266337, 49.246247, 49.211640,
         49.227686, 49.249223, 49.245469, 49.245203, 49.210211, 49.273196, 49.218693, 49.264835])
    neigh_lon = np.array(
        [-123.160392, -123.124528, -123.184910, -123.047880, -123.128809, -123.066884, -123.075129, -123.162490,
         -123.076035, -123.129112, -123.122372, -123.038929, -123.102697, -123.122620, -123.116306, -123.088624,
         -123.090775, -123.201000])
    neighbor_osm_data['lat'] = neigh_lat
    neighbor_osm_data['lon'] = neigh_lon
    neighbor_osm_data.to_csv('./processed/neighborhood_info.csv')


def neighborhood_model():
    neighbor_training_data = osm_data[osm_data.city == 'Vancouver']
    neighbor_training_data = neighbor_training_data[neighbor_training_data.neighborhood.isnull() == False]
    neighbor_training_data_X = neighbor_training_data[['lat', 'lon']].values
    neighbor_training_data_y = neighbor_training_data['neighborhood'].values
    X_train, X_valid, y_train, y_valid = train_test_split(neighbor_training_data_X, neighbor_training_data_y)

    decison_tree_model = DecisionTreeClassifier(max_depth=18)
    decison_tree_model.fit(X_train, y_train)
    return decison_tree_model


def predict_neighborhood(location):
    model = neighborhood_model()
    X_pred = np.stack((location.lat.values, location.lon.values), axis=1)
    return pd.Series(model.predict(X_pred))


def main():
    vancouver_osm_data()

main()