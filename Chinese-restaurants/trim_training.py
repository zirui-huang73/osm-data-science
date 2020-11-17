import pandas as pd
import numpy as np
import sys


def Contains_helper(str):
    return "Chinese" in str and "Restaurants" in str


def get_attribute(row, key):
    row_dict = row['attributes']
    if key in row_dict.keys():
        if row_dict[key] == 'None':
            return None
        return row_dict[key]
    else:
        return None


def main():
	business = pd.read_json(sys.argv[1], lines=True).dropna()
	Chinese_Restaurants = business[ (business['categories'].apply(Contains_helper)) ][['business_id','name','city','attributes','stars']]

	Chinese_Restaurants['delivery'] = Chinese_Restaurants.apply(get_attribute, args=("RestaurantsDelivery",), axis=1)
	Chinese_Restaurants['NoiseLevel'] = Chinese_Restaurants.apply(get_attribute, args=("NoiseLevel",), axis=1)
	Chinese_Restaurants['PriceRange'] = Chinese_Restaurants.apply(get_attribute, args=("RestaurantsPriceRange2",), axis=1)
	Chinese_Restaurants['TakeOut'] = Chinese_Restaurants.apply(get_attribute, args=("RestaurantsTakeOut",), axis=1)
	Chinese_Restaurants['GoodForKids'] = Chinese_Restaurants.apply(get_attribute, args=("GoodForKids",), axis=1)
	Chinese_Restaurants['OutdoorSeating'] = Chinese_Restaurants.apply(get_attribute, args=("OutdoorSeating",), axis=1)
	Chinese_Restaurants['HasTV'] = Chinese_Restaurants.apply(get_attribute, args=("HasTV",), axis=1)

	Chinese_Restaurants = Chinese_Restaurants.dropna()
	Chinese_Restaurants["PriceRange"] = Chinese_Restaurants["PriceRange"].astype(int)
	Chinese_Restaurants = Chinese_Restaurants.drop(columns=['attributes']).reset_index(drop=True)
	Chinese_Restaurants.to_csv(sys.argv[2], index=False)


# python trim_training.py yelp_academic_dataset_business.json trimed.csv
if __name__ == '__main__':
    main()