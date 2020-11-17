import pandas as pd
import numpy as np
import json
import sys

def Contains_helper(str):
    return "Chinese" in str and "Restaurants" in str


def main():

    city_list = ['Toronto', 'Calgary', 'Mississauga', 'Markham', 'Scarborough', 'Montréal']

    business = pd.read_json(sys.argv[1], lines=True)[['business_id','name','city','categories']].dropna()
    selected_business = business[ (business['categories'].apply(Contains_helper)) 
                                    & (business['city'].isin(city_list)) ]

    raw_reviews = pd.read_json(sys.argv[2], lines=True, chunksize=50000)
    reviews = pd.DataFrame(columns=['business_id', 'stars'])
    for chunk in raw_reviews:
        trimed = chunk[['business_id', 'stars']]
        result = trimed[trimed['business_id'].isin( selected_business['business_id'].values )]
        reviews = reviews.append(result, ignore_index=True)
        
    reviews["stars"] = reviews["stars"].astype(float)
    reviews = reviews.merge(selected_business,left_on='business_id',right_on='business_id')

    reviews.to_csv(sys.argv[3], index=False)

if __name__ == '__main__':
    main()