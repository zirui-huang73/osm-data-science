import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import sys

def Contains_helper(str):
    return "Chinese" in str and "Restaurants" in str


def main():

    reviews = pd.read_csv(sys.argv[1])

    reviews["stars"] = reviews["stars"].astype(float)
    
    city_list = ['Toronto', 'Calgary', 'Mississauga', 'Markham', 'Scarborough', 'Montréal']

    df_dict = {city: reviews[reviews['city'] == city].reset_index()['stars'] for city in city_list} 
    cities_rv = pd.DataFrame(df_dict).dropna()

    melted_anova_pvalue = pd.melt(cities_rv)
    posthoc = pairwise_tukeyhsd(
       melted_anova_pvalue['value'], melted_anova_pvalue['variable'],
       alpha=0.05)

    print(posthoc)

    posthoc.plot_simultaneous()

    plt.suptitle("")
    plt.show()


# python review_restaurants.py yelp_academic_dataset_business.json yelp_academic_dataset_review.json
if __name__ == '__main__':
    main()