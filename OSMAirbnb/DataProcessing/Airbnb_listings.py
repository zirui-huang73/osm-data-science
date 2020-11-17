# this script cleans and process the raw airbnb listings data (raw_listings.csv)
# neighborhood added
# reviews of airbnb were examined
# average rate of each airbnb added
# neighborhoods with less than 50 airbnbs were dropped
# input: raw_listings.csv reviews.csv.gz finalOSM.json language.tsv
# output: listings.edited.json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()


def airbnb_data():
    airbnb_listings = pd.read_csv('./raw/raw_listings.csv')
    reviews = pd.read_csv('./raw/reviews.csv.gz')

    airbnb_listings.drop(columns=['host_id', 'host_name', 'neighbourhood_group', 'neighbourhood', 'room_type',
                                  'last_review', 'reviews_per_month', 'calculated_host_listings_count'], inplace=True)
    airbnb_listings.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    airbnb_listings = airbnb_listings[airbnb_listings['number_of_reviews'] > 5]
    airbnb_listings.reset_index(drop=True, inplace=True)
    airbnb_listings = airbnb_listings[airbnb_listings['price'] < 500]
    airbnb_listings.reset_index(drop=True, inplace=True)
    model = neighborhood_model()
    airbnb_listings['neighborhood'] = predict_neighborhood(airbnb_listings, model)
    listings_size = airbnb_listings.groupby('neighborhood').size().sort_values(ascending=False)
    listings_size = listings_size[listings_size > 50]
    airbnb_listings = airbnb_listings[airbnb_listings.neighborhood.isin(listings_size.reset_index().neighborhood.values)]
    airbnb_listings.reset_index(drop=True, inplace=True)
    reviews.drop(columns=['id', 'date', 'reviewer_id', 'reviewer_name'], inplace=True)
    reviews = reviews[reviews['listing_id'].isin(airbnb_listings.id.values)]

    reviews = reviews.copy()
    stops = stopwords.words('english')
    reviews['p_comments'] = reviews['comments'].str.lower()
    reviews['p_comments'] = reviews['p_comments'].str.replace('[^\w\s]', ' ')
    reviews.dropna(subset=['p_comments'], inplace=True)
    reviews['p_comments'] = reviews['p_comments'].apply(lambda x: " ".join(x for x in x.split() if x not in stops))
    predict_language(reviews)
    reviews = reviews.copy()
    print('start')
    reviews['score'] = reviews['comments'].apply(sentiment_score)
    print('end')
    reviews = reviews.copy()
    airbnb_listings = airbnb_listings.copy()
    airbnb_listings['avg_score'] = reviews.groupby('listing_id').mean().score.values
    airbnb_listings.reset_index(drop=True, inplace=True)
    airbnb_listings.to_json('./processed/listings_edited.json')

def neighborhood_model():
    osm_data = pd.read_json('./processed/finalOSM.json')
    neighbor_training_data = osm_data[osm_data.city == 'Vancouver']
    neighbor_training_data = neighbor_training_data[neighbor_training_data.neighborhood.isnull() == False]
    neighbor_training_data_X = neighbor_training_data[['lat', 'lon']].values
    neighbor_training_data_y = neighbor_training_data['neighborhood'].values
    X_train, X_valid, y_train, y_valid = train_test_split(neighbor_training_data_X, neighbor_training_data_y)

    decison_tree_model = DecisionTreeClassifier(max_depth=18)
    decison_tree_model.fit(X_train, y_train)
    return decison_tree_model

def predict_language(reviews):
    language = pd.read_csv('./raw/language.tsv', sep='\t')
    language = language[language['Ambiguous'] == 0]
    language = language[['Tweet', 'Definitely English']]
    language = language.copy()
    language['Tweet'] = language['Tweet'].str.lower()
    language['Tweet'] = language['Tweet'].str.replace('[^\w\s]', ' ')
    stops = stopwords.words('english')
    language['Tweet'] = language['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stops))
    allwords = pd.Series(' '.join(language['Tweet']).split()).value_counts()
    rarewords = allwords[allwords < 3]
    language['Tweet'] = language['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in rarewords))

    X_train, X_valid, y_train, y_valid = train_test_split(language['Tweet'].values,
                                                          language['Definitely English'].values)
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(language['Tweet'])
    X_train_count = count_vect.transform(X_train)

    logreg_model = LogisticRegression(C=1)
    logreg_model.fit(X_train_count, y_train)

    review_count = count_vect.transform(reviews['p_comments'])
    reviews['english'] = logreg_model.predict(review_count)


def predict_neighborhood(location, model):
    X_pred = np.stack((location.lat.values, location.lon.values), axis=1)
    return pd.Series(model.predict(X_pred))


def sentiment_score(text):
    score = analyser.polarity_scores(text)
    return score['compound']


def main():
    airbnb_data()


main()