import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

#reading file
movies = pd.read_csv("movies.csv", header=0)
movies = movies.replace({np.nan: None}) # replace NaN with None
movies['imdb_id']=movies['imdb_id'].loc[:6]
print(movies.head())

ratings = pd.read_csv("ratings_small.csv", header=0)
print(ratings.head())

links = pd.read_csv("links.csv", header=0)
print(links.head())

#Total no. of movies
print(len(movies))

links.columns=["movieId","imdb_id","tmdbId"]
print(links.head())
#mergeddf=movies.append([links,ratings])
#print(mergeddf.head())
#movies['language']=movies['language'].astype(basestring)
print(movies.dtypes)

#Recommender system based on movie similarity
df2=movies[0:10000]
print(df2['description'].head(5))

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['description'] = df2['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['description'])

#Output the shape of tfidf_matrix
print(tfidf_matrix.shape)

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises')

"""
#Genre wise distribution
all_genres = [s.split(", ") for s in movies[movies.genres.notnull()].genres]
genres = [item for l in all_genres for item in l ]
unique_genres = set(genres)
print (f"total of {len(unique_genres)} unique genres from {len(genres)} occurances.")
pd.Series(genres).value_counts().plot(kind='bar', figsize=(10, 3))
plt.title("# of movies per genre")
plt.ylabel("# of movies")
plt.xlabel("genre")
plt.show()


#Year wise distribution
def get_year(date):
    year = None
    if date:
        year = date[:4]
    return year

movies['year'] =movies.date.apply(lambda date:get_year(date))

years = movies[movies.year.notnull()].year # get rows where year is not None
print (f"Total of {len(set(years))} uinque years from {min(years)} to {max(years)}")
pd.Series(years).value_counts().sort_index().plot(kind='bar', figsize=(30, 5))
plt.title("# of movies over time")
plt.ylabel("# of movies")
plt.xlabel("year")
plt.show()

# Sort by popularity
sorted_by_popularity = movies.sort_values(by='popularity', ascending=False)
print ("Most popular movies:\n", sorted_by_popularity['title'].values[:10])

# Get poster image
from IPython.core.display import display, HTML
prefix_url = 'http://image.tmdb.org/t/p/'
poster_sizes = ["w92", "w154", "w185", "w342", "w500", "w780", "original"]
full_url = prefix_url + poster_sizes[2] + movies.poster_url[0]
display(HTML(f'<img src="{full_url}">'))

# Load ratings
ratings = pd.read_csv("ratings_small.csv", header=0)
print(ratings.head())

# Distribution of ratings
print (ratings['rating'].describe())
pd.Series(ratings['rating']).value_counts().sort_index().plot(kind='bar', figsize=(5, 2))
plt.title("Distribution of ratings")
plt.ylabel("# of movies")
plt.xlabel("rating")
plt.show()
"""
