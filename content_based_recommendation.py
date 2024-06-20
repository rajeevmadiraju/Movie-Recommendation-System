import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
import warnings
keywords = pd.read_csv('keywords.csv')
links_small = pd.read_csv('links.csv')
md = pd.read_csv('movies_metadata.csv')
ratings = pd.read_csv('ratings.csv')
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i[
    'name'] for i in x] if isinstance(x, list) else [])

vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')

# this is R
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')

# this is C
C = vote_averages.mean()
m = vote_counts.quantile(0.95)
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = md[(md['vote_count'] >= m) & 
               (md['vote_count'].notnull()) & 
               (md['vote_average'].notnull())][['title', 
                                                'year', 
                                                'vote_count', 
                                                'vote_average', 
                                                'popularity', 
                                                'genres']]

qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)
qualified.head(15)
s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)
gen_md.head(3).transpose()
def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & 
                   (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: 
                        (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C),
                        axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified
  build_chart('Romance').head(15)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings
def convert_int(x):
	try:
		return int(x)
	except:
		return np.nan
		
md['id']= md['id'].apply(convert_int)

print("\n\nRows with missing IDs:\n{}".format(md[md['id'].isnull()]))

#Drop rows with missing IDs 

md= md.drop([19730, 29503, 35587])
md['id']= md['id'].astype('int')

links= pd.read_csv("links_small.csv")
links= links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

smd= md[md['id'].isin(links)]
print("\n\n",smd.shape)


# Movie Description Based Recommender
smd['tagline']= smd['tagline'].fillna('')
smd['description']= smd['overview']#+smd['tagline']
smd['description']= smd['description'].fillna('')


tf= TfidfVectorizer(analyzer= 'word', ngram_range=(1,10), min_df=0, stop_words='english')
tfidf_matrix= tf.fit_transform(smd['description'])
print("\n\nTF-IDF Matrix shape:\t{}".format(tfidf_matrix.shape))

#Cosine Similarity
cosine_sim= linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim[0])

smd= smd.reset_index()
titles= smd['title']
indices= pd.Series(smd.index, index=smd['title'])


def get_recommendations(title):
	idx= indices[title]
	sim_scores= list(enumerate(cosine_sim[idx]))
	sim_scores= sorted(sim_scores, key=lambda x: x[1], reverse=True)
	sim_scores= sim_scores[1:31]
	movie_indices= [i[0] for i in sim_scores]
	return titles.iloc[movie_indices]
	
	
print("\n\n Recommendations are: \t{}".format(get_recommendations('The Godfather').head(10)))
print("\n\n Recommendations are: \t{}".format(get_recommendations('Toy Story').head(10)))
