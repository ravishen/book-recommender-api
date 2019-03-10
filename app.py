import pandas as pd
import numpy as np
from flask import Flask,jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

df = pd.read_csv('books.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')

tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')

books_with_genre = pd.merge(df,tags_join_DF,left_on='book_id',right_on='goodreads_book_id',how='inner')


temp_df = books_with_genre.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
df = pd.merge(df, temp_df, left_on='book_id', right_on='book_id', how='inner')
df['corpus'] = (pd.Series(df[['authors', 'tag_name']]
                .fillna('')
                .values.tolist()
                ).str.join(' '))

tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(df['corpus'])
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

# Build a 1-dimensional array with book titles
titles = df['title']
indices = pd.Series(df.index, index=df['title'])

titles1 = df['title']
indices1 = pd.Series(df.index, index=df['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def corpus_recommendations(title):

#    idx = [k.lower() for k in indices1[title]]
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]




@app.route('/recommend/<string:name>')
def get_stores(name):
    name = name.lower()
    recommended_list = corpus_recommendations(name).to_dict()
    return jsonify(recommended_list)

app.run(port=5000)
