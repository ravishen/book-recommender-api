
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('books.csv')


# In[3]:


df.head(5)


# In[4]:


df['books_count'].max()
print(df.columns)


# In[5]:


df[df['average_rating']==df['average_rating'].max()]


# In[6]:


ratings = pd.read_csv('ratings.csv')


# In[7]:


ratings.head()


# In[8]:


book_tags = pd.read_csv('book_tags.csv')


# In[9]:


book_tags.head()


# In[10]:


tags = pd.read_csv('tags.csv')


# In[11]:


tags.tail()


# In[12]:


tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
tags_join_DF.tail()


# # Author-Author Similarity

# In[13]:


df['authors']


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')


tfidf_matrix = tf.fit_transform(df['authors'])
print(tf.get_feature_names())
print(len(tf.get_feature_names()))
index =0
for i in tf.get_feature_names(): 
    if(i=="suzanne"):
        print(index)
    else:
        index+=1
print(tf.get_feature_names()[13084])
print(tf.get_feature_names()[2626])
print(tf.get_feature_names()[13085])
print(tf.vocabulary_)


# In[15]:


#print(tfidf_matrix[:])
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint

cv = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
cv_fit = cv.fit_transform(df['authors'])
print(type(cv_fit.toarray()[0]))
test_array = cv_fit.toarray()[0]
print(test_array)


x = min(cv.vocabulary_.items(), key=lambda x: x[1])[0]
print(cv.vocabulary_)
print(x)


# In[16]:


print(tfidf_matrix.shape)
print(tf.get_feature_names()[13084])
print(tf.get_feature_names()[2626])
print(tf.get_feature_names()[13085])
print(tfidf_matrix)
print(tfidf_matrix[9999,6746])


# In[17]:


from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[18]:


print(cosine_sim[28])


# In[19]:


cosine_sim.shape


# In[20]:


titles = df['title']
print(titles)


# In[21]:


indices = pd.Series(df.index, index=df['title'])
indices['Romeo and Juliet']


# # Recommendation by Author

# In[22]:


def authors_recommendations(title):
    idx = indices[title]
    print('idx = {}'.format(idx))
    sim_scores = list(enumerate(cosine_sim[idx]))
    print(len(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    print(sim_scores[:21])
    sim_scores = sim_scores[0:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]


# In[23]:


authors_recommendations('The Complete Sherlock Holmes')


# # Recommendation Using Tags/Genre

# In[24]:


books_with_genre = pd.merge(df,tags_join_DF,left_on='book_id',right_on='goodreads_book_id',how='inner')
books_with_genre


# In[25]:


tf1 = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix1 = tf1.fit_transform(books_with_genre['tag_name'].head(10000))
print(tfidf_matrix1.shape)
cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)


# In[26]:


cosine_sim1


# In[27]:


# Build a 1-dimensional array with book titles
titles1 = df['title']
indices1 = pd.Series(df.index, index=df['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def tags_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim1[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles1.iloc[book_indices]


# In[28]:


tags_recommendations('The Complete Sherlock Holmes')


# # Recommendation using both genre and author

# In[29]:


temp_df = books_with_genre.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
temp_df


# In[30]:


df = pd.merge(df, temp_df, left_on='book_id', right_on='book_id', how='inner')


# In[31]:


df.head()


# In[32]:


df['corpus'] = (pd.Series(df[['authors', 'tag_name']]
                .fillna('')
                .values.tolist()
                ).str.join(' '))


# In[33]:


df.head()


# In[34]:


tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(df['corpus'])
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

# Build a 1-dimensional array with book titles
titles = df['title']
indices = pd.Series(df.index, index=df['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def corpus_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

corpus_recommendations("The Complete Sherlock Holmes")


# # Using Collaborative Filtering
# 

# In[35]:


ratingsDF = pd.read_csv('ratings.csv')
ratingsDF.head(6)


# In[36]:


df.head(5)


# In[37]:


testdf = ratingsDF
testdf=testdf[['user_id','rating']].groupby(testdf['book_id'])
testdf.get_group(1)


# In[38]:


print(testdf.groups.keys())


# In[39]:


listOfDictonaries=[]
indexMap = {}
reverseIndexMap = {}
ptr=0;
for groupKey in testdf.groups.keys():
    tempDict={}

    groupDF = testdf.get_group(groupKey)
    for i in range(0,len(groupDF)):
        tempDict[groupDF.iloc[i,0]]=groupDF.iloc[i,1]
    indexMap[ptr]=groupKey
    reverseIndexMap[groupKey] = ptr
    ptr=ptr+1
    listOfDictonaries.append(tempDict)


# In[40]:


print(listOfDictonaries[9999])


# In[41]:


from sklearn.feature_extraction import DictVectorizer
dictVectorizer = DictVectorizer(sparse=True)
vector = dictVectorizer.fit_transform(listOfDictonaries)
print(dictVectorizer.vocabulary_)


# In[42]:


print(vector)


# In[43]:


from sklearn.metrics.pairwise import cosine_similarity
pairwiseSimilarity = cosine_similarity(vector)


# In[44]:


print(pairwiseSimilarity)


# In[45]:


titles = df['title']
indices = pd.Series(df.index, index=df['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def collab_filtering(title):
    idx = indices1[title]
    sim_scores = list(enumerate(pairwiseSimilarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]


# In[46]:


collab_filtering('The Complete Sherlock Holmes')

