#!/usr/bin/env python
# coding: utf-8

# In[4]:


#About the Data
'''This dataset consists of a nearly 3000 Amazon customer reviews (input text), 
#star ratings, date of review, variant and feedback of various amazon Alexa 
#products like Alexa Echo, Echo dots, Alexa Firesticks etc. for learning how to 
train Machine for sentiment analysis.

What you can do with this Data ?
You can use this data to analyze Amazon’s Alexa product ; discover insights into 
consumer reviews and assist with machine learning models.You can also train your 
machine models for sentiment analysis and analyze customer reviews how many positive
reviews ? and how many negative reviews ?
'''

#basic
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.figure_factory as ff

import os
print(os.listdir('C:/Users/kimso/Desktop/amazon'))





# In[5]:


data = pd.read_csv('C:/Users/kimso/Desktop/amazon/amazon_alexa.tsv',delimiter='\t', quoting=3)
data.shape
data.head()


# In[6]:


data.describe()


# In[7]:


data.isnull().values.any()


# In[8]:


data['length of review'] = data['verified_reviews'].apply(len)
data.groupby('length of review').describe().sample(10)


# In[9]:


data.groupby('rating').describe()


# In[10]:


data.groupby('feedback').describe()


# In[11]:


ratings = data['rating'].value_counts()
label_rating = ratings.index
size_rating = ratings.values
colors=['pink','lightblue','aqua','gold','crimson']
rating_piechart= go.Pie(labels=label_rating,
                       values=size_rating,
                       marker=dict(colors=colors),
                        name='Alexa', hole=0.3
                       )
df= [rating_piechart]
layout= go.Layout(
        title= 'Distribution of ratings for Alexa')
fig= go.Figure(data=df,
              layout=layout)
py.iplot(fig)


# In[12]:


color = plt.cm.copper(np.linspace(0,1,15))
data['variation'].value_counts().plot.bar(color=color,figsize=(15,9))
plt.title('Distribution of variation in Alexa', fontsize=20)
plt.xlabel('variations')
plt.ylabel('count')
plt.show


# In[13]:


feedbacks= data['feedback'].value_counts()
label_feedback= feedbacks.index
size_feedback = feedbacks.values
colors=['yellow','lightgreen']
feedback_piechart= go.Pie(labels=label_feedback,
                         values=size_feedback,
                         marker= dict(colors=colors),
                         name= 'Alexa', hole=0.3)
df2 = [feedback_piechart]

layout= go.Layout(
            title= 'Distribution of Feedbacks for Alexa')
fig= go.Figure(data=df2,
              layout=layout)

py.iplot(fig)


# In[14]:


data['length of review'].value_counts().plot.hist(color='skyblue', figsize=(15,5), bins = 50)
plt.title('distribution of length of reviews')
plt.xlabel('length of review')
plt.ylabel('count')
plt.show()


# In[15]:


data[data['length of review']==200]['verified_reviews'].iloc[0]


# In[16]:


plt.rcParams['figure.figsize']=(15,9)
plt.style.use('fivethirtyeight')

sns.swarmplot(data['variation'], data['length of review'], palette='deep')
plt.title('variation vs length of rating')
plt.xticks(rotation=90)
plt.show()


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(stop_words='english')
#print(cv)
words= cv.fit_transform(data.verified_reviews)
#print(words)
sum_words =  words.sum(axis=0)
#print(sum_words)

words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse= True)
#dictionary with the most popular words at the top
frequency =  pd.DataFrame(words_freq, columns=['word', 'freq'])

plt.style.use('fivethirtyeight')
color = plt.cm.ocean(np.linspace(0,1,20))
#linspace splits diff between 0 and 1 into 20 parts
frequency.head(20).plot(x='word')
plt.style.use('fivethirtyeight')
color = plt.cm.ocean(np.linspace(0,1,20))
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize= (15,6), color=color)
#bar graph, top 20 rows, word frequency
plt.title('top 20 most freq words')
plt.show()


# In[18]:


#wordcloud

from wordcloud import WordCloud
wordcloud = WordCloud(background_color = 'lightcyan', width = 200, height=2000).generate_from_frequencies(dict(words_freq))
plt.style.use('fivethirtyeight')
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(wordcloud)
plt.title('vocab from review',
        fontsize=20)
plt.show()


# In[19]:


import spacy
nlp = spacy.load('en_core_web_sm')
def explain_text_entities(text):
    doc =  nlp(text)
    for ent in doc.ents:
        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label)}')
for i in range(15,50):
    one_sent = data['verified_reviews'][i]
    doc = nlp(one_sent)
    spacy.displacy.render(doc, style='ent', jupyter=True)


# In[20]:


#clean text
#import NLP libs

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[21]:


corpus = []
for i in range(0,3150):
    review = re.sub('[^a-zA-Z]',' ', data['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()
y =  data.iloc[:,4].values
print(x.shape)
print(y.shape)


# In[23]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x, y,test_size=0.3, random_state=15)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[24]:


'''Transform features by scaling each feature to a given range.

This estimator scales and translates each feature individually such that it 
is in the given range on the training set, e.g. between zero and one.'''


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
x_train =  mm.fit_transform(x_train)
x_test = mm.transform(x_test)


# In[25]:


from  sklearn.ensemble import RandomForestClassifer
from sklearn.metrics import confusion_matrix
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('training accurary: ', model.score(x_train, y_train))
print('testing accurary: ', model.score(x_test, y_test))

cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:





# In[ ]:




