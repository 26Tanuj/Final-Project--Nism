#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importong libraries
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from string import punctuation
# import tensorflow as tf
import os
import re
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D



data = pd.read_excel("TanujDataset.xlsx")


# In[22]:


# data


# In[2]:


x = data["text"]
y = data["label"]

cv = CountVectorizer()
# x = cv.fit_transform(x)


# In[3]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(x)
freq_term_matrix = count_vectorizer.transform(x)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
# print(tf_idf_matrix)


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(tf_idf_matrix,y, random_state=0)


# In[30]:


# 1 Naive-Bayes 
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state=42)
# model = MultinomialNB()
# model.fit(x_train,y_train)


# In[31]:


# print(model.score(x_test,y_test)*100)


# In[29]:


# news_headline = "Ukraine slaps sanctions on senior clerics in pro-Moscow church"
# df = count_vectorizer.transform([news_headline]).toarray()
# print(model.predict(df))


# In[28]:


# # 2 Logistic Regression
# from sklearn.linear_model import LogisticRegression

# logreg = LogisticRegression()
# logreg.fit(x_train,y_train)
# Accuracy = logreg.score(x_test,y_test)
# print(Accuracy*100)


# In[6]:


# 3 Decision Tree\
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
accuracyclf = clf.score(x_test,y_test)
# print(accuracyclf*100)


# In[5]:


# 4 Passive_Agressive Classifier 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(x_train,y_train)
y_pred = pac.predict(x_test)
score = accuracy_score(y_test,y_pred)
# print(f'Accuracy : {round(score*100,2)}%')


# In[13]:


# pip install Jinja2


# In[32]:


import streamlit as st
import time
st.title("Fake News Detection System")
def fakenewsdetection():
    time.sleep(2)
    user = st.text_area("Enter or Copy Any News Headline from Twitter,News Sites etc... : ")
    result = st.button("Click here to check news.")
    if len(user) < 1:
        st.write()
    else:
        sample = user
        userdata = count_vectorizer.transform([sample]).toarray()
        a = pac.predict(userdata)
#         b = st.title(a)
#         st.title(a)
    if result:
        st.write("looks like the news is  ",str(a))
    else:
        st.write(' ')
fakenewsdetection()


# In[26]:


# news_headline = "“Gas prices are down back to where they were before Russia invaded Ukraine."
# df = count_vectorizer.transform([news_headline]).toarray()
# print(clf.predict(df))

