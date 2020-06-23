#!/usr/bin/env python
# coding: utf-8

# In[14]:


import re

replace_no_space = re.compile("[.;:!\'?,\"()\[\]]")
replace_with_space = re.compile("(<br\s*\/><br\s*\/>)|(\-)|(\/)|(\\n)")

def preprocess_reviews(reviews):
    reviews = [replace_no_space.sub("",line.lower()) for line in reviews]
    reviews = [replace_with_space.sub(" ", line) for line in reviews]
    return reviews

class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self,text,overall):
        self.text = text
        self.overall = overall
        self.sentiment = self.get_sentiment()
        
    def get_sentiment(self):
        if self.overall <= 2:
            return Sentiment.NEGATIVE
        elif self.overall == 3:
            return Sentiment.NEUTRAL
        else: #Score of 4 or 5
            return Sentiment.POSITIVE


# In[15]:


import json

file_name = (r'C:\Users\dongn\Desktop\ML_practice\Projects\Yelp\Clothing_Shoes_and_Jewelry_5.json')

reviews= []
with open(file_name, encoding="utf8") as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'].strip(), review['overall']))
        


# # Prep Data

# In[16]:


from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size = 0.33, random_state=42)


# In[17]:


train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]

test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]


# # regex data

# In[18]:


train_x = preprocess_reviews(train_x) #remove commas, periods, semi-colons and so on
train_x = [x.lower() for x in train_x] #lower case

test_x = preprocess_reviews(test_x) 
test_x = [x.lower() for x in test_x]


# # Bag of words vectorization

# In[19]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)


# # Classification

# In[ ]:


from sklearn import svm

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
clf_svm.predict(test_x_vectors[2])


# In[ ]:


from sklearn import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier()
clf_dt.fit(train_x_vectors, train_y)
clf_dt.predict(test_x_vectors[2])


# # F1 Score

# In[ ]:


from sklearn.metrics import f1_score

f1_score(test_y, clf_svm.predict(test_x_vectors), average= None, labels = [Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment. NEGATIVE])


# # Credit to Keith Galli for his youtube tutorials. I learned a ton about machine learning and python ####

# In[ ]:




