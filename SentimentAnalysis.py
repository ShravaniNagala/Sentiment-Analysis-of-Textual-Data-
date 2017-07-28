
# coding: utf-8

# In[1]:

from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
from sklearn.svm import LinearSVC
# In[3]:

import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer


# In[19]:

from sklearn.svm import LinearSVC


# In[4]:

from sklearn.model_selection import train_test_split


# In[5]:

from sklearn.grid_search import GridSearchCV


# In[6]:

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[7]:

from sklearn.pipeline import Pipeline


# In[8]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier


# In[9]:

import numpy as np


# In[2]:

from sklearn.naive_bayes import MultinomialNB


# In[16]:

from sklearn.linear_model.logistic import LogisticRegression


# In[21]:

import csv


# In[10]:

df = pd.read_csv('train.tsv', header=0, delimiter='\t')
dft = pd.read_csv('test.tsv', header=0, delimiter='\t')

# In[7]:

df['Sentiment'].value_counts()


# In[20]:


def stem(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]
 
classifySVM = Pipeline([
    ('vect', TfidfVectorizer(tokenizer=stem,
                             stop_words=stopwords.words('english') + list(string.punctuation))),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC())),
])

# In[11]:

X, y = df['Phrase'], df['Sentiment'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

Z_test=dft['Phrase']
# In[21]:

classifySVM.fit(X_train, y_train)


# In[22]:

predictedSVM = classifySVM.predict(X_test)
checkSVM = classifySVM.predict(Z_test)

# In[20]:
print ('USING SVM TO CLASSIFY:')
print ('Accuracy:', accuracy_score(y_test, predictedSVM))
print ('Confusion Matrix:', confusion_matrix(y_test, predictedSVM))
print ('Classification Report:', classification_report(y_test, predictedSVM))


# In[30]:

out = open('SVMPrediction_testF1.csv', 'w')
out.write('category => Phrase')
for phrase, cat in zip(Z_test, checkSVM):
    out.write('%s => %s' % (cat, phrase))
    #out.write(str(row))
    out.write('\n')
out.close()

NaiveBayes = Pipeline([
    ('vect', TfidfVectorizer(tokenizer=stem,
                             stop_words=stopwords.words('english') + list(string.punctuation))),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(MultinomialNB()))])


# In[13]:

NaiveBayes.fit(X_train, y_train)


# In[14]:

predictedNB = NaiveBayes.predict(X_test)
checkNB = NaiveBayes.predict(Z_test)

# In[15]:
print ('USINGMULTINOMIAL NAIVE BAYES TO CLASSIFY:')
print ('Accuracy:', accuracy_score(y_test, predictedNB))
print ('Confusion Matrix:', confusion_matrix(y_test, predictedNB))
print ('Classification Report:', classification_report(y_test, predictedNB))

outnb = open('MNBPrediction_testF1.csv', 'w')
outnb.write('category => Phrase')
for phrase, cat in zip(Z_test, checkNB):
    outnb.write('%s => %s' % (cat, phrase))
    #out.write(str(row))
    outnb.write('\n')
outnb.close()
# In[ ]:

logisticReg = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
    ])
    
parameters = {
    'vect__max_df': (0.25, 0.5),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'clf__C': (0.1, 1, 10),
    }


# In[ ]:

grid_search = GridSearchCV(logisticReg, parameters, n_jobs=3, verbose=1, scoring='accuracy')


# In[ ]:

grid_search.fit(X_train, y_train)


# In[ ]:

predictedLR = grid_search.predict(X_test)
checkLR = grid_search.predict(Z_test)

print ('USING LOGISTIC REGRESSION WITH PARAMETER TUNING')
print ('Accuracy:', accuracy_score(y_test, predictedLR))
print ('Confusion Matrix:', confusion_matrix(y_test, predictedLR))
print ('Classification Report:', classification_report(y_test, predictedLR))

outlr = open('LRPrediction_testF1.csv', 'w')
outlr.write('category => Phrase')
for phrase, cat in zip(Z_test, checkLR):
    outlr.write('%s => %s' % (cat, phrase))
    #out.write(str(row))
    outlr.write('\n')
outlr.close()


# In[ ]:



