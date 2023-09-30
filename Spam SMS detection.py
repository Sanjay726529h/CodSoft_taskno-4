#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
from tensorflow.keras import layers


# In[9]:


df=pd.read_csv("spam.csv",encoding='latin-1')
df.head()


# In[10]:


df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)


# In[11]:


df=df.rename(columns={'v1':'label','v2':'text'})
df['label_enc']=df['label'].map({'ham':0,'spam':1})
df.head()


# In[12]:


sb.countplot(x=df['label'])
plt.show()


# In[13]:


#Find the average number of tokenn in the sentences
avg_words_len=round(sum([len(i.split())for i in df['text']])/len (df['text']))
print(avg_words_len)


# In[14]:


#Finding the total no of unique words in corpus
s=set()
for sent in df['text']:
    for word in sent.split():
        s.add(word)
total_words_length=len(s)
print(total_words_length)


# In[15]:


#splititing data for Training and Testing 
from sklearn.model_selection import train_test_split

X,y=np.asanyarray(df['text']),np.asanyarray(df['label_enc'])
new_df=pd.DataFrame({'text':X,'label':y})
X_train,X_test,y_train,y_test=train_test_split(new_df['text'],new_df['label'],test_size=0.3,random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer  # Corrected the import name
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score  # Corrected the import name

# Create the TF-IDF vectorizer
tfidf_vec = TfidfVectorizer().fit(X_train)

# Transform the training and testing data
X_train_vec = tfidf_vec.transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)  # Corrected the variable name to 'X_test_vec'

# Create a Multinomial Naive Bayes model
baseline_model = MultinomialNB()

# Fit the model to the training data
baseline_model.fit(X_train_vec, y_train)


# In[17]:


nb_accuracy=accuracy_score(y_test,baseline_model.predict(X_test_vec))
print(nb_accuracy)
print(classification_report(y_test,baseline_model.predict(X_test_vec)))


# In[18]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(baseline_model,X_test_vec,y_test)


# In[19]:


import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[21]:


with open('spam_sms_detection_model.pkl', 'wb') as file:
    pickle.dump(baseline_model, file)


# In[25]:


with open('spam_sms_detection_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[34]:


joblib.dump(tfidf_vec, 'tfidf_vectorizer.pkl')


# In[38]:


import joblib

model = joblib.load('spam_sms_detection_model.pkl')
tfidf_vec = joblib.load('tfidf_vectorizer.pkl')

# Define the function to classify a message
def classify_message(input_message):
    # Transform the input message into a TF-IDF vector
    message_vec = tfidf_vec.transform([input_message])
    
    # Make a prediction
    prediction = model.predict(message_vec)
    
    # Return the result
    if prediction == 1:
        return "spam"
    else:
        return "ham"

# Take user input
user_input = input("Enter a message to classify as ham or spam: ")

# Call the classify_message function and display the result
result = classify_message(user_input)
print("The message is classified as:", result)

