#!/usr/bin/env python
# coding: utf-8

# # SMS SPAM DETECTION DATASET

# In[1]:


#Importing necessary libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#IMPORTING THE DATASET


# In[4]:


data = pd.read_csv('C:/Users/ASUS/Downloads/spam.csv', encoding ='ISO-8859-1')
data.head()


# In[5]:


data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis =1,inplace =True)


# In[6]:


data.head()


# In[7]:


data.shape


# # DATA CLEANING

# In[8]:


data.info()


# In[ ]:


#DROPPING DUPLICATE VALUES


# In[9]:


data.drop_duplicates(inplace =True)


# In[10]:


data.shape


# In[11]:


# MISSING VALUES


# In[12]:


data.isnull().sum()


# # EDA

# In[13]:


data['Category'].value_counts()


# In[14]:


sns.countplot('Category',data=data)


# In[15]:


data_ohe = pd.get_dummies(data['Category'], drop_first= True)
data_ohe.head()


# In[16]:


# here we have te value 0 for ham and  1 for spam


# In[17]:


data =pd.concat([data,data_ohe], axis=1)
data.head()


# In[18]:


data.drop('Category',axis =1,inplace=True)
data.head()


# In[19]:


data.rename(columns={'spam':'Target'},inplace =True)


# In[20]:


data.head()


# In[21]:


# we have to calculate how many sentances ,words and charecters are used in the messages


# In[22]:


get_ipython().system('pip install nltk')


# In[23]:


import nltk


# In[24]:


nltk.download('punkt')


# In[25]:


data['num_char']= data['Message'].apply(len)   # creating column for num of charecters in the meassages
data.head()


# In[26]:


#creating a column for nu of word in mwssages
data['num_word']=data['Message'].apply(lambda x: len(nltk.word_tokenize(x)))


# In[27]:


data.head()


# In[28]:


#creating column for no of sentences


# In[29]:


data['num_sent']=data['Message'].apply(lambda x:len(nltk.sent_tokenize(x)))
data.head()


# In[30]:


data[data['Target']==0][['num_char','num_word','num_sent']].describe() # for ham dataset 


# In[31]:


#here we can see that for ham mesaages the max no of words  is 220 and charecters is 910 no of sentence is 38
#In total there are 4516 rows of ham datas


# In[32]:


data[data['Target']==1][['num_char','num_word','num_sent']].describe()


# In[33]:


#Comparing with ham datas spam messages are having very less no of charecters ,words and sentences


# In[34]:


plt.figure(figsize=(12,7))
sns.histplot(data[data['Target']==0]['num_char']) #for ham messages
sns.histplot(data[data['Target']==1]['num_char'],color='red') #for spam
plt.title('Ham and Spam messages ')


# In[35]:


plt.figure(figsize=(12,7))
sns.histplot(data[data['Target']==0]['num_word']) #for ham messages
sns.histplot(data[data['Target']==1]['num_word'],color='red') #for spam
plt.title('Ham and Spam messages ')


# In[36]:


corr =data.corr()
sns.heatmap(corr,annot= True)


# In[37]:


#here we can seethat there is a strong corelation between varieble there is multicolinearity


# In[38]:


nltk.download('stopwords')


# In[39]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[40]:


import string
string.punctuation


# Finding out the top words used in HAm and Spam messages

# In[41]:


from nltk.stem.porter import PorterStemmer
ps =PorterStemmer()
ps.stem('Dancing')

# Data preprocessing
. Lower case
. Tokenization
. Removing special charecters
. Removing stop words and punctuations
. Stemming
# In[42]:


def transform_text(text):
    text=text.lower()                         #lowercase
    text=nltk.word_tokenize(text )             #tokenising
    y=[]
    for i in text:
        if i.isalnum():                            #removing special cahrecters
            y.append(i)
    text =y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text =y[:]
    y.clear()
    
    for  i in text:
            y.append(ps.stem(i))     #stemming
            
          
            
    return " ".join(y)


# In[43]:


transform_text(data['Message'][10])


# In[44]:


data['Message'][10]


# In[45]:


# creating a extra column of transformed texts


# In[46]:


data['Transformed_text'] = data['Message'].apply(transform_text)


# In[47]:


data.head()


# In[48]:


#Using wordcloud to highlight the important words


# In[49]:


get_ipython().system('pip install wordcloud')


# In[50]:


from wordcloud import WordCloud
wc = WordCloud(width=600,height=500,min_font_size=12,background_color='black')


# In[51]:


spam_wc = wc.generate(data[data['Target']== 1]['Transformed_text'].str.cat(sep=' '))    #wordcloud for spam messages showing most occuring words


# In[52]:


plt.figure(figsize=(10,7))
plt.imshow(spam_wc)


# In[53]:


#here are the words that occur in spam messages most commonly free,call,text etc are the most occuring words


# In[54]:


ham_wc = wc.generate(data[data['Target']== 0]['Transformed_text'].str.cat(sep=' '))
plt.figure(figsize=(10,7))
plt.imshow(ham_wc)   #for ham messages 


# In[55]:


# for personal messages we can see that call,love,u,come occures moslty


# In[56]:


from collections import Counter
#importing counter so that we can count the words that occur mostly


# In[57]:


spam_corpus=[]
for msg in (data[data['Target']==1]['Transformed_text'].tolist()):
    for words in msg.split():
        spam_corpus.append(words)


# In[58]:


len(spam_corpus)


# In[59]:


plt.figure(figsize=(7,5))
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation ='vertical')
plt.show()


# here we can see that call is the most occuring word followed by free these are alll the common words which we can see in our spam messgaes also,because these messages gives mostly offers 

# In[60]:


ham_corpus=[]
for msg in (data[data['Target']==0]['Transformed_text'].tolist()):
    for words in msg.split():
        ham_corpus.append(words)
        


# In[61]:


len(ham_corpus)


# In[62]:


plt.figure(figsize=(7,5))
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation ='vertical')
plt.show()


# for ham messages we can see that u,go, get,love like etc words are most occuring which we can relate to our personal and official messages
# 

# # MODEL BUILDING

# In[63]:


from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer()


# In[64]:


x = cv.fit_transform(data['Transformed_text']).toarray()


# In[65]:


x.shape


# In[66]:


y =data['Target'].values
y


# In[67]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[68]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[69]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[70]:


gnb =GaussianNB()
mnb =MultinomialNB()
bnb =BernoulliNB()


# In[71]:


gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[72]:


mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[73]:


bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[74]:


#model using tfidf to get more precise and accurate values


# In[75]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()


# In[76]:


x = tfidf.fit_transform(data['Transformed_text']).toarray()


# In[77]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[78]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb =GaussianNB()
mnb =MultinomialNB()
bnb =BernoulliNB()


# In[79]:


gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[80]:


mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[81]:


bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# ## here we can go with mnb or bnb 
#   since precision score matters motre than accuracy score 
#   we can go for Multinomial naive bayes

# In[ ]:




