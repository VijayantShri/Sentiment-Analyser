#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


nltk.download('twitter_samples')


# In[3]:


# Tokenizing the data
from nltk.corpus import twitter_samples as ts


# In[4]:


positive_tweets = ts.strings('positive_tweets.json')
negative_tweets = ts.strings('negative_tweets.json')
text = ts.strings('tweets.20150430-223406.json')


# In[5]:


nltk.download('punkt')


# In[6]:


tweet_tokens = ts.tokenized('positive_tweets.json')


# In[7]:


tweet_tokens[0][0]


# In[8]:


# Normalizing the data.
nltk.download('wordnet')


# In[9]:


nltk.download('averaged_perceptron_tagger')


# In[10]:


from nltk.tag import pos_tag


# In[11]:


print(pos_tag(tweet_tokens[0]))


# In[12]:


from nltk.stem.wordnet import WordNetLemmatizer


# In[13]:


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatize_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos='v'
        else:
            pos = 'a'
        lemmatize_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatize_sentence


# In[14]:


lemmatize_sentence(tweet_tokens[0])


# In[15]:


# Removing Noise from the data.

import re, string


# In[16]:


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith("VB"):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        
        if len(token)>0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# In[17]:


nltk.download('stopwords')


# In[18]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[19]:


remove_noise(tweet_tokens[0], stop_words)


# In[20]:


positive_tweets_tokens = ts.tokenized('positive_tweets.json')
negative_tweets_tokens = ts.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = [remove_noise(tokens, stop_words) for tokens in positive_tweets_tokens]
negative_cleaned_tokens_list = [remove_noise(tokens, stop_words) for tokens in negative_tweets_tokens]


# In[21]:


print(positive_tweets_tokens[500])
print(positive_cleaned_tokens_list[500])


# In[22]:


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


# In[23]:


all_pos_words = get_all_words(positive_cleaned_tokens_list)


# In[24]:


from nltk import FreqDist


# In[25]:


freq_dist_pos = FreqDist(all_pos_words)


# In[26]:


freq_dist_pos.most_common(10)


# In[27]:


# Preparing Data for the Model:

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


# In[28]:


positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)


# In[29]:


import random


# In[30]:


positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]
dataset = positive_dataset + negative_dataset
random.shuffle(dataset)
train_data = dataset[:7000]
test_data = dataset[7000:]


# In[31]:


# Building and Testing the Model:


# In[32]:


from nltk import classify
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)


# In[33]:


print("Accuracy is: ", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(10))


# In[34]:


from nltk.tokenize import word_tokenize


# In[35]:


cutom_tweet = 'I ordered just once from TerribleCo, they screwed up, never used the app again.'
custom_tokens = remove_noise(word_tokenize(cutom_tweet))


# In[36]:


classifier.classify(dict([token, True] for token in custom_tokens))


# In[37]:


custom_tweets = "Congrats #SportStar on your 7th best goal last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies"


# In[38]:


custom_tokens = remove_noise(word_tokenize(custom_tweets))


# In[39]:


classifier.classify(dict([token, True] for token in custom_tokens))


# In[40]:


custom_tweet = 'Thank you for sending my baggage to CityX and flying me to CityY at the same time. Brilliant service. #thanksGenericAirline'


# In[41]:


custom_token = remove_noise(word_tokenize(custom_tweet))
x = classifier.classify(dict([token, True] for token in custom_token))


# In[42]:


from tkinter import *
from tkinter import messagebox


# In[45]:


def button(x):
    custom_tweet = e.get()
    if custom_tweet=="":
        var = "Please write Tweet..."
    else:
        custom_token = remove_noise(word_tokenize(custom_tweet))
        x = classifier.classify(dict([token, True] for token in custom_token))
        var = "Sentiment is "+x
    messagebox.showinfo("Sentiment Result", var)


if __name__=="__main__":
    tk = Tk()

    tk.geometry("900x400")
    tk.configure(bg='blue')
    tk.title("Sentiment Analyser")
    labelFrame = LabelFrame(tk, text="Tweet Entry Window: ", font=15, bg='pink', bd=4, height=300, relief=RAISED)
    labelFrame.pack(fill=X, anchor=N, padx=50, pady=60, ipady=10)

    l1 = Label(labelFrame, text="Enter you Tweet:", font=25, relief=RAISED)
    l1.pack(anchor=NW, padx=70, pady=15, ipadx=10 )

    e = Entry(labelFrame, width=80, font=40, relief=RAISED, xscrollcommand=True)
    e.pack(anchor=NW, ipady=10, padx=100, pady=10)

    b1 = Button(labelFrame, text="Sentiment", bg='skyblue', cursor='tcross', font=20, command=lambda:button(x))
    b1.pack(anchor=N, expand="yes", ipady=5, ipadx=7, pady=20)

    tk.resizable(0, 0)
    tk.mainloop()


# In[ ]:




