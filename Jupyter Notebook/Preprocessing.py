#!/usr/bin/env python
# coding: utf-8

# #### Prerocessing File for (products, restaurants, movies) Datasets.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# for text cleaning
import re
import string
from nltk.corpus import stopwords
import os
import warnings
warnings.simplefilter("ignore")
import nltk


# In[203]:


pd.set_option('display.max_colwidth', None)


# In[26]:


df = pd.read_csv("C:/Users/raimn/mlproject/Res.csv")
df.head()


# In[27]:


df.iloc[0:50]


# In[28]:


df.info()


# In[29]:


df['polarity'].value_counts().plot(kind='bar');


# In[17]:


df['polarity'].value_counts()


# In[18]:


df.duplicated().sum()


# In[19]:


df.drop_duplicates(inplace=True)


# In[20]:


df.isnull().sum()


# In[30]:


arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ»«•'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations


# In[31]:


def strip_tags_and_punctuations(text):
    mention_and_hashtag_prefixes = ['@','#']

    # replace all punctuations except the above with space
    for separator in  punctuations_list:
        if separator not in mention_and_hashtag_prefixes:
            text = text.replace(separator,' ')

    # remove mentions and hashtags
    words = []
    for word in text.split():
        word = word.strip()
        if len(word) != 1:
            if word[0] not in mention_and_hashtag_prefixes and word[1] not in mention_and_hashtag_prefixes: 
                words.append(word)
        else:
            if word[0] not in mention_and_hashtag_prefixes:
                words.append(word)
    return ' '.join(words)


# In[39]:


# Check
print('Before:', df['text'][30])
print('After:', strip_tags_and_punctuations(df['text'][30]))


# In[215]:


# Apply
df['text'] = df['text'].apply(strip_tags_and_punctuations)


# In[34]:


# All Diacritics
arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


# In[35]:


# Check 
print('Before:', df['text'][0])
print('After:', re.sub(arabic_diacritics, '', df['text'][0]))


# In[36]:


# Aplply
df['text'] = df['text'].apply(lambda text: re.sub(arabic_diacritics, '', text))


# In[219]:


# helpful function
def normalize_text(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("ڤ", "ف", text)
    return text


# In[220]:


# Check 
print('Before:', df['text'][33])
print('After:', normalize_text(df['text'][33]))


# In[221]:


df['text'] = df['text'].apply(normalize_text)


# In[222]:



def normalize_text(text):
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    return text


# In[223]:



print('Before:', df['text'][3])
print('After:', normalize_text(df['text'][3]))


# In[224]:


df['text'] = df['text'].apply(normalize_text)


# In[37]:


def normalize_numbers(text):
    text = re.sub("٠", "0", text)
    text = re.sub("١", "1", text)
    text = re.sub("٢", "2", text)
    text = re.sub("٣", "3", text)
    text = re.sub("٤", "4", text)
    text = re.sub("٥", "5", text)
    text = re.sub("٦", "6", text)
    text = re.sub("٧", "7", text)
    text = re.sub("٨", "8", text)
    text = re.sub("٩", "9", text)
    return text


# In[38]:


print('Before:', df['text'][0])
print('After:', re.sub("\d+", "",df['text'][0]))


# In[227]:


df['text'] = df['text'].apply(normalize_numbers)


# In[40]:



print('Before:', df['text'][30])
print('After:', re.sub(r'\s*[A-Za-z]+\b', '', df['text'][30]))


# In[229]:


df['text'] = df['text'].apply(lambda text: re.sub(r'\s*[A-Za-z]+\b', '', text))


# In[230]:


def remove_emojis(text):
    emojis_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emojis_pattern.sub(r'', text)


# In[231]:


# Check
print('Before:', df['text'][10])
print('After:', remove_emojis(df['text'][10]))


# In[232]:


df['text'] = df['text'].apply(remove_emojis)


# In[233]:


df['text'] = df['text'].apply(lambda text: text.strip())


# In[234]:


df['text'] = df['text'].str.replace('\d+', '')


# In[235]:


df.head()


# In[77]:


import emoji
#Stats about Text
def avg_word(sentence):
    words = sentence.split()
    if len(words) == 0:
        return 0
    return (sum(len(word) for word in words)/len(words))

def emoji_counter(sentence):
    return emoji.emoji_count(sentence)

df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))
df['char_count'] = df['text'].str.len() ## this also includes spaces
df['avg_char_per_word'] = df['text'].apply(lambda x: avg_word(x))
stop = stopwords.words('arabic')
df['stopwords'] = df['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
df['emoji_count'] = df['text'].apply(lambda x: emoji_counter(x))
df = df.sort_values(by='word_count',ascending=[0])
df.head()


# In[236]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = nltk.corpus.stopwords.words("arabic")

def remove_stopwords(text):
    
    text_tokenized = word_tokenize(text)
    text_no_stop = [word for word in text_tokenized if word not in stop_words]
    
    return " ".join(text_no_stop)


# In[237]:


# Testing remove_stopwords function
print('Before:', df['text'][10])
print('After:', remove_stopwords(df['text'][10]))


# In[238]:


df['text'] = df['text'].apply(remove_stopwords)


# In[239]:


df.to_csv('C:/Users/raimn/mlproject/ResNew1.csv', index=False)


# In[240]:


df22 = pd.read_csv("C:/Users/raimn/mlproject/ResNew1.csv")


# In[241]:


df22.head()


# In[242]:


df22 = df22.dropna()
df22 = df22.reset_index(drop=True)


# In[243]:


df22.info()


# In[244]:


df22['polarity'].value_counts()


# In[254]:


df22.iloc[0:50]


# In[255]:


def removeUnnecessarySpaces(text):
    return re.sub(r'[\n\t\ ]+', ' ', text)

def removeNonArabicChar(text):
    return re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD.0-9]+', ' ', text)

def sentTokenize(text):
    return text.replace(".", ". \n- ")

def clean(text):
    text = removeUnnecessarySpaces(text)
    text = removeNonArabicChar(text)
    text = removeUnnecessarySpaces(text)
    return sentTokenize(text)


# In[258]:



print('Before:', df22['text'][36])
print('After:', clean(df22['text'][36]))


# In[259]:


df22['text'] = df22['text'].apply(clean)


# In[261]:


df22 = df22.dropna()
df22 = df22.reset_index(drop=True)


# In[276]:


nan_value = float("NaN")
df23.replace(" ", nan_value, inplace=True)
df23.dropna(subset = ["text"], inplace=True)


# In[278]:


df23.to_csv('C:/Users/raimn/mlproject/ResNew1.csv', index=False)


# In[279]:


df24 = pd.read_csv("C:/Users/raimn/mlproject/ResNew1.csv")


# In[280]:


df24.info()


# In[281]:


df24.iloc[0:50]


# In[282]:


df24['polarity'].value_counts()

