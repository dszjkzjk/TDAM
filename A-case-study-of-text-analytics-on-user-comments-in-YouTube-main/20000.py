#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:49:10 2019

@author: junkangzhang
"""
import pandas as pd
import numpy as np
import nltk
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist
import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn import manifold
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from itertools import combinations 

def get_sentiment(rating_data):
    """
    https: // github.com / cjhutto / vaderSentiment
    :return:
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    rating_data['sent_compound'] = -10
    rating_data['sent_pos'] = -10
    rating_data['sent_neg'] = -10
    rating_data['sent_neu'] = -10
    for i in range(len(rating_data)):
        sentence = rating_data['commentText'][i]
        ss = sid.polarity_scores(sentence)
        rating_data.iloc[i, 2] = float(ss.get('compound'))
        rating_data.iloc[i, 3] = float(ss.get('pos'))
        rating_data.iloc[i, 4] = float(ss.get('neg'))
        rating_data.iloc[i, 5] = float(ss.get('neu'))
    return rating_data

df = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY695/Final Project/20000.csv',thousands=',', encoding='latin')
#df= df.drop(["user","date", "timestamp",'likes','hasReplies','numberOfReplies'], axis=1)
df = df.drop('date',axis=1)
df['commentText'] = df['commentText'].str.replace('ï','')
df['commentText'] = df['commentText'].str.replace('¿','')
df['commentText'] = df['commentText'].str.replace('½','')
df = df.rename(columns={ df.columns[1]: "commentText" })

df = get_sentiment(df)
#df.to_csv("C:/Users/ruite/OneDrive - McGill University/2018 MMA/Text analysis/final/sentiment_data.csv", index = False)
models = pd.read_csv("/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY695/Final Project/model.csv", encoding = 'latin')

df['commentText'] = df['commentText'].str.lower()
df['empty_list'] = [list() for x in range(len(df.index))]
df['wish_list'] = [list() for x in range(len(df.index))]

models['brand'] = models['brand'].str.lower()
models['model'] = models['model'].str.lower()

df['commentText'] = df['commentText'].str.replace('ï','')
df['commentText'] = df['commentText'].str.replace('¿','')
df['commentText'] = df['commentText'].str.replace('½','')

#models['brand'] = models['brand'].str.replace(',','')
#models['brand'] = models['brand'].str.replace('.','')   
#df.drop(df.index[[1653,2235,4247,4970,4974,5033]], inplace=True)

tokenizer = RegexpTokenizer(r'\w+')
wnl = nltk.WordNetLemmatizer()

#df['commentText'] = df.apply(lambda row: wnl.lemmatize(row['commentText']), axis=1)

df['tokenized_sents'] = df.apply(lambda row: tokenizer.tokenize(row['commentText']), axis=1)
#lowercase  

model = []
brand = []
for b in range(len(models['model'])):
    brand.append(models.brand[b])
for m in range(len(models['model'])):
    model.append(models.model[m])
                       
for i in range(len(df['tokenized_sents'])):
    for k in df.tokenized_sents.iloc[i]:
        if k in model:
            location = model.index(k)
            df.empty_list.iloc[i].append(brand[location])
        elif k in brand:
            df.empty_list.iloc[i].append(k)

def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

df['empty_list'] = df.apply(lambda row: Remove(row['empty_list']), axis=1)

text = []
for word in df['empty_list']:
    for key in word:
        text.append(key)

fdist = FreqDist()
for w in text:
    fdist[w] += 1     

fdist.most_common(10)

kk = 0
for i in range(len(df)):
    if df.empty_list[i] == ['samsung']:
        kk += 1
print(str(kk) + ' huawei')


lst = []
for i in range(len(df)):
    if df.empty_list[i] == ['apple'] or df.empty_list[i] == ['huawei'] or df.empty_list[i] == ['samsung']:
        lst.append(i)
df_new = df.iloc[lst,:]

#df_new['empty_list'] = [k[0] for k in df_new['empty_list']]
#avg = df_new.groupby('empty_list')['sent_compound'].mean()


attr_list = ['battery','camera','weight','resolution','video','processor',
             'ram','storage','fingerprint','face','quality','speaker','durability','cpu','gpu','best','bad']

#for i in attr_list:
    #df_new[i] = [None] * len(df_new)
    
d = {}
for i in attr_list:
    print(i)
    d["dummy_" + str(i)] = []
    for j in range(len(df_new)):
        if i in df_new.tokenized_sents.iloc[j]:
            print(1)
            d["dummy_" + str(i)].append(1)
        else:
            d["dummy_" + str(i)].append(0)


for i in d.keys():
    df_new[i] = d[i]


#df_new.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY695/Final Project/df_new.csv')

########### lift score


pair = []


for i in range(len(attr_list)):
    for j in range(len(df_new)):
        if df_new.iloc[j,i+9] == 1:
            df_new['empty_list'].iloc[j].append(attr_list[i])

def lift_score(test_pair, test_col):
    count_1 = 0
    count_2 = 0
    all_count = 0
    for i in test_col:
        if test_pair[0] in i and test_pair[1] in i:
            all_count += 1
            count_1 += 1
            count_2 += 1
        elif test_pair[0] in i:
            count_1 += 1
        elif test_pair[1] in i:
            count_2 += 1
    lift = all_count/((count_1 * count_2) + 1)
    return lift*len(test_col)

lift_collection = []
"""
for p in pair:
    ratio = lift_score(p,df.empty_list)
    lift_collection.append(ratio)

lift_score_df = pd.DataFrame(
    {'pair': pair,
     'lift_score': lift_collection
     }) 
"""