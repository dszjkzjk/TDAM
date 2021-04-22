# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 16:00:51 2019

@author: ruite/junkangzhang
"""

# the input filename is limit_post.csv in line 30, change it as needed
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

df_top3 = pd.read_csv("/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY695/Final Project/top_3.csv",thousands=',', encoding='utf-8')

df_top3= df_top3.drop(["sent_compound","sent_pos", "sent_neg",'sent_neu','empty_list','wish_list','tokenized_sents'], axis=1)

models = pd.read_csv("/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY695/Final Project/model.csv", encoding = 'latin')

df_top3['empty_list'] = [list() for x in range(len(df_top3.index))]
df_top3['brand_list'] = [list() for x in range(len(df_top3.index))]
df_top3['commentText'] = df_top3['commentText'].str.replace('ï','')
df_top3['commentText'] = df_top3['commentText'].str.replace('¿','')
df_top3['commentText'] = df_top3['commentText'].str.replace('½','')
df_top3 = df_top3.rename(columns={ df_top3.columns[1]: "commentText" })
df_top3['commentText'] = df_top3['commentText'].str.lower()


models['brand'] = models['brand'].str.lower()
models['model'] = models['model'].str.lower()


tokenizer = RegexpTokenizer(r'\w+')
#wnl = nltk.WordNetLemmatizer()

#df_top3['commentText'] = df_top3.apply(lambda row: wnl.lemmatize(row['commentText']), axis=1)

df_top3['tokenized_sents'] = df_top3.apply(lambda row: tokenizer.tokenize(row['commentText']), axis=1)

#lowercase  
model = []
brand = []
for b in range(len(models['model'])):
    brand.append(models.brand[b])
for m in range(len(models['model'])):
    model.append(models.model[m])
                       
for i in range(len(df_top3['tokenized_sents'])):
    for k in df_top3.tokenized_sents.iloc[i]:
        if k in model:
            df_top3.empty_list.iloc[i].append(k)
#        elif k in brand:
#            df_top3.empty_list.iloc[i].append(k)

def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

df_top3['empty_list'] = df_top3.apply(lambda row: Remove(row['empty_list']), axis=1)

text = []
for word in df_top3['empty_list']:
    for key in word:
        text.append(key)

fdist = FreqDist()
for w in text:
    fdist[w] += 1     

fdist.most_common(10)

####################################################################################
top_list = ['note', 'iphone','mate','galaxy']
# check detailed model
for i in range(len(df_top3['tokenized_sents'])):
        if not set(top_list).isdisjoint(df_top3.tokenized_sents.iloc[i]):
            for top in top_list:
                if top in df_top3.tokenized_sents.iloc[i]:
                    location_1 = df_top3.tokenized_sents.iloc[i].index(top)
                    location_2 = location_1 + 1
                    if location_2 < len(df_top3.tokenized_sents.iloc[i]):
                        mod = top + str(df_top3.tokenized_sents.iloc[i][location_2])
                        if mod in model:
                            df_top3.brand_list.iloc[i].append(mod)
                    
            
#########################################################################################
#merge empyty_list and brand list
filter_list = ['s9','p20','mate20','a9','note9']

for i in range(len(df_top3)):
    for k in df_top3.empty_list.iloc[i]:
        if k in filter_list:
            df_top3.brand_list.iloc[i].append(k)
            
def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

df_top3['brand_list'] = df_top3.apply(lambda row: Remove(row['brand_list']), axis=1)

text_2 = []
for word in df_top3['brand_list']:
    for key in word:
        text_2.append(key)

fdist_2 = FreqDist()
for w in text_2:
    fdist_2[w] += 1     

fdist_2.most_common(7)    



########### lift score
df_new = pd.read_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY695/Final Project/df_new.csv',thousands=',', encoding='latin')
df_new['model_list'] = df_top3['brand_list']

attr_list = ['battery','camera','weight','resolution','video','processor',
             'ram','storage','fingerprint','face','quality','speaker','durability','cpu','gpu','best','bad']

for i in range(len(attr_list)):
    for j in range(len(df_new)):
        if df_new.iloc[j,i+9] == 1:
            df_new['model_list'].iloc[j].append(attr_list[i])
            
model_listo = list(fdist.keys())

pair = []
for i in model_listo:
    for j in attr_list:
        pair.append((i,j))

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


for p in pair:
    ratio = lift_score(p,df_new.model_list)
    lift_collection.append(ratio)

lift_score_df = pd.DataFrame(
    {'pair': pair,
     'lift_score': lift_collection
     }) 
    
df_new = df_new.drop('wish_list',axis = 1)
df_new.to_csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY695/Final Project/destination.csv')