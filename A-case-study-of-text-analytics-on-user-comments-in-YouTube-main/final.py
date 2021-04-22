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

# nltk.download("vader_lexicon")

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

df= pd.read_csv("/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY695/Final Project/final.csv",thousands=',', encoding='latin')
df= df.drop(["user","date", "timestamp",'likes'], axis=1)
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

##################################################################################################
#### Task A
text = []
for word in df['empty_list']:
    for key in word:
        text.append(key)

fdist = FreqDist()
for w in text:
    fdist[w] += 1     

fdist.most_common(10)

#lift score

pair = list(combinations(list(np.unique(brand)),2))

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
    ratio = lift_score(p,df.empty_list)
    lift_collection.append(ratio)

lift_score_df = pd.DataFrame(
    {'pair': pair,
     'lift_score': lift_collection
     })    


#MDS mapping
top_list = ['apple','huawei','xiaomi', 'samsung','oneplus','google','lg','sony','oppo',
            'honor','nokia','vivo', 'zte','problem']
top_lift_collection = []

top_pair = list(combinations(top_list,2))

for tp in top_pair:
    ratio = lift_score(tp,df.empty_list)
    top_lift_collection.append(ratio)

top_lift_score_df = pd.DataFrame(
    {'pair': top_pair,
     'lift_score': top_lift_collection
     })    

#MDS preworks
data = np.array([np.arange(14)]*14).T
df_mds = pd.DataFrame(data,index = top_list, columns = top_list)
df_mds.reset_index(level=0, inplace=True)

for z in top_pair:
    loc = top_pair.index(z)
    if top_lift_collection[loc] != 0:
        df_mds.set_value(z[0], z[1], 1/top_lift_collection[loc])
        df_mds.set_value(z[1],z[0],1/top_lift_collection[loc])
    elif top_lift_collection[loc] == 0:
        df_mds.set_value(z[0], z[1], 0)
        df_mds.set_value(z[1], z[0], 0)
        
df_mds = df_mds.drop(df_mds.columns[0], axis=1)
df_mds = df_mds.drop(df_mds.index[0:14])

df_mds.values[[np.arange(13)]*2] = 0

df_temp = df_mds.iloc[0,:]

df_mds.iloc[13,13] = 0

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
results = mds.fit(df_mds)

coords = results.embedding_

plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    coords[:, 0], coords[:, 1], marker = 'o'
    )
for label, x, y in zip(top_list, coords[:, 0], coords[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()

## Task B - Yuan Li

## Task C 
# Break down text with frequencies
words_df = pd.DataFrame(df.commentText.str.split(expand=True).stack().value_counts())
words_df['keys'] = words_df.index
words_df.columns = ['count', 'keys']
# Remove stop words
s = stopwords.words('english')
words_df_nostop = words_df[words_df['keys'].isin(s) == False]

###############################################################################################
## Top 5 words with their synonyms
#w1 = ['performance','engine','power','v8','acceleration','sports','torque','speed']
#w2 = ['luxury','premium','expensive','prestige']
#w3 = ['awd','4wd','quattro']
#w4 = ['interior','styling','design','interiors','exteriors']
#w5 = ['reliability','quality','reliable']
#
## Dummify top5 attributes
#performance_categ = []
#luxury_categ = []
#awd_categ = []
#design_categ = []
#reliability_categ = []
#
## Add 5 attributes picked to empty_list
#for i in range(df.shape[0]):
#    count1 = 0
#    count2 = 0
#    count3 = 0
#    count4 = 0
#    count5 = 0
#    for j in df.iloc[i,4]:
#        if j in w1:
#            count1 += 1
#        else:
#            count1 = count1
#        if j in w2:
#            count2 += 1
#        else:
#            count2 = count2
#        if j in w3:
#            count3 += 1
#        else:
#            count3 = count3
#        if j in w4:
#            count4 += 1
#        else:
#            count4 = count4
#        if j in w5:
#            count5 += 1
#        else:
#            count5 = count5
#    performance_categ.append(count1)
#    luxury_categ.append(count2)
#    awd_categ.append(count3)
#    design_categ.append(count4)
#    reliability_categ.append(count5)
#
#df['performance_categ'] = performance_categ
#df['luxury_categ'] = luxury_categ 
#df['awd_categ'] = awd_categ
#df['design_categ'] = design_categ 
#df['reliability_categ'] = reliability_categ 
#
#top_5_attr = ['performance','luxury','awd','design','reliability']
#top_5_brands = ['bmw','audi', 'lexus','infiniti','acura']
#
#for i in range(df.shape[0]):
#    for j in range(5):
#        if df.iloc[i,5+j] != 0:
#            df.iloc[i,3].append(top_5_attr[j])
#
## lift score
#pair_list_bmw = ['performance','luxury','awd','design','reliability','bmw']
#pair_list_audi = ['performance','luxury','awd','design','reliability','audi']
#pair_list_lexus = ['performance','luxury','awd','design','reliability','lexus']
#pair_list_infiniti = ['performance','luxury','awd','design','reliability','infiniti']
#pair_list_acura = ['performance','luxury','awd','design','reliability','acura']
#
#from itertools import combinations
#pair_bmw = list(combinations(list(np.unique(pair_list_bmw)),2))
#pair_audi = list(combinations(list(np.unique(pair_list_audi)),2))
#pair_lexus = list(combinations(list(np.unique(pair_list_lexus)),2))
#pair_infiniti = list(combinations(list(np.unique(pair_list_infiniti)),2))
#pair_acura = list(combinations(list(np.unique(pair_list_acura)),2))
#
#
#lift_collection_bmw = []
#for p in pair_bmw:
#    ratio = lift_score(p,df.empty_list)
#    lift_collection_bmw.append(ratio)
#lift_score_df_bmw = pd.DataFrame(
#    {'pair': pair_bmw,
#     'lift_score': lift_collection_bmw
#     }) 
#
#lift_collection_audi = []
#for p in pair_audi:
#    ratio = lift_score(p,df.empty_list)
#    lift_collection_audi.append(ratio)
#lift_score_df_audi = pd.DataFrame(
#    {'pair': pair_audi,
#     'lift_score': lift_collection_audi
#     }) 
#
#lift_collection_lexus = []
#for p in pair_lexus:
#    ratio = lift_score(p,df.empty_list)
#    lift_collection_lexus.append(ratio)
#lift_score_df_lexus = pd.DataFrame(
#    {'pair': pair_lexus,
#     'lift_score': lift_collection_lexus
#     }) 
#    
#lift_collection_infiniti = []
#for p in pair_infiniti:
#    ratio = lift_score(p,df.empty_list)
#    lift_collection_infiniti.append(ratio)
#lift_score_df_infiniti = pd.DataFrame(
#    {'pair': pair_infiniti,
#     'lift_score': lift_collection_infiniti
#     }) 
#    
#lift_collection_acura = []
#for p in pair_acura:
#    ratio = lift_score(p,df.empty_list)
#    lift_collection_acura.append(ratio)
#lift_score_df_acura = pd.DataFrame(
#    {'pair': pair_acura,
#     'lift_score': lift_collection_acura
#     }) 
#
#### BMW - Ultimate Driving Machine
#ultimate = ['ultimate','driving','machine','drive','excellent','machines','xdrive','steering','wheel','performance','sports']
#bmw_list = []
#for i in range(df.shape[0]):
#    count = 0
#    for j in df.iloc[i,4]:
#        if j in ultimate:
#            count += 1
#        else:
#            count = count
#    bmw_list.append(count)
#
#df['bmw_ultimate'] = bmw_list
#sum(bmw_list)
#
#for i in range(df.shape[0]):
#    if df.iloc[i,10] != 0:
#        df.iloc[i,3].append('ultimate driving machine')
#        
#pair4_list = ['ultimate driving machine','bmw']
#pair4 = list(combinations(list(np.unique(pair4_list)),2))
#
#lift_collection4 = []
#ratio = lift_score(pair4[0],df.empty_list)
#lift_collection4.append(ratio)
#
#lift_score_df4 = pd.DataFrame(
#    {'pair': pair4,
#     'lift_score': lift_collection4
#     }) 

###################################################
### Task E
#word_want = ['purchase','want','wish','hope','choose','prefer','like','love','pick','buy','dream']
#
#for want in range(len(df['tokenized_sents'])):
#    for wk in df.tokenized_sents.iloc[want]:
#        if wk in word_want:
#            df.wish_list.iloc[want].append('buy')
#
#for i in range(len(df['tokenized_sents'])):
#    for k in df.tokenized_sents.iloc[i]:
#        if k in model:
#            location = model.index(k)
#            df.wish_list.iloc[i].append(brand[location])
#        elif k in brand:
#            df.wish_list.iloc[i].append(k)
#
#df['wish_list'] = df.apply(lambda row: Remove(row['wish_list']), axis=1)
#
#top_list_wish = ['car','bmw','audi', 'lexus','infiniti','acura','toyota','sedan',
#            'honda','problem','mercedes benz','seat','nissan','ford']
#
#top_lift_collection_wish = []
#top_pair_wish = [['buy','bmw'],['buy', "audi"],["buy", "lexus"],["buy",'infiniti'],["buy",'acura'],["buy",'toyota'],["buy",'sedan'],["buy",'honda'],["buy",'mercedes benz'],["buy",'nissan'],["buy",'ford']]
#
#for tp in top_pair_wish:
#    ratio = lift_score(tp,df.wish_list)
#    top_lift_collection_wish.append(ratio)
#
#top_wish_lift_score_df = pd.DataFrame(
#    {'wish_pair': top_pair_wish,
#     'lift_score': top_lift_collection_wish
#     })    
#
brand_new = Remove(brand)

def model_iden(a):
    lis = [0] * len(brand_new)
    for i in a:
        if i in brand_new:
            k = brand_new.index(i)
            lis[k] = 1
    return lis

dummy_rows = []
for k in range(len(df.empty_list)):
    kk = model_iden(df.empty_list.iloc[k])
    dummy_rows.append(kk)

df_new = pd.DataFrame(dummy_rows,columns=brand_new)
df_new = pd.concat([df,df_new],axis=1)       

dic = {}
for i in brand_new:
    dic[i] = df_new.groupby(i)['sent_compound'].mean()[1]

