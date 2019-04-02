import numpy as np, pandas as pd
from textblob import TextBlob
import pickle
from scipy import spatial
import warnings
warnings.filterwarnings('ignore')
import spacy
en_nlp = spacy.load('en_core_web_sm')
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()

test = pd.read_excel(r'..\Data\Test_QA_Comprehension_Grade5.xlsx')

with open(r'..\Data\dict_test5_embeddings1.pickle', "rb") as f:
    d1 = pickle.load(f)
    
with open(r'..\Data\dict_test5_embeddings2.pickle', "rb") as f:
    d2 = pickle.load(f)
    
dict_emb_test = dict(d1)
dict_emb_test.update(d2)

len(dict_emb_test)

del d1, d2

#train.dropna(inplace=True)

test.shape

import re
def process_data(test):
    
    print("step 1")
    test['sentences'] = test['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])
    test['original_sentences'] = test['sentences']
    
    i=0
    for index, row in test.iterrows():
      test1=pd.DataFrame()
      test1['sentences']=row['sentences']
    
      test1['sentences'] = test1.apply(lambda x:re.sub(r'\s+', ' ', x["sentences"]),axis=1)
      
      test1['sentences'] = test1.apply(lambda x: x['sentences'].lower(), axis=1)

      test1['sentences'] = test1.apply(lambda x:re.sub(r'(\d+/\d+/\d+)|(\d+\.\d+\.\d+)|(\d+\-\d+\-\d+)|(\d+\/\d+)|(\d+th)|(\d+nd)|(\d+rd)|(\d+st)', ' DATE ', x["sentences"]),axis=1)
      test1['sentences'] = test1.apply(lambda x:re.sub(r'\b(mon|tue|wed|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b',' DATE ', x["sentences"]),axis=1)
      test1['sentences'] = test1.apply(lambda x:re.sub(r'(\$\d+\,\d+\.\d+)|(\$\d+\,\d+)|(\$\d+\.\d+)|(\$\d+)|(\$\ d+\,\d+\.\d+)|(\$ \d+\,\d+)|(\$ \d+\.\d+)|(\$ \d+)|(\d+\,\d+\.\d+)|(\d+\,\d+)|(\d+\.\d+)', ' AMOUNT ', x["sentences"]),axis=1)
      test1['sentences'] = test1.apply(lambda x:re.sub(r'(#\d+)|(# \d+)|(\d+)', ' NUMBER ', x["sentences"]),axis=1)
      test1['sentences'] = test1.apply(lambda x:re.sub(r'(\d+\.\d+)|(\d+)', ' AMOUNT ', x["sentences"]),axis=1)
      test1['sentences'] = test1.apply(lambda x:re.sub(r'[^\s]+@[^\s]+\.[^\s]+',' MAIL ', x["sentences"]),axis=1) 
      
      test1['sentences'] = test1.apply(lambda x:re.sub(r'\s+', ' ', x["sentences"]),axis=1)
      test1['sentences'] = test1.apply(lambda x:re.sub(r'(\()|(\))', '', x["sentences"]),axis=1)

      test1['sentences'] = test1.apply(lambda x:re.sub(r'[^a-zA-Z]', ' ', x["sentences"]),axis=1)
      test1['sentences'] = test1.apply(lambda x:re.sub(r'\s+', ' ', x["sentences"]),axis=1)
      
      test1['sentences'] = test1.apply(lambda x:re.sub(r'\.', '', x["sentences"]),axis=1)

      test1['sentences'] = test1.apply(lambda x: x['sentences'].lower(), axis=1)
      
      test1=test1['sentences']
      
      test.loc[i,'sentences']=list(test1)
      i=i+1

    
    #print("step 2")
    #train["target"] = train.apply(get_target, axis = 1)
    
    print("step 3")
    test['sent_emb'] = test['sentences'].apply(lambda x: [dict_emb_test[item][0] if item in\
                                                           dict_emb_test else np.zeros(4096) for item in x])
    print("step 4")
    test['quest_emb'] = test['questions'].apply(lambda x: dict_emb_test[x] if x in dict_emb_test else np.zeros(4096) )
        
    return test

test = process_data(test)

print(test.head(5))

#Predicted Cosine & Euclidean Index
def cosine_sim(x):
    li = []
    for item in x["sent_emb"]:
        li.append(spatial.distance.cosine(item,x["quest_emb"][0]))
    return li

def pred_idx(distances):
    return np.argmin(distances)

def predictions(train):
    
    train["cosine_sim"] = train.apply(cosine_sim, axis = 1)
    train["diff"] = (train["quest_emb"] - train["sent_emb"])**2
    train["euclidean_dis"] = train["diff"].apply(lambda x: list(np.sum(x, axis = 1)))
    del train["diff"]
    
    print("cosine start")
    
    train["pred_idx_cos"] = train["cosine_sim"].apply(lambda x: pred_idx(x))
    train["pred_idx_euc"] = train["euclidean_dis"].apply(lambda x: pred_idx(x))
    
    return train

predicted = predictions(test)

predicted["cosine_sim"][0]

predicted["euclidean_dis"][0]

ques=[]
answer_cos=[]
answer_euc=[]
cosine=[]
euc=[]
for i in range(len(predicted)):
  answer_cos.append(predicted.loc[i,'original_sentences'][predicted.loc[i,'pred_idx_cos']])
  answer_euc.append(predicted.loc[i,'original_sentences'][predicted.loc[i,'pred_idx_euc']])
  ques.append(predicted.loc[i,'questions'])
  cosine.append(predicted.loc[i,'cosine_sim'][predicted.loc[i,'pred_idx_cos']])
  euc.append(predicted.loc[i,'euclidean_dis'][predicted.loc[i,'pred_idx_euc']])
  
df=pd.DataFrame()
df['Question']=ques
df['Answer_Cos']=answer_cos
df['Cosine Sim']=cosine
df['Answer_Euc']=answer_euc
df['Euclidean Dis']=euc

df.to_csv(r'..\Results\Grade5_Answers.csv')