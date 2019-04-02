import warnings
warnings.filterwarnings('ignore')
import pickle
import pandas as pd
from textblob import TextBlob
import torch

valid = pd.read_excel("../Data/Test_QA_Comprehension_Grade5.xlsx")

valid.shape

valid.head(6)

i=0
for index, rows in valid.iterrows():
    valid.loc[i,'context']=re.sub(r'\n', ' ', rows['context'])
    i=i+1

#Creating dictionary for sentence embeddings for faster computation
paras = list(valid["context"].drop_duplicates().reset_index(drop= True))

len(paras)

blob = TextBlob(" ".join(paras))
sentences = [item.raw for item in blob.sentences]

len(sentences)

from models import InferSent

GLOVE_PATH = '../Data/glove.840B.300d.txt'
MODEL_PATH = '../InferSent/encoder/infersent1.pickle'
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
model.set_w2v_path(GLOVE_PATH)

model.build_vocab(sentences, tokenize=True)

sent=sentences.copy()
sentences=pd.DataFrame()
sentences['cms_notes']=sent

import re
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'\s+', ' ', x["cms_notes"]),axis=1)

sentences['cms_notes'] = sentences.apply(lambda x: x['cms_notes'].lower(), axis=1)

#Replacing numbers
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'(\d+/\d+/\d+)|(\d+\.\d+\.\d+)|(\d+\-\d+\-\d+)|(\d+\/\d+)|(\d+th)|(\d+nd)|(\d+rd)|(\d+st)', ' DATE ', x["cms_notes"]),axis=1)
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'\b(mon|tue|wed|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b',' DATE ', x["cms_notes"]),axis=1)
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'(\$\d+\,\d+\.\d+)|(\$\d+\,\d+)|(\$\d+\.\d+)|(\$\d+)|(\$\ d+\,\d+\.\d+)|(\$ \d+\,\d+)|(\$ \d+\.\d+)|(\$ \d+)|(\d+\,\d+\.\d+)|(\d+\,\d+)|(\d+\.\d+)', ' AMOUNT ', x["cms_notes"]),axis=1)
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'(#\d+)|(# \d+)|(\d+)', ' NUMBER ', x["cms_notes"]),axis=1)
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'(\d+\.\d+)|(\d+)', ' AMOUNT ', x["cms_notes"]),axis=1)
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'[^\s]+@[^\s]+\.[^\s]+',' MAIL ', x["cms_notes"]),axis=1)

#Removing remaining numbers and spaces
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'\s+', ' ', x["cms_notes"]),axis=1)
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'(\()|(\))', '', x["cms_notes"]),axis=1)

#Remove punctuations
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'[^a-zA-Z]', ' ', x["cms_notes"]),axis=1)
sentences['cms_notes'] = sentences.apply(lambda x:re.sub(r'\s+', ' ', x["cms_notes"]),axis=1)

sentences['cms_notes'] = sentences.apply(lambda x: x['cms_notes'].lower(), axis=1)

sentences=sentences['cms_notes']
sent=[]
for i in sentences:
    sent.append(i)

sentences=sent

dict_embeddings = {}
for i in range(len(sentences)):
    print(i)
    dict_embeddings[sentences[i]] = model.encode([sentences[i]], tokenize=True)
    
questions = list(valid["questions"])

len(questions)

for i in range(len(questions)):
    print(i)
    dict_embeddings[questions[i]] = model.encode([questions[i]], tokenize=True)

d1 = {key:dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 0}
d2 = {key:dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 1}


with open('../Data/dict_test5_embeddings1.pickle', 'wb') as handle:
    pickle.dump(d1, handle)

with open('../Data/dict_test5_embeddings2.pickle', 'wb') as handle:
    pickle.dump(d2, handle)

del dict_embeddings