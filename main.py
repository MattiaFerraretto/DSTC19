import binascii
import sys

import nltk
import pandas as pd
from twarc import Twarc
import json
import psycopg2
#from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig
from transformers import pipeline
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import emoji
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import unicodedata
import numpy as np
import textacy.preprocessing.replace as rp
import textacy.preprocessing.normalize as nr
from sqlalchemy import create_engine
import csv
import random as rnd

tweet_model = {
    "created_at": "",
    "id": -1,
    "text": "",
    "entities": {
        "hashtags": [],
        "user_mentions": []
    },
    "user": {
        "id": -1,
        "name": "",
        "screen_name": "",
        "location": "",
        "description": "",
        "url": "",
        "created_at": "",
        "verified": ""
    },
    "coordinates": {
        "type": "",
        "coordinates": []   #array di float [longitude, latitude]
    },
    "place": {
        "place_type": "",
        "name": "",
        "full_name": "",
        "country_code": "",
        "country": ""
    },
    "retweet_count": -1,
    "lang": ""
}

def get_twarc():
    api_key = "WVGPuX3XizTgFTzpJo7JFMRfy";
    api_secret_key = "1JwqtgpglZzpE2pJxpBhfxvV05gQ60P6tLBRwL4rd9Rr6eFS0B";
    acces_token = "958719482725314560-T1QGudnbM6wNTJIGGadYe7d8q4DwX2i";
    acces_token_secret = "HQR5PAfWYUGICIMOPgt6OAsVILM6vIt3kBfBEiySqe7tn";

    return Twarc(api_key, api_secret_key, acces_token, acces_token_secret);

def get_db_connection(host):
    conn = psycopg2.connect(dbname='tweets_dataset', user='mferraretto', password='qwerty', host=host)

    return conn

def filter():

    QUERY = "SELECT tweet_id FROM tweets WHERE country_place = 'IT' AND lang = 'it' AND _date BETWEEN '2020-03-01' AND '2020-12-31' ORDER BY _date ASC"
    db = get_db_connection('192.168.182.188')
    curs = db.cursor()
    print("Query execution...")
    curs.execute(QUERY)
    tweets_id = [record[0] for record in curs]
    db.close()
    print("Query executed.")
    print("Write records...")
    pd.DataFrame(tweets_id).to_csv('tweets_id_ita_IT.csv', index=False, header=None)
    print("Done.")

def hydrate_tweets(out_f_name):
    twrc = get_twarc()

    print("Hydrating tweets...")
    with open(out_f_name, "w") as out:
        for tweet in twrc.hydrate(open('tweets_id_ita_IT.csv')):
            tweet_dict = dict(tweet_model)
            tweet_dict['created_at'] = tweet['created_at']
            tweet_dict['id'] = tweet['id']
            tweet_dict['text'] = tweet['full_text']
            tweet_dict['entities']['hashtags'] = [dict({'text': hastag['text']}) for hastag in tweet['entities']['hashtags']]
            tweet_dict['entities']['user_mentions'] = [dict({'screen_name': user_mention['screen_name'], 'name': user_mention['name'], 'id': user_mention['id']}) for user_mention in tweet['entities']['user_mentions']]
            tweet_dict['user']['id'] = tweet['user']['id']
            tweet_dict['user']['name'] = tweet['user']['name']
            tweet_dict['user']['screen_name'] = tweet['user']['screen_name']
            tweet_dict['user']['location'] = tweet['user']['location']
            tweet_dict['user']['description'] = tweet['user']['description']
            tweet_dict['user']['url'] = tweet['user']['url']
            tweet_dict['user']['created_at'] = tweet['user']['created_at']
            tweet_dict['user']['verified'] = tweet['user']['verified']

            if tweet['coordinates']:
                tweet_dict['coordinates']['type'] = tweet['coordinates']['type']
                tweet_dict['coordinates']['coordinates'] = tweet['coordinates']['coordinates']
            else:
                tweet_dict.update({'coordinates': None})

            if tweet['place']:
                tweet_dict['place']['place_type'] = tweet['place']['place_type']
                tweet_dict['place']['name'] = tweet['place']['name']
                tweet_dict['place']['full_name'] = tweet['place']['full_name']
                tweet_dict['place']['country_code'] = tweet['place']['country_code']
                tweet_dict['place']['country'] = tweet['place']['country']
            else:
                tweet_dict.update({'place': None})

            tweet_dict['retweet_count'] = tweet['retweet_count']
            tweet_dict['lang'] = tweet['lang']
            json.dump(tweet_dict, out)
            out.write("\n")

    print("Hydrated tweets.")

def count_rows():
    dt = pd.read_csv('tweets_id_ita_IT(2).csv', header=None)
    print(sum(1 for row in dt[0]))

def cleaner(tweet_text):

    #tweet_text = tweet_text.lower()
    tweet_text = nr.unicode(tweet_text, form="NFC")
    tweet_text = re.sub(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', "emailid", tweet_text)
    tweet_text = re.sub(r"[a-zA-Z]{2}[0-9]{2}[a-zA-Z0-9]{4}[0-9]{7}([a-zA-Z0-9]?){0,16}", "ibanid", tweet_text)
    tweet_text = rp.user_handles(tweet_text, "usermention")
    tweet_text = re.sub(r"\B#\w*[a-zA-Z]+\w*", "hashtag", tweet_text, 0, re.MULTILINE)
    #tweet_text = rp.hashtags(tweet_text, "hashtag")
    #tweet_text = re.sub(r"@[\S]+", "usermention", tweet_text)
    #tweet_text = re.sub(r"#[\S]+", "hashtag", tweet_text)
    tweet_text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "url", tweet_text)
    tweet_text = re.sub(r'(\(?\d{2,4}\)?\D{0,3}\d{6,10}|\+\d{10,12})', "phonenumber", tweet_text)
    #tweet_text = re.sub(r"(\+\d{3,5}[-\.\s]??\d{2}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{2,4}[-\.\s\/]??\d{5,8})", "phonenumber", tweet_text)
    #tweet_text = re.sub(r"(\+\d{3,5}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3,4}[-\.\s]??\d{4,7})", "phonenumber", tweet_text)
    #tweet_text = re.sub(r'(\(?\d{2,3}\)?\D{0,3}\d{6,10}|\+\d{10,12})', "phonenumber", tweet_text)
    tweet_text = nr.quotation_marks(tweet_text)
    tokens = re.split(r'\s+',tweet_text)
    #regx = r"\w|!|%|'|\(|\)|,|\.|:|;|\?|_|`|/|-|\U00002019|\"|€"
    regx = r"\w|!|%|'|\(|\)|,|\.|:|;|\?|_|`|/|-|\"|€"
    #regx = r"\w|!|#|\$|%|&|'|\(|\)|\*|\+|,|-|\.|/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|\||}|~"
    words = [''.join(re.findall(regx, word)) for word in tokens if re.findall(regx, word)]
    #stop_word = set(stopwords.words('italian'))
    #words = [word for word in words if word not in stop_word]
    return ' '.join(words)


def clear_tweets(infile):
    print('Cleaning up tweets...')
    tweets = [json.loads(line) for line in open(infile, "r")]

    ids = []
    text_cleaned = []
    for tweet in tweets:
        ids.append(tweet['id'])
        text_cleaned.append(cleaner(tweet['full_text']))
        #print(str(i) +" "+ str(tweet['id'])+ " "+nltk_cleaner(tweet['full_text']))
    df = pd.DataFrame(data=dict({'tweets_ids': ids, 'cleaned_text': text_cleaned}))
    df.to_csv('dataset_cleaned.csv', index=None, header=None)
    #nltk_cleaner(tweets[2]['full_text'])
    print('Cleaned tweets.')


def create_sample_file(inf):

    df = pd.read_csv(inf, header=None)
    text = df[1]

    with open('sample_file.csv', 'w') as sf:
        sf.write("SENTENCE\n")
        for line in text:
            line = re.sub("\"", "\"\"", line, 0, re.MULTILINE)
            sf.write('"'+line+'"\n')

def load_italian_bert_model():
    model_name = "dbmdz/bert-base-italian-cased"
    config = BertConfig.from_pretrained('bert-base-italian-cased/config.json')
    #model_name = 'bert-base-italian-cased/'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    return  tokenizer, model

def generate_baseline_dataset(dataset_path, n_samples):

    '''mask = np.zeros(len(dataset), dtype=bool)
    mask[np.random.randint(len(dataset), size=n_samples)] = True

    baseline_dataset = dataset[mask]
    raw_dataset = dataset[~mask]

    print(baseline_dataset)
    print(raw_dataset)'''
    print("Baseline dataset generation...")
    dataset = [json.loads(line) for line in open(dataset_path, "r")]
    len_new_ds = len(dataset)-n_samples
    indexs = np.random.randint(len(dataset), size=n_samples)
    baseline_dataset = [dataset[index] for index in indexs]
    dataset = [dataset[index] for index in range(len(dataset)) if index not in indexs]

    with open("baseline_dataset("+str(n_samples)+").json", 'w') as blds:
        for tweet in baseline_dataset:
            json.dump(tweet, blds)
            blds.write("\n")

    with open("dataset(" + str(len_new_ds) + ").json", 'w') as ds:
        for tweet in dataset:
            json.dump(tweet, ds)
            ds.write("\n")

    print("Baseline dataset generated.")

def gnr_blds(dataset_path, n_samples):
    df = pd.read_csv(dataset_path, header=None, names=['tid','text'])
    df1 = pd.read_csv('blds_l.csv')[['tid']]
    df = df[~df['tid'].isin(df1.loc[:, 'tid'])]

    blds = df.sample(n=n_samples)
    ds = df.drop(blds.index)
    blds.to_csv('baseline_dataset.csv', index=None, header=['TID', 'SENTENCE'], quoting=csv.QUOTE_NONNUMERIC)
    ds.to_csv('dataset.csv',index=None, header=None)
    #ds.sample(n=500).to_csv('my_camp.csv', index=None, header=None)

def gnr_blds_wd(dataset_path, n_samples, batch_size, n_dup_pb):
    df = pd.read_csv(dataset_path, header=None, names=['tid', 'text'])
    #df = pd.read_csv(dataset_path, header=None, names=['index', 'tid','text'])
    df1 = pd.read_csv('blds_l.csv')[['tid']]
    df = df[~df['tid'].isin(df1.loc[:, 'tid'])]

    iters = int(n_samples/batch_size)
    step = batch_size-n_dup_pb

    blds = df.sample(n=iters*step)
    ds = df.drop(blds.index)
    blds.index = list(range(len(blds)))


    dfs = [blds.loc[r_bound-48:r_bound-1] for r_bound in range(48, (iters*step)+step, step)]

    with open('duplicates.txt','w') as fp:
        for i in range(len(dfs)):
            elems = rnd.sample(range(dfs[i].index[0],dfs[i].index[-1]+1),2)
            fp.write(str(dfs[i].loc[elems[0],'tid'])+"\n"+str(dfs[i].loc[elems[1],'tid'])+"\n")
            dfs[i] = dfs[i].append(dfs[i].loc[elems], ignore_index=True)
            dfs[i] = dfs[i].sample(frac=1)

    blds = pd.concat(dfs)

    #blds[['index']].to_csv('drop.csv', index=None)
    #blds = blds[['tid','text']]
    blds.to_csv('baseline_dataset_wd.csv', index=None, header=['TID', 'SENTENCE'], quoting=csv.QUOTE_NONNUMERIC)
    ds.to_csv('dataset.csv',index=None, header=None)
    #ds.sample(n=500).to_csv('my_camp.csv', index=None, header=None)



def main():
    #filter()
    #hydrate_tweets(out_f_name='hydrated_tweets.json')
    #clear_tweets(infile='hydrated_tweets(58350).json')
    #create_sample_file(inf='baseline_dataset_cleaned.csv')
    #generate_baseline_dataset(dataset_path='hydrated_tweets(58350).json',n_samples=5000)
    #gnr_blds('dataset_cleaned.csv',n_samples=5000)
    gnr_blds_wd('dataset_cleaned_wol.csv', n_samples=5000, batch_size=50, n_dup_pb=2)
    #text = "voglio dell'acqua,      4344 per. ubriacarmi #COVIDー19_ @dioSca-_mello99 b😁 IT41-I076-0103-4000-0001-5177-827 IT25A3608105138245038445039 IT 61 V 05387 66320 000003105914 sdsssdsdsd"
    #text = cleaner(text)

    #dt = pd.read_csv('baseline_dataset_cleaned.csv', header=None)
    #dt = dt[1]
    #dt.to_csv('baseline_dataset_cleaned_ot.csv', index=False, header=None)

    #text = re.sub(r"[a-zA-Z]{2}[0-9]{2}[a-zA-Z0-9]{4}[0-9]{7}([a-zA-Z0-9]?){0,16}", "ibanid", text)
    #text = re.sub(r"[A-Z]{2}[A-Za-z0-9\-]{25,31}","ibanid",text)
    #out ='task_46837115-nr(1).csv'
    #df = pd.read_csv('batchs/task_46837115-nr.csv',header=None)
    #print(len(df))
    #df.index = [i for i in range(1,51)]
    #print(df)

    #df.to_csv('pr.csv', header=None)
    #df[['TASK_ID','START_TIME','Sentence', 'Label']].to_csv(out, index=None, header=None)
    #print(text)



    '''files = ['task_46837115', 'task_46877441','task_46858182','task_46947355','task_46924667','task_46924988','task_46949083','task_46986988']
    index = [1, 51, 101, 151, 201, 251, 301, 351]
    engine = create_engine('postgresql://mferraretto:qwerty@192.168.1.162:5432/DSTC19')

    for i in range(8):
        df = pd.read_csv('batchs/'+files[i]+'.csv', names=['id','time','tweet','label'])[['tweet','label']]
        df.index = list(range(index[i], index[i]+50))
        df.to_sql('blds_l', engine, if_exists='append', index_label='n_row')


    #df = pd.read_csv('baseline_dataset_cleaned.csv', names=['tid', 'tweet'])
    #df.index = list(range(1, len(df)+1))




    for file in files:
        df = pd.read_csv('batchs/'+file+'_i.csv', names=['index','sentence','label'])
        print(df)

    db = get_db_connection('172.20.10.11')
    cur = db.cursor()

    cur.execute("SELECT DISTINCT B.tid, L.tweet, L.label FROM blds_labelled AS L JOIN baseline_dataset AS B ON L.index = B.index")
    print("The number of parts: ", cur.rowcount)

    with open('baseline_dataset_l.csv', 'w') as fp:
        row = cur.fetchone()
        while row is not None:
            row = cur.fetchone()
            fp.write(str(row[0])+','+row[1]+','+row[2]+'\n')


    cur.close()'''

    #engine = create_engine('postgresql://mferraretto:qwerty@192.168.182.171:5432/DSTC19')

    #df = pd.read_csv('baseline_dataset_cleaned.csv', header=None, names=['tid','tweet'])
    #df.index = list(range(1,len(df)+1))
    #print(df)
    #df.to_sql('blds', engine, if_exists='append', index_label='n_row')
    #QUERY = "SELECT S.tid, L.tweet, L.label FROM blds_l L JOIN blds S ON L.n_row = S.n_row"
    #QUERY= "SELECT tid, COUNT(*) FROM blds WHERE n_row <= 400 GROUP BY tid HAVING COUNT(*) > 1"
    #df = pd.read_sql_query(QUERY, engine)
    #df.to_csv('TID.csv', index=None)

    #df = pd.read_csv('dlds_l.csv')
    #print(len(df))
    #df1 = df.drop_duplicates(subset=['tid'])
    #print(len(df1))
    #df1.to_csv('blds_l.csv', index=None)

    '''df = pd.read_csv('baseline_dataset_cleaned.csv', header=None)

    df1 = df.sample(n=1000)
    df2 = df.drop(labels=df1.index)
    print(len(df1))
    print(len(df2))'''

    #print(list(range(0,int((5000/50))*(50-2),50-2)))



if __name__ == '__main__':
    main()

    #print("CIs \ud83d\udc4d")



"!|\"|#|$|%|&|'|\(|\)|*|+|,|-|.|/|:|;|<|=|>|?|@|[|\|]|^|_|`|{|||}|~"
