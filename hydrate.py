"""
DSTC19
Copyright (C) 2021  Mattia Ferraretto <ferrar3tto.mattia@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import pandas as pd
from twarc import Twarc
import json
import psycopg2

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
        "coordinates": []   #float array [longitude, latitude]
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
    api_key = "API_KEY";
    api_secret_key = "API_SECRET_KEY";
    acces_token = "ACCESS_TOKEN";
    acces_token_secret = "ACCCES_TOKEN_SECRET";

    return Twarc(api_key, api_secret_key, acces_token, acces_token_secret);

def get_db_connection(host):
    return psycopg2.connect(dbname='tweets_dataset', user='mferraretto', password='qwerty', host=host)

def filter(out_f):

    QUERY = "SELECT tweet_id FROM tweets WHERE country_place = 'IT' AND lang = 'it' AND _date BETWEEN '2020-03-01' AND '2020-12-31' ORDER BY _date ASC"
    db = get_db_connection('localhost')
    curs = db.cursor()
    print("Query execution...")
    curs.execute(QUERY)
    tweets_id = [record[0] for record in curs]
    db.close()
    print("Query executed.")
    print("Write records...")
    pd.DataFrame(tweets_id).to_csv(out_f, index=False, header=None)
    print("Done.")

def hydrate_tweets(in_f, out_f):
    twrc = get_twarc()

    print("Hydrating tweets...")
    with open(out_f, "w") as out:
        for tweet in twrc.hydrate(open(in_f)):
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

def main():
    filter(out_f='tweets_id_ita_IT.csv')
    hydrate_tweets(in_f='tweets_id_ita_IT.csv', out_f='raw_dataset.json')


if __name__ == '__main__':
    main()