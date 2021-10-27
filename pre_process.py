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
import json
import re
import textacy.preprocessing.replace as rp
import textacy.preprocessing.normalize as nr

def cleaner(tweet_text):

    #tweet_text = tweet_text.lower()
    tweet_text = nr.unicode(tweet_text, form="NFC")
    tweet_text = re.sub(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', "emailid", tweet_text)
    tweet_text = re.sub(r"[a-zA-Z]{2}[0-9]{2}[a-zA-Z0-9]{4}[0-9]{7}([a-zA-Z0-9]?){0,16}", "ibanid", tweet_text)
    tweet_text = rp.user_handles(tweet_text, "usermention")
    tweet_text = re.sub(r"\B#\w*[a-zA-Z]+\w*", "hashtag", tweet_text, 0, re.MULTILINE)
    tweet_text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "url", tweet_text)
    tweet_text = re.sub(r'(\(?\d{2,4}\)?\D{0,3}\d{6,10}|\+\d{10,12})', "phonenumber", tweet_text)
    tweet_text = nr.quotation_marks(tweet_text)
    tokens = re.split(r'\s+',tweet_text)
    regx = r"\w|!|%|'|\(|\)|,|\.|:|;|\?|_|`|/|-|\"|€"
    words = [''.join(re.findall(regx, word)) for word in tokens if re.findall(regx, word)]
    #stop_word = set(stopwords.words('italian'))
    #words = [word for word in words if word not in stop_word]
    return ' '.join(words)

def clear_tweets(in_f, out_f):

    print('Cleaning up tweets...')
    tweets = [json.loads(line) for line in open(in_f, "r")]

    ids = []
    text_cleaned = []
    for tweet in tweets:
        ids.append(tweet['id'])
        text_cleaned.append(cleaner(tweet['full_text']))

    df = pd.DataFrame(data=dict({'tweets_ids': ids, 'cleaned_text': text_cleaned}))
    df.to_csv(out_f, index=None, header=None)
    print('Cleaned tweets.')

def main():

    clear_tweets(inf_f='raw_dataset.csv', out_f='dataset.csv')


if __name__ == '__main__':
    main()