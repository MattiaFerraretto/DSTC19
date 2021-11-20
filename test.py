"""
SDTH
Copyright (C) 2021  Mattia Ferraretto <ferraretto.mattia@protonmail.com>
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

import sys
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from tabulate import tabulate


def do_predictions(df):

    tokenizer = AutoTokenizer.from_pretrained("model/")
    model = AutoModelForSequenceClassification.from_pretrained("model/")
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

    label = []
    score = []

    for sentence in tqdm(df['text']):
        result = classifier(sentence)[0]
        label.append(result['label'])
        score.append(round(result['score'], 4))

    df = df.assign(prediction=pd.Series(label))
    df = df.assign(score=pd.Series(score))
    df.to_csv('test_data-result.csv', index=None)


def confusion_matrix(df):
    c_mat =[[0, 0],
            [0, 0]]

    for i in range(len(df)):
        label = df['label'].iloc[i]
        pred = df['prediction'].iloc[i]

        if pred == 'SD' and label == 'SD':
            c_mat[0][0] += 1
        elif pred == 'NSD' and label == 'SD':
            c_mat[0][1] += 1
        elif pred == 'SD' and label == 'NSD':
            c_mat[1][0] += 1
        else:
            c_mat[1][1] += 1

    return c_mat

def print_metrics(df):

    c_mat = confusion_matrix(df)
    tab = tabulate([['Positive', c_mat[0][0], c_mat[0][1]],
                    ['Negative', c_mat[1][0], c_mat[1][1]]], headers=['', 'PP', 'PN'], tablefmt='orgtbl')

    precision = round(c_mat[0][0]/(c_mat[0][0]+c_mat[1][0]), 4)
    recall = round(c_mat[0][0]/(c_mat[0][0]+ c_mat[0][1]), 4)
    f_score = round(2*c_mat[0][0]/(2*c_mat[0][0]+c_mat[1][0]+c_mat[0][1]), 4)

    result_str = f"\nPP = Prediction positive\nPN = Prediction negative\n\n{tab}\n\nPrecision: {precision}\nRecall: {recall}\nF_1 score: {f_score}"
    print(result_str)
    with open('test_result.txt', 'w') as fp:
        fp.write(result_str)

def main():

    df = pd.read_csv(str(sys.argv[1]))

    #do_predictions(df)
    #print_metrics(df)



if __name__ == '__main__':
    main()
