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
import pandas as pd
import random as rnd
import csv

def gnr_blds(dataset_path, n_samples):
    df = pd.read_csv(dataset_path, header=None, names=['tid','text'])
    #df1 = pd.read_csv('blds_l.csv')[['tid']]
    #df = df[~df['tid'].isin(df1.loc[:, 'tid'])]

    blds = df.sample(n=n_samples)
    ds = df.drop(blds.index)
    blds.to_csv('baseline_dataset.csv', index=None, header=['TID', 'SENTENCE'], quoting=csv.QUOTE_NONNUMERIC)
    ds.to_csv('dataset_wo_blds.csv',index=None, header=None)

def gnr_blds_wd(dataset_path, n_samples, batch_size, n_dup_pb):
    df = pd.read_csv(dataset_path, header=None, names=['tid', 'text'])
    #df1 = pd.read_csv('blds_l.csv')[['tid']]
    #df = df[~df['tid'].isin(df1.loc[:, 'tid'])]

    iters = int(n_samples/batch_size)
    step = batch_size-n_dup_pb

    blds = df.sample(n=iters*step)
    ds = df.drop(blds.index)
    blds.index = list(range(len(blds)))


    dfs = [blds.loc[r_bound-step:r_bound-1] for r_bound in range(step, (iters*step)+step, step)]

    with open('duplicates.txt','w') as fp:
        for i in range(len(dfs)):
            elems = rnd.sample(range(dfs[i].index[0],dfs[i].index[-1]+1),2)
            fp.write(str(dfs[i].loc[elems[0],'tid'])+"\n"+str(dfs[i].loc[elems[1],'tid'])+"\n")
            dfs[i] = dfs[i].append(dfs[i].loc[elems], ignore_index=True)
            dfs[i] = dfs[i].sample(frac=1)

    blds = pd.concat(dfs)

    blds.to_csv('baseline_dataset_wd.csv', index=None, header=['TID', 'SENTENCE'], quoting=csv.QUOTE_NONNUMERIC)
    ds.to_csv('dataset_wo_blds.csv',index=None, header=None)

def main():
    #gnr_blds(dataset_path='dataset.csv', n_samples=5000)
    #gnr_blds_wd(dataset_path='dataset.csv', n_samples=5000, batch_size=50, n_dup_pb=2)


if __name__ == '__main__':
    main()
