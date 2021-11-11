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
import spacy
from spacy import displacy
import pandas as pd
from spacy.symbols import nsubj, VERB
from tabulate import tabulate


DEP_DICT = {
    "SUBJECTS" : ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"],
    "OBJECTS" : ["dobj", "dative", "attr", "oprd", "obj", "iobj"],
    "AUX" : ["aux", "cop"]
}

ENTITIES = ["DATE", "CARDINAL", "GPE", "FAC", "LOC", "NORP"]

def find_verb_wdep(document, dep):
    verbs = set()
    for token in document:
        if token.dep_ in dep and token.head.pos == VERB:
            verbs.add(token.head)

    return verbs

def check_conj(token_morph):

    if len(token_morph.get("Number")) == 0 or len(token_morph.get("Person")) == 0:
        return False

    return token_morph.get("Number")[0] == "Sing" and token_morph.get("Person")[0] == "1"

def analyzer(vset, dep):
    b_set = set()

    for verb in vset:
        b_set.add(check_conj(verb.morph))
        for vchild in verb.children:
            if vchild.dep_ in dep:
                if vchild.has_morph:
                    b_set.add(check_conj(vchild.morph))

    return {True}.issubset(b_set)

def step_1(doc):
    subj_vb = find_verb_wdep(document=doc, dep=DEP_DICT["SUBJECTS"])

    vb_obj = find_verb_wdep(document=doc, dep=DEP_DICT["OBJECTS"])

    aux_vb = find_verb_wdep(document=doc, dep=DEP_DICT["AUX"])

    subj_verb_obj = set.intersection(subj_vb,vb_obj)

    ok_svo = False
    ok_subj = False
    ok_aux = False
    ok_obj = False

    if len(subj_verb_obj) > 0:
        ok_svo = analyzer(vset=subj_verb_obj, dep=DEP_DICT["SUBJECTS"] + DEP_DICT["OBJECTS"])
    else:
        if len(subj_vb) > 0:
            ok_subj = analyzer(vset=subj_vb, dep=DEP_DICT["SUBJECTS"])

        if len(vb_obj) > 0:
            ok_obj = analyzer(vset=vb_obj, dep=DEP_DICT["OBJECTS"])

        if len(aux_vb) > 0:
            ok_aux = analyzer(vset=aux_vb, dep=DEP_DICT["AUX"])

    return ok_svo or ok_subj or ok_aux or ok_obj

def step_2(doc):
    return len([ent for ent in doc.ents if ent.label_ in ENTITIES]) > 0

def print_stats(df, step_1_msk, step_2_msk, only_step_1_msk, only_step_2_msk, both_steps_msk):

    df_stats = df['label'].value_counts()
    df1 = df[step_1_msk]
    df2 = df[step_2_msk]
    df3 = df[both_steps_msk]
    df4 = df[only_step_1_msk]
    df5 = df[only_step_2_msk]
    df1_stats = df1['label'].value_counts()
    df2_stats = df2['label'].value_counts()
    df3_stats = df3['label'].value_counts()
    df4_stats = df4['label'].value_counts()
    df5_stats = df5['label'].value_counts()

    print(tabulate([['Baseline dataset', len(df), df_stats['NSD'], df_stats['SD']],
                    ['Step 1', len(df1), df1_stats['NSD'], df1_stats['SD']],
                    ['Step 2', len(df2), df2_stats['NSD'], df2_stats['SD']],
                    ['Step 1 and Step 2', len(df3), df3_stats['NSD'], df3_stats['SD']],
                    ['Step 1 and not Step 2', len(df4), df4_stats['NSD'], df4_stats['SD']],
                    ['not Step 1 and Step 2', len(df5), df5_stats['NSD'], df5_stats['SD']]], headers=['', 'TOTAL', 'NSD', 'SD'], tablefmt='orgtbl'))



def main():
    nlp = spacy.load("it_core_news_lg")



    df = pd.read_csv(str(sys.argv[1]), header=None, names=['tid','text'])
    #df = pd.read_csv(str(sys.argv[1]))

    step_1_msk = []
    step_2_msk = []
    only_step_1_msk = []
    only_step_2_msk = []
    both_steps_msk = []


    for i in range(len(df)):
        doc = nlp(df.loc[i,'text'])
        isOk_step_1 = step_1(doc)
        isOk_step_2 = step_2(doc)
        step_1_msk.append(isOk_step_1)
        step_2_msk.append(isOk_step_2)
        only_step_1_msk.append(isOk_step_1 and not isOk_step_2)
        only_step_2_msk.append(not isOk_step_1 and isOk_step_2)
        both_steps_msk.append(isOk_step_1 and isOk_step_2)

    df_only_1 = df[only_step_1_msk]
    df_only_1.to_csv('only_step_1.csv', index=None)
    df_both_steps = df[both_steps_msk]
    df_both_steps.to_csv('both_steps.csv', index=None)


    #print_stats(df, step_1_msk, step_2_msk,only_step_1_msk, only_step_2_msk, both_steps_msk)

    #displacy.serve(doc, style="ent")

if __name__ == '__main__':
    main()
