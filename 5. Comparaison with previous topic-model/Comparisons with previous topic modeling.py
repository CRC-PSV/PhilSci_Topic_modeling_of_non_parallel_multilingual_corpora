# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Mon Mar 22 10:03:21 2021
@author: Francis Lareau
This is Projectjstor.
Topic model with LDA (Gibbs sampling) comparison
"""

#==============================================================================
# ############################################################## Import library
#==============================================================================

import os
import pickle
import bz2
import numpy as np
import pandas as pd
from scipy.spatial import distance
from gensim.matutils import hellinger

#==============================================================================
# #################################################### Initialize project paths
#==============================================================================

main_path = os.path.join("D:\projetjstor\Consolidation\Translation") #where is input and output
os.chdir(main_path)

#==============================================================================
# ################################################################# Import data
#==============================================================================


DTM_en = pd.read_pickle(bz2.BZ2File(
        os.path.join(main_path,
                     "0. Data",
                     "DTM_philosophy_of_science_en.pbz2"), 'rb'))   

with open(os.path.join(main_path,
                       "0. Data",
                       "Vocabulary_philosophy_of_science_en.pkl"), "rb") as f:
    Vocab_en = pickle.load(f) 

DF_statistique_generale_en = pd.read_pickle(
        os.path.join(main_path,
                     "0. Data",
                     "DF_philosophy_of_science_en_metadata.pkl"))

with open(os.path.join(main_path,
                       "0. Data",
                       "LDA_model_philosophy_of_science_en_K25.pkl"), "rb") as f:
    ldamodel_lda_en = pickle.load(f)

DTM = pd.read_pickle(bz2.BZ2File(
        os.path.join(main_path,
                     "0. Data",
                     "DTM_philosophy_of_science_all.pbz2"), 'rb'))  

with open(os.path.join(main_path,
                       "0. Data",
                       "Vocabulary_philosophy_of_science_all.pkl"), "rb") as f:
    Vocab = pickle.load(f) 

DF_statistique_generale = pd.read_pickle(
        os.path.join(main_path,
                     "0. Data",
                     "DF_philosophy_of_science_all_metadata.pkl"))

with open(os.path.join(main_path,
                       "0. Data",
                       "LDA_model_philosophy_of_science_all_K25.pkl"), "rb") as f:
    ldamodel_lda = pickle.load(f)

#==============================================================================
# ########################################################### Document x Topics
#==============================================================================

checklist = list()
for i in range(len(DF_statistique_generale)):
    if DF_statistique_generale.Citation[i] in [x for x in DF_statistique_generale_en.Citation]:
        checklist.append(True)
    else:
        checklist.append(False)
#test        
[x for x in DF_statistique_generale.Citation[checklist]] == [x for x in DF_statistique_generale_en.Citation]
 
DF_DT_euc = pd.DataFrame(columns=["Topic_t_" + str(i) for i in range(len(ldamodel_lda_en.components_))],
                       index=["Topic_" + str(i) for i in range(len(ldamodel_lda_en.components_))])

for topic_t in range(25):
    for topic in range(25):
        dist = distance.minkowski(ldamodel_lda.doc_topic_[checklist].transpose()[topic_t],
                                  ldamodel_lda_en.doc_topic_.transpose()[topic], 2)
        DF_DT_euc['Topic_t_'+str(topic_t)][topic]=dist
        
DF_DT_cos = pd.DataFrame(columns=["Topic_t_" + str(i) for i in range(len(ldamodel_lda_en.components_))],
                       index=["Topic_" + str(i) for i in range(len(ldamodel_lda_en.components_))])

for topic_t in range(25):
    for topic in range(25):
        dist = distance.cosine(ldamodel_lda.doc_topic_[checklist].transpose()[topic_t],
                                  ldamodel_lda_en.doc_topic_.transpose()[topic])
        DF_DT_cos['Topic_t_'+str(topic_t)][topic]=dist
        
DF_DT_hel = pd.DataFrame(columns=["Topic_t_" + str(i) for i in range(len(ldamodel_lda_en.components_))],
                       index=["Topic_" + str(i) for i in range(len(ldamodel_lda_en.components_))])

#for topic_t in range(25):
#    for topic in range(25):
#        dist = hellinger(ldamodel_lda.doc_topic_[checklist].transpose()[topic_t],
#                                  ldamodel_lda_en.doc_topic_.transpose()[topic])
#        DF_DT_hel['Topic_t_'+str(topic_t)][topic]=dist
        
for topic_t in range(25):
    for topic in range(25):
        dist = hellinger(ldamodel_lda.doc_topic_[checklist].transpose()[topic_t]/sum(ldamodel_lda.doc_topic_[checklist].transpose()[topic_t]),
                         ldamodel_lda_en.doc_topic_.transpose()[topic]/sum(ldamodel_lda_en.doc_topic_.transpose()[topic]))
        DF_DT_hel['Topic_t_'+str(topic_t)][topic]=dist
        
DF_DT_shan = pd.DataFrame(columns=["Topic_t_" + str(i) for i in range(len(ldamodel_lda_en.components_))],
                                   index=["Topic_" + str(i) for i in range(len(ldamodel_lda_en.components_))])

for topic_t in range(25):
    for topic in range(25):
        dist = distance.jensenshannon(ldamodel_lda.doc_topic_[checklist].transpose()[topic_t],
                                      ldamodel_lda_en.doc_topic_.transpose()[topic])
        DF_DT_shan['Topic_t_'+str(topic_t)][topic]=dist

#==============================================================================
# ############################################################### Word x Topics
#==============================================================================

word_list_en = sorted([i for i in Vocab_en.keys()])
word_list = sorted([i for i in Vocab.keys()])

checklist_w = list()
for i in range(len(word_list_en)):
    if word_list_en[i] in word_list:
        checklist_w.append(True)
    else:
        checklist_w.append(False)

checklist_w_t = list()
for i in range(len(word_list)):
    if word_list[i] in word_list_en:
        checklist_w_t.append(True)
    else:
        checklist_w_t.append(False)
#test        
[x for x in np.array(word_list)[checklist_w_t]] == [x for x in np.array(word_list_en)[checklist_w]]


DF_WT_euc = pd.DataFrame(columns=["Topic_t_" + str(i) for i in range(len(ldamodel_lda_en.components_))],
                       index=["Topic_" + str(i) for i in range(len(ldamodel_lda_en.components_))])

for topic_t in range(25):
    for topic in range(25):
        dist = distance.minkowski(ldamodel_lda.components_.transpose()[checklist_w_t,].transpose()[topic_t],
                                  ldamodel_lda_en.components_.transpose()[checklist_w].transpose()[topic], 2)
        DF_WT_euc['Topic_t_'+str(topic_t)][topic]=dist
        
DF_WT_cos = pd.DataFrame(columns=["Topic_t_" + str(i) for i in range(len(ldamodel_lda_en.components_))],
                       index=["Topic_" + str(i) for i in range(len(ldamodel_lda_en.components_))])

for topic_t in range(25):
    for topic in range(25):
        dist = distance.cosine(ldamodel_lda.components_.transpose()[checklist_w_t,].transpose()[topic_t],
                                  ldamodel_lda_en.components_.transpose()[checklist_w].transpose()[topic])
        DF_WT_cos['Topic_t_'+str(topic_t)][topic]=dist
        
DF_WT_hel = pd.DataFrame(columns=["Topic_t_" + str(i) for i in range(len(ldamodel_lda_en.components_))],
                       index=["Topic_" + str(i) for i in range(len(ldamodel_lda_en.components_))])

#for topic_t in range(25):
#    for topic in range(25):
#        dist = hellinger(ldamodel_lda.components_.transpose()[checklist_w_t,].transpose()[topic_t],
#                                  ldamodel_lda_en.components_.transpose()[checklist_w].transpose()[topic])
#        DF_WT_hel['Topic_t_'+str(topic_t)][topic]=dist
        
for topic_t in range(25):
    for topic in range(25):
        dist = hellinger(ldamodel_lda.components_.transpose()[checklist_w_t,].transpose()[topic_t]/sum(ldamodel_lda.components_.transpose()[checklist_w_t,].transpose()[topic_t]),
        ldamodel_lda_en.components_.transpose()[checklist_w].transpose()[topic]/sum(ldamodel_lda_en.components_.transpose()[checklist_w].transpose()[topic]))
        DF_WT_hel['Topic_t_'+str(topic_t)][topic]=dist

DF_WT_shan = pd.DataFrame(columns=["Topic_t_" + str(i) for i in range(len(ldamodel_lda_en.components_))],
                                   index=["Topic_" + str(i) for i in range(len(ldamodel_lda_en.components_))])
for topic_t in range(25):
    for topic in range(25):
        dist = distance.jensenshannon(ldamodel_lda.components_.transpose()[checklist_w_t,].transpose()[topic_t],
                                      ldamodel_lda_en.components_.transpose()[checklist_w].transpose()[topic])
        DF_WT_shan['Topic_t_'+str(topic_t)][topic]=dist
        
#==============================================================================
# ################################################################ Save results
#==============================================================================
        
writer = pd.ExcelWriter(os.path.join(
        main_path,
        "5. Comparisons with previous topic modeling",
        "Results_from_comparisons_with_previous_topic_modeling.xlsx"))
DF_DT_euc.to_excel(writer,'Documents Euclidian',encoding='utf8') 
DF_DT_cos.to_excel(writer,'Documents Cosinus',encoding='utf8') 
DF_DT_hel.to_excel(writer,'Documents Hellinger',encoding='utf8')
DF_DT_shan.to_excel(writer,'Documents Jensen-Shannon',encoding='utf8')
DF_WT_euc.to_excel(writer,'Words Euclidian',encoding='utf8') 
DF_WT_cos.to_excel(writer,'Words Cosinus',encoding='utf8') 
DF_WT_hel.to_excel(writer,'Words Hellinger',encoding='utf8') 
DF_WT_shan.to_excel(writer,'Words Jensen-Shannon',encoding='utf8') 
writer.save()
writer.close()
