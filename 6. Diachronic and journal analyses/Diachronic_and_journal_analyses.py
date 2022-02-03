# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Tue Mar 23 16:41:29 2021
@author: Francis Lareau
This is Projectjstor.
Data for diachronic and journal analysis
"""

#==============================================================================
# ############################################################## Import library
#==============================================================================

import os
import numpy as np
import pandas as pd
import pickle
import bz2

#==============================================================================
# #################################################### Initialize project paths
#==============================================================================

main_path = os.path.join("D:\projetjstor\Consolidation\Translation") #where is input and output
os.chdir(main_path)

#==============================================================================
# ################################################################# Import data
#==============================================================================

with open(os.path.join(main_path,
                       "0. Data",
                       "LDA_model_philosophy_of_science_all_K25.pkl"), "rb") as f:
    ldamodel_lda = pickle.load(f)

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

#==============================================================================
# ##################### Data statistic, lda model score and lda hyperparameters
#==============================================================================
  
df_param=pd.DataFrame(index=['Value'])
df_param['Sparsity']=((DTM.todense() > 0).sum() / 
        DTM.todense().size*100) #sparsicity (% nonzero)
df_param['Log Likelyhood']=ldamodel_lda.loglikelihood() #Log Likelyhood (higher better)
df_param['Perplexity']='' #Perplexity (lower better, exp(-1. * log-likelihood per word)
df_param['alpha']=ldamodel_lda.alpha
df_param['eta']=ldamodel_lda.eta
df_param['n_iter']=ldamodel_lda.n_iter
df_param['n_components']=ldamodel_lda.n_topics
df_param['random_state']=ldamodel_lda.random_state
df_param['refresh']=ldamodel_lda.refresh

#==============================================================================
# ########################################################### Topic by document
#==============================================================================

#Topic for each document
lda_output=ldamodel_lda.doc_topic_
topicnames = ["Topic_" + str(i) for i in range(len(ldamodel_lda.components_))]
docnames = [i for i in range(DTM.shape[0])]
df_document_topic = pd.DataFrame(lda_output, 
                                 columns=topicnames,
                                 index=docnames)
dominant_topic = np.argmax(df_document_topic.values, axis=1)
#add results to statistic general
DF_statistique_generale['Dom_topic'] = dominant_topic
DF_topic=pd.concat([DF_statistique_generale,df_document_topic],
                   axis=1,
                   join='inner')
    
#count document by topic
df_topic_distribution = DF_statistique_generale['Dom_topic'].value_counts(
        ).reset_index(name="Num_Documents")
df_topic_distribution.columns = ['Topic_Num', 'Num_Doc']
# Topic - keyword Matrix
df_topic_keywords = pd.DataFrame(ldamodel_lda.components_)#every row =1
df_topic_keywords.index = topicnames
#Transpose to topic - keyword matrix
df_keywords_topic = df_topic_keywords.transpose()
df_keywords_topic.index = sorted([i for i in Vocab.keys()])
# Topic - Top Keywords Dataframe
n_top_words = 50+1
DF_Topic_TKW = pd.DataFrame(columns=range(n_top_words-1),index=range(len(ldamodel_lda.components_)))
vocab = sorted([i for i in Vocab.keys()])
topic_word = ldamodel_lda.components_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    DF_Topic_TKW.loc[i]=topic_words

DF_Topic_TKW.columns = ['Word_'+str(i) for i in range(DF_Topic_TKW.shape[1])]
DF_Topic_TKW.index = ['Topic_'+str(i) for i in range(DF_Topic_TKW.shape[0])]
DF_Topic_TKW['Sum_Doc'] = np.array(DF_statistique_generale['Dom_topic'].value_counts(
        ).sort_index())
DF_Topic_TKW['Top-10_Words'] = ''
for idx,row in DF_Topic_TKW.iterrows():
    DF_Topic_TKW['Top-10_Words'][idx]=(row['Word_0']+'; '+row['Word_1']+'; '+
                row['Word_2']+'; '+row['Word_3']+'; '+row['Word_4']+'; '+
                row['Word_5']+'; '+row['Word_6']+'; '+row['Word_7']+'; '+
                row['Word_8']+'; '+row['Word_9'])

#==============================================================================
# ############################################################# Topic by period
#==============================================================================

#DF_topic['Period']=DF_topic.Year.apply(lambda x: #2 years period
#    x+'-'+str(int(x)+1) if int(x) % 2 == 0 else str(int(x)-1)+'-'+x)    

DF_topic['Period']=DF_topic.Year.apply(lambda x: #4 years period
    str(int(x)-(int(x)+2)%4)+'-'+str((int(x)-(int(x)+2)%4)+3))
    
# Topic - Period Matrix
DF_PT=pd.DataFrame(lda_output,
                   columns=topicnames,
                   index=docnames)

DF_PT['Period']=DF_topic.Period
DF_PT = DF_PT.groupby(['Period']).sum()
DF_TP = DF_PT.transpose()
DF_TP = DF_TP/DF_TP.sum()
DF_TP_Overall = DF_PT.transpose()
DF_TP_Overall['Raw'] = DF_PT.sum()
DF_TP_Overall['Overall'] = DF_PT.sum() / sum(DF_PT.sum())

# Topic - Period Matrix (not translated)
NT = ~DF_statistique_generale.translated
DF_PTNT=pd.DataFrame(lda_output[NT],
                     columns=topicnames,
                     index=np.array(docnames)[NT])

DF_PTNT['Period']=DF_topic.Period[NT]
DF_PTNT = DF_PTNT.groupby(['Period']).sum()
DF_TPNT = DF_PTNT.transpose()
DF_TPNT = DF_TPNT/DF_TPNT.sum()
DF_TPNT_Overall = DF_PTNT.transpose()
DF_TPNT_Overall['Raw'] = DF_PTNT.sum()
DF_TPNT_Overall['Overall'] = DF_PTNT.sum() / sum(DF_PTNT.sum())

# Topic - Period Matrix (translated)
T = DF_statistique_generale.translated
DF_PTT=pd.DataFrame(lda_output[T],
                    columns=topicnames,
                    index=np.array(docnames)[T])

DF_PTT['Period']=DF_topic.Period[T]
DF_PTT = DF_PTT.groupby(['Period']).sum()
DF_TPT = DF_PTT.transpose()
DF_TPT = DF_TPT/DF_TPT.sum()
DF_TPT_Overall = DF_PTT.transpose()
DF_TPT_Overall['Raw'] = DF_PTT.sum()
DF_TPT_Overall['Overall'] = DF_PTT.sum() / sum(DF_PTT.sum())

#==============================================================================
# ################################################### Topic by period + journal
#==============================================================================

# Topic - Journal + Period Matrix
DF_temp = pd.DataFrame([[item for sublist in [sorted(set(DF_topic.Period))]*len(set(DF_topic.Journal_id)) for item in sublist],
                         [item for sublist in [[x]*len(set(DF_topic.Period)) for x in set(DF_topic.Journal_id)] for item in sublist]
                         ], index=['Period','Journal_id']).transpose()
    
DF_PJT=pd.DataFrame(lda_output,
                   columns=topicnames,
                   index=docnames)
DF_PJT['Period']=DF_topic.Period
DF_PJT['Journal_id']=DF_topic.Journal_id
count_all = DF_PJT.groupby(['Journal_id','Period']).count().transpose().iloc[0]
count_t = DF_PJT[T].groupby(['Journal_id','Period']).count().transpose().iloc[0]

DF_PJT = pd.concat([DF_temp,DF_PJT])

DF_PJT = DF_PJT.groupby(['Journal_id','Period']).sum()
DF_PJT['count_all'] = 0
DF_PJT['count_tran'] = 0
DF_TPJ = DF_PJT.transpose()
DF_TPJ = DF_TPJ/DF_TPJ.sum()
DF_TPJ.loc['count_all'] = count_all
DF_TPJ.loc['count_tran'] = count_t

DF_TPJ_Overall = DF_PJT.transpose()
DF_TPJ_Overall['Raw'] = DF_PJT.sum()
DF_TPJ_Overall['Overall'] = DF_PJT.sum() / sum(DF_PJT.sum())
DF_TPJ_Overall.loc['count_all'] = count_all
DF_TPJ_Overall.loc['count_tran'] = count_t

# Topic - Journal Matrix
DF_JT = pd.concat([DF_statistique_generale['Journal_id'],df_document_topic],
                   axis=1,
                   join='inner')
DF_JT = DF_JT.groupby(['Journal_id']).sum()
DF_TJ = DF_JT.transpose()
DF_TJ = DF_TJ/DF_TJ.sum() 
DF_TJ_Overall = DF_JT.transpose()
DF_TJ_Overall['Raw'] = DF_JT.sum()
DF_TJ_Overall['Overall'] = DF_JT.sum() / sum(DF_JT.sum())
# Periods - Topics top_10 articles Matrix (sorted by year)
DF_PT_T10A=pd.DataFrame(data='', index=DF_TP.columns,columns=DF_TP.index)
for period in DF_TP.columns:
    for topic in DF_TP.index:
        for idx in DF_topic[DF_topic.Period==period].nlargest(
                10,topic).sort_values('Year',ascending=False).index:
            DF_PT_T10A[topic][period]=DF_PT_T10A[topic][period]+DF_topic.Citation[idx]+'\n'

#==============================================================================
# ############################################################# Topic by Author
#==============================================================================
            
# Author - Topic Matrix
authors = set()
for group_author in DF_statistique_generale['Author']:
    for author in group_author:
        authors.add(author)
authors = sorted(authors)

DF_AT = pd.DataFrame(data='', index=range(len(authors)),columns=topicnames)
for idx,author in enumerate(authors):
    list_bool = DF_statistique_generale.Author.apply(lambda x: True if author in x else False)
    DF_AT.loc[idx]=sum(lda_output[list_bool])/len(lda_output[list_bool])
        
DF_AT['Author']={', '.join(x) for x in authors}

# Author T - Matrix
DF_ATT = pd.concat([DF_statistique_generale[T]['Author'],df_document_topic[T]],
                   axis=1,
                   join='inner')
DF_ATT.Author = DF_ATT.Author.apply(lambda x: '; '.join(', '.join(e) for e in x))
DF_ATT = DF_ATT.groupby(['Author']).sum()
DF_TAT = DF_ATT.transpose()
count_tran = DF_TAT.sum()
DF_TAT = DF_TAT/count_tran
DF_ATT = DF_TAT.transpose()
DF_ATT['NDoc'] = count_tran

#==============================================================================
# ########################################################### Topic correlation
#==============================================================================

# Topic Pearson Correlation
DF_TfromD = df_document_topic.corr(method='pearson')
DF_TfromD_Stack = pd.DataFrame(columns=['Topic_A','Topic_B','Correlation'])
for id1,topic1 in enumerate(topicnames):    
    for id2,topic2 in enumerate(topicnames):
        n_id = DF_TfromD_Stack.shape[0]
        DF_TfromD_Stack.loc[n_id] = [str(id1+1),str(id2+1),DF_TfromD[topic1][topic2]]
##
DF_TfromW=df_topic_keywords.T.corr(method='pearson')
DF_TfromW_Stack = pd.DataFrame(columns=['Topic_A','Topic_B','Correlation'])
for id1,topic1 in enumerate(topicnames):    
    for id2,topic2 in enumerate(topicnames):
        n_id = DF_TfromW_Stack.shape[0]
        DF_TfromW_Stack.loc[n_id] = [str(id1+1),str(id2+1),DF_TfromW[topic1][topic2]]
    
#==============================================================================
# ################################################################ Save results
#==============================================================================
        
# Save lda results to excel
writer = pd.ExcelWriter(os.path.join(main_path,
                                     "6. Diachronic and journal analyses",
                                     "Results_from_diachronic_and_journal_analyses.xlsx"))
df_param.T.to_excel(writer,'Para Score',encoding='utf8')        
DF_topic.to_excel(writer,'Doc vs Topic',encoding='utf8')
DF_Topic_TKW.to_excel(writer,'Top 50 Topics Words',encoding='utf8')
df_keywords_topic.to_excel(writer,'Words vs Topics',encoding='utf8',
                           header=topicnames,
                           index=sorted([i for i in Vocab.keys()]))
DF_AT.to_excel(writer,'Authors vs Topics T',encoding='utf8')
DF_ATT.to_excel(writer,'Authors vs Topics',encoding='utf8')
DF_TP.to_excel(writer,'Topics vs Periods',encoding='utf8')
DF_TPNT.to_excel(writer,'Topics vs Periods NT',encoding='utf8')
DF_TPT.to_excel(writer,'Topics vs Periods T',encoding='utf8')
DF_TP_Overall.to_excel(writer,'Overall Topics vs Periods',encoding='utf8')
DF_TPNT_Overall.to_excel(writer,'Overall Topics vs Periods NT',encoding='utf8')
DF_TPT_Overall.to_excel(writer,'Overall Topics vs Periods T',encoding='utf8')
DF_TPJ.to_excel(writer,'Topics vs Periods+Journals',encoding='utf8')
DF_TPJ_Overall.to_excel(writer,'Over Topics vs Periods+Journals',encoding='utf8')
DF_TJ.to_excel(writer,'Topics vs Journals',encoding='utf8')
DF_TJ_Overall.to_excel(writer,'Overall Topics vs Journals',encoding='utf8')
DF_PT_T10A.to_excel(writer,'Top 10 articles',encoding='utf8')
DF_topic_sorted = df_document_topic.apply(lambda x: x.sort_values(ascending=False).values)
DF_topic_sorted.to_excel(writer,'Doc vs Topic Decreas',encoding='utf8')
DF_TfromD.to_excel(writer,'Topic Cor. from Doc',encoding='utf8')
DF_TfromD_Stack.to_excel(writer,'Topic Cor. from Doc Stack',encoding='utf8')
DF_TfromW.to_excel(writer,'Topic Cor. from Word',encoding='utf8')
DF_TfromW_Stack.to_excel(writer,'Topic Cor. from Word Stack',encoding='utf8')
writer.save()
writer.close()
