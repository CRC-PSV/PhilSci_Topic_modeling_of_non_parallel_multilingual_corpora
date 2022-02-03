# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Tue Mar 23 18:35:08 2021
@author: Francis Lareau
This is Projectjstor.
Machine translation
"""
#==============================================================================
# ############################################################## Import library
#==============================================================================

import re
import time
import numpy as np
import pandas as pd
import os
import spacy
nlp = spacy.load('de_core_news_sm')
import html
from google.cloud import translate_v2 as translate
import langid
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

#==============================================================================
# #################################################### Initialize project paths
#==============================================================================

main_path = os.path.join("your_main_path")
os.chdir(main_path)

#==============================================================================
# ################################################################# Import data
#==============================================================================

DF = pd.read_pickle(
        os.path.join(main_path,
                     "0. Data",
                     "Private",
                     "DataFrame_Consolidation_updated_notenglish_source.pkl"))

#==============================================================================
# ############################################################# Detect language
#==============================================================================
        
DF['Lang_detect_1']=''

counter=0
for i in range(0,len(DF)):
    try: #try isn't needed for langid, but it is for langdetect
        lang=langid.classify(DF.Article[i])[0]
    except:
        lang=''
        counter+=1
    DF['Lang_detect_1'][i]=lang
    
DF['Lang_detect_2']=''

counter=0
for i in range(0,len(DF)):
    try: #try isn't needed for langid, but it is for langdetect
        lang=detect(DF.Article[i])
    except:
        lang=''
        counter+=1
    DF['Lang_detect_2'][i]=lang

#==============================================================================
# ############################################################## Filter article​
#==============================================================================​

DF = DF[((DF.Language!='')&(DF.Language!='en'))|(DF.Lang_detect_1!='en')|(DF.Lang_detect_2!='en')]
DF = DF[(DF.Article_type_1!='Erratum')|(DF.Article_type_1!='Bibliography')|(DF.Article_type_2!='OBITUARY')|(DF.Article_type_2!='BOOKSRECEIVED')]
DF = DF[~DF.Title.str.contains('[Bb]iblio|[Bb]ook|Correspondentie|[Ee]rrat|[Cc]orrigendum|[Cc]orrection|[Ll]iteratur', regex= True, na=False)]
DF = DF[DF.Article.str.len() > 4000]
DF = DF[DF.Author.astype(bool)]
DF.reset_index(inplace=True)

#==============================================================================
# ######################################################## Google Translate API​
#==============================================================================​

​#Google Translate API need an account and a key (https://cloud.google.com/)​

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\GoogleCloudKey_.json'
translateclient = translate.Client()

DF['Lang_detect_3']=''
DF['Article_original']=DF_ger.Article
for i in range(len(DF)):
    print(i)
    text=DF.Article_original[i].split('.')
    text_df=pd.DataFrame(columns=['chunk'])
    text_df.loc[0]=''
    for item in text:
        if len(text_df.chunk[len(text_df)-1])<25000: #some max of character per request (30000)
            text_df.chunk[len(text_df)-1]=text_df.chunk[len(text_df)-1]+item+'.'
        else:
            text_df.loc[len(text_df)]=item+'.'
    article=''
    for chunk in text_df.chunk:
        translation=translateclient.translate(chunk,target_language='en')
        article=article+html.unescape(translation['translatedText'])+' '
        time.sleep(1) #some max of request by second
    DF.Article[i]=article
    DF.Lang_detect_3[i]=translation['detectedSourceLanguage']

#==============================================================================
# ################################################################ Save results
#==============================================================================

pd.to_pickle(DF,os.path.join(main_path,
                             "0. Data",
                             "Private",
                             "DataFrame_Consolidation_updated_notenglish.pkl"))

DF.to_csv(DF,os.path.join(main_path,
                          "0. Data",
                          "Private",
                          "DataFrame_Consolidation_updated_notenglish.csv"))