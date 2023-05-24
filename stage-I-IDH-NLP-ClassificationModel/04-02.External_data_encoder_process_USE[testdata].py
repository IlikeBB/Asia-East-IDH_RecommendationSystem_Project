# %%
import os, glob

# %%
import numpy as np, pandas as pd, torch, os, glob, pickle,re
os.environ['TF_CPP_MIN_VLOG_LEVEL']="3"
from sentence_transformers import SentenceTransformer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_hub as hub
import tensorflow_text
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# device = torch.device("cuda:4" if torch.cuda.is_available() else 'cpu')
# BERT_model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device=device)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[3], 'GPU')
USE_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3", )

# %%
save_path = '/ssd8/chih/project/yadong/predict_baseline_version02_ACM/dataset/baseline-testdataset_pickle_save'
save_folder = 'sample_500_seed_42_feature'
os.makedirs(os.path.join(save_path, save_folder), exist_ok=True)

# %% [markdown]
# ## ADD doctor free-text label

# %%
testdata_mark_periHD = pd.read_csv("./dataset/PeriHD_ExternalData_05-17.csv")
testdata_mark_intraHD = pd.read_csv("./dataset/IntraHD_ExternalData_05-17.csv")
testdata_mark_periHD['病人透析時間'] = pd.to_datetime(testdata_mark_periHD['病人透析時間'])
testdata_mark_intraHD['病人透析時間'] = pd.to_datetime(testdata_mark_intraHD['病人透析時間'])
print(testdata_mark_periHD.shape, testdata_mark_intraHD.shape)

# %%
#pickle loading
USD_data_test_pkl = glob.glob("./dataset/baseline-testdataset_pickle_save/sample_500_seed_42/*.pkl")
break_count = 0
for next_pkl in (USD_data_test_pkl):
    save_list = os.listdir(os.path.join(save_path,save_folder))
    with open(next_pkl, 'rb') as f:
        singel_patient_df = pickle.load(f)
        single_patient_periHD = singel_patient_df['PeriHD']
        single_patient_intraHD = singel_patient_df['IntraHD']
        try:
            intraHD_same_time = testdata_mark_intraHD[testdata_mark_intraHD['病人透析時間']==list(single_patient_intraHD['洗腎紀錄時間'])[0]]
            intraHD_answer = intraHD_same_time.label.values[0]
            peirHD_same_time = testdata_mark_periHD[testdata_mark_periHD['病人透析時間']==single_patient_periHD['洗腎紀錄時間']]
            periHD_answer = peirHD_same_time.label.values[0]
        
        except:
            if single_patient_periHD['ID'] == "06821J282":
                periHD_answer = 0
                intraHD_answer = 0
            if single_patient_periHD['ID'] == "7845777":
                periHD_answer = 0
                intraHD_answer = 0
        print("I_ANS: {} P_ANS: {}".format(intraHD_answer, periHD_answer))
        # 轉換IntraHD和PeriHD的斷句特徵
        # IntraHD
        save_name = "baseline_01_sample_500_feature_ID-{}-DateTime-{}.pkl".format(singel_patient_df['ID'], str(singel_patient_df['DateTime']).replace(':','-').replace(' ','-'))
        if save_name not in save_list:
            # break
            IntraHD_data = single_patient_intraHD['處置其他+症狀處置（描述）'].astype(str).to_list()
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), 
                            tf.tables_initializer()])
                IntraHD_feature = session.run(USE_model(IntraHD_data))
            session.close()
            # PeriHD處置其他結束
            text = str(single_patient_periHD['處置其他結束']).replace('\t','').replace('\n','').replace('。','，').replace(',','，')
            text_sp = np.array([i for i in text.split('，') if len(i)>1])
            PeriHD_data = []
            for i in range(len(text_sp)-1):
                PeriHD_data.append(text_sp[i]+'，'+text_sp[i+1])
            if len(text_sp)==1:
                PeriHD_data.append(text_sp[0])
            
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), 
                            tf.tables_initializer()])
                PeriHD_feature = session.run(USE_model(PeriHD_data))
            session.close()
            with open(os.path.join(save_path, save_folder, "baseline_01_sample_500_feature_ID-{}-DateTime-{}.pkl".format(singel_patient_df['ID'], str(singel_patient_df['DateTime']).replace(':','-').replace(' ','-'))), "wb") as f:
                new_pickle_feature = {'ID':singel_patient_df['ID'], 'DateTime': singel_patient_df['DateTime'], 
                                    'PeriHD_data_sbp':single_patient_periHD['NEW開始血壓SBP'],
                                    'PeriHD_data_dbp':single_patient_periHD['NEW開始血壓DBP'],
                                    'PeriHD_Feature': PeriHD_feature, 
                                    'PeriHD_context': PeriHD_data,
                                    'PeriHD_answer': periHD_answer, 
                                    'IntraHD_data_sbp':single_patient_intraHD['sbp'],
                                    'IntraHD_data_dbp':single_patient_intraHD['dbp'],                        
                                    'IntraHD_Feature': IntraHD_feature,
                                    'IntraHD_context': IntraHD_data,
                                    'IntraHD_answer': intraHD_answer,
                                    'IDH_answer':1 if periHD_answer+intraHD_answer!=0 else 0}
                pickle.dump(new_pickle_feature, f)
            del new_pickle_feature, IntraHD_feature, PeriHD_feature
            # break
        else:
            print("{} pass".format(save_name))

# %%



