# %%
import numpy as np, pandas as pd, torch,os
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
    tf.config.set_visible_devices(gpus[0], 'GPU')
USE_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3", )

# %%
import pickle

# %%
# data_train, data_test = train_test_split(new_fix_get_2693_sample_df, test_size=0.1, random_state=42, stratify=new_fix_get_2693_sample_df['label'])
# data_train, data_test = train_test_split(get_2693_sample_df, test_size=0.1, random_state=42, stratify=get_2693_sample_df['label'])
data_train = pd.read_csv("./dataset/pre-processing/PeriHD_Sentence-merge_internal-train[pos+neg].csv")
data_train['label'] = [1 if i=='Y' else 0 for i in data_train.Truth.values.tolist()]
data_valid = pd.read_csv("./dataset/pre-processing/PeriHD_Sentence-merge_internal-test[pos+neg].csv")
data_valid['label'] = [1 if i=='Y' else 0 for i in data_valid.Truth.values.tolist()]
print(data_train.shape, data_valid.shape)

# %%
print(len(data_train.label.values)-sum(data_train.label.values), sum(data_train.label.values))
print(len(data_valid.label.values)-sum(data_valid.label.values), sum(data_valid.label.values))

# %%
# pd.set_option('display.max_rows', 2000)
# pd.set_option('display.max_columns', 20)
# data_train[data_train.label==1]

# %%
save_path = "/ssd8/chih/project/yadong/predict_baseline_version02_ACM/dataset/PeriHD_pickle_save_05-17"
save_type_folder = ['USD_sample2693_only-merge-sequence_contextsplit']
for save_type_folder_ls in ['USD_sample2693_only-merge-sequence_contextsplit']:
    for dp in ['','train','valid','test']:
        os.makedirs(os.path.join(save_path, save_type_folder_ls, dp), exist_ok=True)

# %%
USD_training_embeddings=[]
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), 
                 tf.tables_initializer()])
    run_data = data_train['處置其他結束（描述）'].astype(str).to_list()
    USD_training_embeddings = session.run(USE_model(run_data))
save_folder = save_type_folder[0]
# save_folder = "USD_sample2693_contextsplit"
for idx, (embed_, text_, label_) in enumerate(zip(USD_training_embeddings, data_train['處置其他結束（描述）'].to_list(),data_train['label'].to_list())):
    with open(os.path.join(save_path, save_folder, "train", "train_USD_internal_pk_data_{}_fixed_re-marked_merge.pkl".format(idx)), "wb") as f:
        save_dict = {"embed": embed_, "text": text_, "label": label_}
        pickle.dump(save_dict, f)

# %%
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), 
                 tf.tables_initializer()])
    run_data = data_valid['處置其他結束（描述）'].astype(str).to_list()
    USD_validation_embeddings = session.run(USE_model(run_data))
save_folder = save_type_folder[0]
# save_folder = "USD_sample2693_contextsplit"
for idx, (embed_, text_, label_) in enumerate(zip(USD_validation_embeddings, data_valid['處置其他結束（描述）'].to_list(),data_valid['label'].to_list())):
    with open(os.path.join(save_path, save_folder, "valid", "valid_USD_internal_pk_data_{}_fixed_re-marked_merge.pkl".format(idx)), "wb") as f:
        save_dict = {"embed": embed_, "text": text_, "label": label_}
        pickle.dump(save_dict, f)

# %%
# save_path = "/ssd8/chih/project/yadong/predict_baseline_version01/dataset/PeriHD_pickle_save"
# save_type_folder = ['USD_sample2693_only-merge-sequence_fullcontext']
# for save_type_folder_ls in save_type_folder:
#     for dp in ['','train','test']:
#         os.makedirs(os.path.join(save_path, save_type_folder_ls, dp), exist_ok=True)

# %%
# data_train = pd.read_csv("./dataset/PeriHD_Sample-2693_Sentence-merge_internal-train.csv")
# data_train['label'] = [1 if i=='Y' else 0 for i in data_train.Truth.values.tolist()]
# data_valid = pd.read_csv("./dataset/PeriHD_Sample-2693_Sentence-merge_internal-test.csv")
# data_valid['label'] = [1 if i=='Y' else 0 for i in data_valid.Truth.values.tolist()]
# merge_df = pd.concat([data_train, data_valid])
# concat1 = pd.read_csv('/ssd8/chih/project/yadong/process_csv/PeriHD_not_markdata-sampe2000.csv')['處置其他結束（描述）']
# concat2 = pd.read_csv('/ssd8/chih/project/yadong/mark_csv/train_peri_mark_sample_693_JYY.csv')['處置其他結束（描述）']
# concat_df = pd.concat([concat1, concat2])
# positive_df = merge_df[merge_df['label']==1]
# print(concat_df.shape)
# print(positive_df.shape)

# %%
# ans_list = []
# for full_text in concat_df.tolist():
#     ans_temp = 0
#     text = str(full_text).replace('\t','').replace('\n','').replace('。','，').replace(',','，')
#     # text_sp = np.array([i for i in text.split('，') if len(i)>0])
#     for pos_text in positive_df['處置其他結束（描述）'].tolist():
#         if str(pos_text) in text:
#             # print(text)
#             # print([str(pos_text)])
#             # break
#             ans_temp+=1
#             break
#     # break
#     if ans_temp!=0:
#         ans_list.append(1)
#     else:
#         ans_list.append(0)

# %%
# save_folder = save_type_folder[0]

# %%
# save_folder = save_type_folder[0]
# for idx, (text, ans) in enumerate(zip(concat_df.tolist(), ans_list)):
#     text = str(text).replace('\t','').replace('\n','').replace('。','，').replace(',','，')
#     text_sp = np.array([i for i in text.split('，') if len(i)>0])
#     context_stack = []
#     for i in range(len(text_sp)-1):
#         context_stack.append(text_sp[i]+'，'+text_sp[i+1])
#     if len(text_sp)==1:
#         context_stack.append(text_sp[0])
#     with tf.Session() as session:
#         session.run([tf.global_variables_initializer(), 
#                     tf.tables_initializer()])
#         temp_embed = session.run(USE_model(context_stack))
#     with open(os.path.join(save_path, save_folder, "train", "train_USD_external_pk_data_{}.pkl".format(idx)), "wb") as f:
#         save_dict = {"full_embed": temp_embed, "full_text": context_stack, "label": ans}
#         pickle.dump(save_dict, f)
#     # break

# %%
# external_data_test = pd.read_csv("./dataset/baseline_PeriHD_notmark_pred_100sample_JYY.csv", encoding='utf-8-sig')
# external_data_test['Truth'] = [0 if i=='N' else 1 for i in external_data_test['Truth']]
# external_data_test = external_data_test.rename(columns={'Truth': 'label'})

# save_folder = save_type_folder[0]
# for idx, (text, ans) in enumerate(zip(external_data_test['處置其他結束（描述）'].tolist(), external_data_test['label'].tolist())):
#     text = str(text).replace('\t','').replace('\n','').replace('。','，').replace(',','，')
#     text_sp = np.array([i for i in text.split('，') if len(i)>0])
#     context_stack = []
#     for i in range(len(text_sp)-1):
#         context_stack.append(text_sp[i]+'，'+text_sp[i+1])
#     if len(text_sp)==1:
#         context_stack.append(text_sp[0])
#     with tf.Session() as session:
#         session.run([tf.global_variables_initializer(), 
#                     tf.tables_initializer()])
#         temp_embed = session.run(USE_model(context_stack))
#     # print(len(context_stack))
#     # print(temp_embed.shape)
#     with open(os.path.join(save_path, save_folder, "test", "train_USD_external_pk_data_{}.pkl".format(idx)), "wb") as f:
#         save_dict = {"full_embed": temp_embed, "full_text": context_stack, "label": ans}
#         pickle.dump(save_dict, f)

# %%



