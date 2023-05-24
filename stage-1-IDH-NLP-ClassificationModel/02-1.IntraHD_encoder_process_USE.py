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
    tf.config.set_visible_devices(gpus[6], 'GPU')
USE_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3", )

# %%
import pickle

# %%
data_train = pd.read_csv("./dataset/pre-processing/IntraHD_Sentence_internal-train.csv")
# data_train['label'] = [1 if i=='Y' else 0 for i in data_train.label.values.tolist()]
data_valid = pd.read_csv("./dataset/pre-processing/IntraHD_Sentence_internal-test.csv")
# data_valid['label'] = [1 if i=='Y' else 0 for i in data_valid.label.values.tolist()]
print(data_train.shape, data_valid.shape)

# %%
print(len(data_train.label.values)-sum(data_train.label.values), sum(data_train.label.values))
print(len(data_valid.label.values)-sum(data_valid.label.values), sum(data_valid.label.values))

# %%
# pd.set_option('display.max_rows', 2000)
# pd.set_option('display.max_columns', 20)
# data_train[data_train.label==1]

# %%
save_path = "/ssd8/chih/project/yadong/predict_baseline_version02_ACM/dataset/IntraHD_pickle_save_05-17"
save_type_folder = ['USD_sample1124_sequence_fullcontext']
for save_type_folder_ls in save_type_folder:
    for dp in ['','train','valid','test']:
        os.makedirs(os.path.join(save_path, save_type_folder_ls, dp), exist_ok=True)

# %%
USD_training_embeddings=[]
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), 
                 tf.tables_initializer()])
    run_data = data_train['處置其他+症狀處置（描述）'].astype(str).to_list()
    USD_training_embeddings = session.run(USE_model(run_data))
save_folder = save_type_folder[0]
# save_folder = "USD_sample2693_contextsplit"
for idx, (embed_, text_, label_) in enumerate(zip(USD_training_embeddings, data_train['處置其他+症狀處置（描述）'].to_list(),data_train['label'].to_list())):
    with open(os.path.join(save_path, save_folder, "train", "train_USD_internal_pk_data_{}_fixed_re-marked_merge.pkl".format(idx)), "wb") as f:
        save_dict = {"embed": embed_, "text": text_, "label": label_}
        pickle.dump(save_dict, f)

# %%
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), 
                 tf.tables_initializer()])
    run_data = data_valid['處置其他+症狀處置（描述）'].astype(str).to_list()
    USD_validation_embeddings = session.run(USE_model(run_data))
save_folder = save_type_folder[0]
# save_folder = "USD_sample2693_contextsplit"
for idx, (embed_, text_, label_) in enumerate(zip(USD_validation_embeddings, data_valid['處置其他+症狀處置（描述）'].to_list(),data_valid['label'].to_list())):
    with open(os.path.join(save_path, save_folder, "valid", "valid_USD_internal_pk_data_{}_fixed_re-marked_merge.pkl".format(idx)), "wb") as f:
        save_dict = {"embed": embed_, "text": text_, "label": label_}
        pickle.dump(save_dict, f)


