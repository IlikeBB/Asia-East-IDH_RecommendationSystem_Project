# %%
import numpy as np, pandas as pd, torch,os
from sentence_transformers import SentenceTransformer, util
import tensorflow_hub as hub
import tensorflow_text
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
ST_BERT = text_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
model_name_s = "ST-DBMCv1"
save_type_folder = [f'{model_name_s}_sample2693_only-merge-sequence_contextsplit']
for save_type_folder_ls in save_type_folder:
    for dp in ['','train','valid','test']:
        os.makedirs(os.path.join(save_path, save_type_folder_ls, dp), exist_ok=True)

# %%
BERT_training_embeddings=[]
run_data = data_train['處置其他結束（描述）'].astype(str).to_list()
BERT_training_embeddings = ST_BERT.encode(run_data)
save_folder = save_type_folder[0]
# save_folder = "USD_sample2693_contextsplit"
for idx, (embed_, text_, label_) in enumerate(zip(BERT_training_embeddings, data_train['處置其他結束（描述）'].to_list(),data_train['label'].to_list())):
    with open(os.path.join(save_path, save_folder, "train", "train_{}_internal_pk_data_{}_fixed_re-marked_merge.pkl".format(model_name_s,idx)), "wb") as f:
        save_dict = {"embed": embed_, "text": text_, "label": label_}
        pickle.dump(save_dict, f)

# %%
BERT_validation_embeddings=[]
run_data = data_valid['處置其他結束（描述）'].astype(str).to_list()
BERT_validation_embeddings = ST_BERT.encode(run_data)
save_folder = save_type_folder[0]
# save_folder = "USD_sample2693_contextsplit"
for idx, (embed_, text_, label_) in enumerate(zip(BERT_validation_embeddings, data_valid['處置其他結束（描述）'].to_list(),data_valid['label'].to_list())):
    with open(os.path.join(save_path, save_folder, "valid", "valid_{}_internal_pk_data_{}_fixed_re-marked_merge.pkl".format(model_name_s,idx)), "wb") as f:
        save_dict = {"embed": embed_, "text": text_, "label": label_}
        pickle.dump(save_dict, f)

# %%



