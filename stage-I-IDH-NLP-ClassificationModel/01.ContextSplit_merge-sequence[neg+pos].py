# %%
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

# %%
get_2693_sample_df = pd.read_csv('./dataset/PeriHD_InternalData_05-17.csv')
get_2693_sample_df
positive_context_split_csv = (get_2693_sample_df[get_2693_sample_df.label==1])['處置其他結束（描述）'].values.tolist()
print(len(positive_context_split_csv))


concat1 = pd.read_csv('/ssd8/chih/project/yadong/process_csv/PeriHD_not_markdata-sampe2000.csv')
concat2 = pd.read_csv('/ssd8/chih/project/yadong/mark_csv/train_peri_mark_sample_693_JYY.csv')
peri_filter_dataset = pd.concat([concat1, concat2])
# peri_filter_dataset = peri_filter_dataset['處置其他結束（描述）']
# peri_693_dataset = pd.read_csv('../mark_csv/train_peri_mark_sample_693_JYY.csv')[['truth', '處置其他結束（描述）']]
# peri_filter_dataset = peri_693_dataset[peri_693_dataset['truth']=='Y']
print(peri_filter_dataset.shape)

# %%
# 先切割標記後的資料以確保訓練和驗證資料無重疊
keyword_col = []
for seq in peri_filter_dataset['處置其他結束（描述）'].values.tolist(): #逐筆搜尋pos
    seq_sp = str(seq).replace("。","，").replace(",","，").replace("\n","").split('，')
    temp_keyword = ''
    for pos in positive_context_split_csv:
        if str(pos) in seq_sp:
            temp_keyword = temp_keyword + str(pos)+'，'
    if len(temp_keyword)<1:
        temp_keyword = 'Negative_Sequence'
    keyword_col.append(temp_keyword)
    # break

# %%
peri_filter_dataset['KeyWord_dict'] = keyword_col
peri_filter_dataset = peri_filter_dataset[['處置其他結束（描述）', 'KeyWord_dict']]
peri_filter_dataset['label'] = [0 if i =='Negative_Sequence' else 1 for i in peri_filter_dataset.KeyWord_dict.tolist()]
data_train, data_test = train_test_split(peri_filter_dataset, test_size=0.1, random_state=42, stratify=peri_filter_dataset['label'])

# %%
data_train

# %%
merage_sentence = []
for seq in range(len(data_test)):
    if True:
        rep_seq = data_test.iloc[seq]['處置其他結束（描述）']
        rep_seq = str(rep_seq).replace("。","，").replace(",","，").replace("\n","").replace(";","，").replace("；","，").split('，')
        key_word = data_test.iloc[seq]['KeyWord_dict']
        key_word = str(key_word).replace("。","，").replace(",","，").replace("\n","").replace(";","，").replace("；","，").split('，')
        key_word = [x for x in key_word if x != ""]
        if len(rep_seq)!=1:
            for i in range(len(rep_seq)-1):
                new_sentence = rep_seq[i]+"，"+rep_seq[i+1]
                if (rep_seq[i] in key_word) or (rep_seq[i+1] in key_word):
                    merage_sentence.append(['Y',new_sentence])
                elif len(rep_seq[i+1])>0:
                    merage_sentence.append(['N',new_sentence])
        # else:
        #     print('pass')
        #     if rep_seq in key_word:
        #         merage_sentence.append(['Y-pass',new_sentence])
        #     else:
        #         merage_sentence.append(['N-pass',new_sentence])

# %%
(pd.DataFrame(merage_sentence, columns=['Truth', '處置其他結束（描述）']).drop_duplicates(subset=['處置其他結束（描述）'])).to_csv("./dataset/pre-processing/PeriHD_Sentence-merge_internal-test[pos+neg].csv", encoding='utf-8-sig', index=False)

# %%
merage_sentence = []
for seq in range(len(data_train)):
    rep_seq = data_train.iloc[seq]['處置其他結束（描述）']
    rep_seq = str(rep_seq).replace("。","，").replace(",","，").replace("\n","").replace(";","，").replace("；","，").split('，')
    key_word = data_train.iloc[seq]['KeyWord_dict']
    key_word = str(key_word).replace("。","，").replace(",","，").replace("\n","").replace(";","，").replace("；","，").split('，')
    key_word = [x for x in key_word if x != ""]
    if len(rep_seq)!=1:
        for i in range(len(rep_seq)-1):
            new_sentence = rep_seq[i]+"，"+rep_seq[i+1]
            if (rep_seq[i] in key_word) or (rep_seq[i+1] in key_word):
                merage_sentence.append(['Y',new_sentence])
            elif len(rep_seq[i+1])>0:
                merage_sentence.append(['N',new_sentence])
    # if seq==10:
    #     break

# %%
len(merage_sentence)

# %%
(pd.DataFrame(merage_sentence, columns=['Truth', '處置其他結束（描述）']).drop_duplicates(subset=['處置其他結束（描述）'])).to_csv("./dataset/pre-processing/PeriHD_Sentence-merge_internal-train[pos+neg].csv", encoding='utf-8-sig', index=False)

# %%
get_1124_sample_df = pd.read_csv("./dataset/IntraHD_InternalData_05-17.csv")
# get_1124_sample_df = get_1124_sample_df.rename(columns={'truth':'label'})
get_1124_sample_df = get_1124_sample_df.dropna()
print(get_1124_sample_df.shape)
data_train, data_test = train_test_split(get_1124_sample_df, test_size=0.1, random_state=42, stratify=get_1124_sample_df['label'])
data_train.to_csv("./dataset/pre-processing/IntraHD_Sentence_internal-train.csv", encoding='utf-8-sig', index=False)
data_test.to_csv("./dataset/pre-processing/IntraHD_Sentence_internal-test.csv", encoding='utf-8-sig', index=False)
print(data_train.shape, data_test.shape)

# %%
# pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_columns', 20)
# data_test

# %%



