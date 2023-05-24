# %%
import pandas as pd, numpy as np
import glob

# %%
def get_newlabel(data_frame):
    # print(data_frame.shape)
    new_lable = []
    for J,S,T in zip(data_frame['JYY_truth'], data_frame['SKH_truth'], data_frame['Tsai_truth']):
        new_lable.append(1 if (J+S+T)>=2 else 0)
    data_frame = data_frame[[i for i in data_frame.columns.to_list() if 'truth'not in i]]
    temp = data_frame.copy()
    del data_frame
    temp.loc[:, 'label'] = new_lable.copy()
    # print(temp.shape)
    return temp

# %%
IntraHD_test = [dir for dir in sorted(glob.glob("./dataset/05-16-new_csv_revise_remark/*.csv")) if "IntraHD_ExternalData" in dir ]
IntraHD_train = [dir for dir in sorted(glob.glob("./dataset/05-16-new_csv_revise_remark/*.csv")) if "IntraHD_InternalData" in dir ]
PeriHD_test = [dir for dir in sorted(glob.glob("./dataset/05-16-new_csv_revise_remark/*.csv")) if "PeriHD_ExternalData" in dir ]
PeriHD_train = [dir for dir in sorted(glob.glob("./dataset/05-16-new_csv_revise_remark/*.csv")) if "PeriHD_InternalData" in dir ]
IntraHD_test = pd.concat(
    [pd.read_csv(IntraHD_test[0]).rename(columns={'truth':"JYY_truth"}),
     pd.read_csv(IntraHD_test[1]).rename(columns={'truth':'SKH_truth'})['SKH_truth'],
     pd.read_csv(IntraHD_test[2]).rename(columns={'truth':'Tsai_truth'})['Tsai_truth']
     ],axis=1)
IntraHD_test = IntraHD_test[sorted(IntraHD_test.columns)]
IntraHD_test = get_newlabel(IntraHD_test)
IntraHD_test.to_csv("./dataset/IntraHD_ExternalData_05-17.csv", index=False, encoding='utf-8-sig')

IntraHD_train = pd.concat(
    [pd.read_csv(IntraHD_train[0]).rename(columns={'truth':"JYY_truth"}),
     pd.read_csv(IntraHD_train[1]).rename(columns={'truth':'SKH_truth'})['SKH_truth'],
     pd.read_csv(IntraHD_train[2]).rename(columns={'truth':'Tsai_truth'})['Tsai_truth']
     ],axis=1)
IntraHD_train = IntraHD_train[sorted(IntraHD_train.columns)]
IntraHD_train = get_newlabel(IntraHD_train)
IntraHD_train.to_csv("./dataset/IntraHD_InternalData_05-17.csv", index=False, encoding='utf-8-sig')

PeriHD_test = pd.concat(
    [pd.read_csv(PeriHD_test[0]).rename(columns={'truth':"JYY_truth"}),
     pd.read_csv(PeriHD_test[1]).rename(columns={'truth':'SKH_truth'})['SKH_truth'],
     pd.read_csv(PeriHD_test[2]).rename(columns={'truth':'Tsai_truth'})['Tsai_truth']
     ],axis=1)
PeriHD_test = PeriHD_test[sorted(PeriHD_test.columns)]
PeriHD_test = get_newlabel(PeriHD_test)
PeriHD_test.to_csv("./dataset/PeriHD_ExternalData_05-17.csv", index=False, encoding='utf-8-sig')

PeriHD_train = pd.concat(
    [pd.read_csv(PeriHD_train[0]).rename(columns={'truth':"JYY_truth"}),
     pd.read_csv(PeriHD_train[1]).rename(columns={'truth':'SKH_truth'})['SKH_truth'],
     pd.read_csv(PeriHD_train[2]).rename(columns={'truth':'Tsai_truth'})['Tsai_truth']
     ],axis=1)
PeriHD_train = PeriHD_train[sorted(PeriHD_train.columns)]
PeriHD_train = get_newlabel(PeriHD_train)
PeriHD_train.to_csv("./dataset/PeriHD_InternalData_05-17.csv", index=False, encoding='utf-8-sig')


# %%



