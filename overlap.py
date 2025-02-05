import pickle
import pandas as pd

with open('output/rf_hr.pickle', 'rb') as h:
    rf_hr = pickle.load(h)

hr_files = ['gpt_binary1536_3-6m',
            'gpt_binary1536_6-12m',
            'gpt_binary1536_12-36m']

ms = [(3,6), (6,12), (12,36)]
mod_hr = {}
i=0
for f in hr_files:
    with open('output/'+'_HR_'+f, 'rb') as h:
        mod_hr[ms[i]] = pickle.load(h)
    i=i+1

rf_files = ['datasets/CA19.csv',
            'datasets/type2diabetes.csv',
            'datasets/pancreatitis.csv']
rows_hr = []
for rf_file in rf_files:
    for m in ms:
        rf_hr_lst = rf_hr[rf_file]['TP'][m] + rf_hr[rf_file]['FP']
        mod_hr_lst = mod_hr[m]['TP'] + mod_hr[m]['FP']
        # print(len(rf_hr_lst))
        # print(len(mod_hr_lst))
        # print(len(set(rf_hr_lst)))
        # print(len(set(mod_hr_lst)))

        common = len(set(rf_hr_lst) & set(mod_hr_lst))
        # print(rf_file)
        # print(m, common/len(set(rf_hr_lst+mod_hr_lst)))

        row_data_hr = {
            "month": m,
            "rf": rf_file,
            "rf_hr_num": len(rf_hr_lst),
            "model_hr_num": len(mod_hr_lst),
            "common_num": common,
            "overlap [%]": common/len(set(rf_hr_lst+mod_hr_lst))*100
        }
        rows_hr.append(row_data_hr)
for m in ms:
    rf_all = []
    for rf_file in rf_files:
        rf_hr_lst = rf_hr[rf_file]['TP'][m] + rf_hr[rf_file]['FP']
        rf_all = list(set(rf_all+rf_hr_lst))
    mod_hr_lst = mod_hr[m]['TP'] + mod_hr[m]['FP']
    common = len(set(rf_all) & set(mod_hr_lst))

    row_data_hr = {
        "month": m,
        "rf": 'all rf',
        "rf_hr_num": len(rf_all),
        "model_hr_num": len(mod_hr_lst),
        "common_num": common,
        "overlap [%]": common/len(set(rf_all+mod_hr_lst))*100
    }
    rows_hr.append(row_data_hr)

df = pd.DataFrame(rows_hr)
df.to_csv('output/overlap.csv', index = None)