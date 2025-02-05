import pickle
import pandas as pd
from collections import Counter
import numpy as np
from datetime import timedelta
from collections import defaultdict
from tqdm import tqdm

# rf_path = 'datasets/type2diabetes.csv'
rf_paths = ['datasets/CA19.csv','datasets/type2diabetes.csv', 'datasets/pancreatitis.csv']

pat_data_path = 'datasets/pat_data.pkl'

##############data load###################
with open(pat_data_path, 'rb') as h:
    pat_dat = pickle.load(h)
with open('datasets/pc_diag.pickle', 'rb') as h:
    pc_diag = pickle.load(h)

gaps = [(0,3), (3,6), (6,12), (12,36), (36, np.inf)]
ctrl, case = [], {}
for gap in gaps:
    case[gap] = []
for k,v in pat_dat.items():
    if sum(v['label']) ==0:
        ctrl.append(k)
    else:
        for i in range(5):
            if sum(v['label'])==5-i:
                case[gaps[i]].append(k)

print('number of cases and controls')
print('control:', len(ctrl))
for gap in gaps:
    print(gap, len(case[gap]))
#############RF###################
hr = {}
for rf_path in rf_paths:
    hr[rf_path] = {}
    print(rf_path.split('/')[1])
    rf = pd.read_csv(rf_path)
    rf.columns = ['person_id', 'date', 'concept_id']
    rf['date'] = pd.to_datetime(rf['date'], errors = 'coerce')

    print('id_type:', Counter(rf['concept_id']))

    ################count individuals with risk factor#################
    rf['sorted_date'] = rf.sort_values(by='date').groupby('person_id')['date'].transform('first')
    rf_dict = rf.drop_duplicates(subset='person_id').set_index('person_id')['sorted_date'].to_dict()

    ctrl_rf = list(set(ctrl) & set(list(rf_dict.keys())))
    print('ctrl with risk factor (FP):', len(ctrl_rf), len(ctrl_rf)/len(ctrl))


    ########################TP############################
    before = defaultdict(list)
    after = []

    for k, v in rf_dict.items():
        if k in pc_diag:  # No need to convert to a list
            diff = pc_diag[k] - v
            for gap in gaps:
                # Check if the difference in days falls within the current gap
                if gap[0] * 30 <= diff.days <= gap[1] * 30:
                    before[gap].append(k)
                    break
            else:  # Only execute if the loop wasn't broken (i.e., diff.days < 0)
                if diff.days < 0:
                    after.append(k)

    print('TP: ', {gap:len(before[gap]) for gap in gaps})
    print('PPV: ', {gap:len(before[gap])/(len(before[gap])+len(ctrl_rf)) for gap in gaps})
    hr[rf_path]['TP'] = before
    hr[rf_path]['FP'] = ctrl_rf
with open('output/rf_hr.pickle', 'wb') as h:
    pickle.dump(hr, h)
