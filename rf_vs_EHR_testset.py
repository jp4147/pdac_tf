import pickle
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np
from training import Training
from load_data import DataLoader
import torch
from collections import Counter
from sklearn.metrics import precision_recall_curve, roc_curve
import os

def find_optimal_threshold(precision, recall, thresholds):
    # Example: Choose the threshold that maximizes the F1 score
    f1_scores = 2*(precision*recall) / (precision+recall)
    f1_scores = np.nan_to_num(f1_scores)
    max_f1_index = np.argmax(f1_scores[:-1])  # exclude last value which corresponds to recall=0
    optimal_threshold = thresholds[max_f1_index]
    return optimal_threshold, f1_scores[max_f1_index]

def find_maxPPV_threshold(precision, recall, thresholds):
    max_precision_idx = np.argmax(precision[:-1])
    optimal_threshold = thresholds[max_precision_idx]
    return optimal_threshold, max_precision_idx

file_path = input("Please provide the file location where the list pickle containing person_ids with variants is saved: ").strip().strip("'").strip('"')
# Validate the file path
while not os.path.exists(file_path) or not file_path.endswith(".pickle"):
    print("Invalid file path. Please enter a valid path to a .pickle file.")
    file_path = input("Please provide the file location where the list pickle containing person_ids with variants is saved: ").strip().strip("'").strip('"')

# Load the pickle file
try:
    with open(file_path, "rb") as f:
        person_ids_variant = pickle.load(f)
    print(f"Successfully loaded {len(person_ids_variant)} person_ids from the pickle file.")

except Exception as e:
    print(f"Error loading the pickle file: {e}")

optimal_threshold_criteron = 'f1'
one_rf_tps = {}
for emb in ['gpt', 'base']:
    one_rf_tps[emb] = {}
    for model_type in ['binary_6-12m']:
    # for model_type in ['multi', 'multi_rm03', 'binary_3-6m', 'binary_6-12m', 'binary_12-36m']:
        testset_path = 'datasets/'+model_type+'_testset.pickle'
        rf_paths = ['datasets/CA19.csv','datasets/type2diabetes.csv', 'datasets/pancreatitis.csv']
        with open('datasets/pc_diag.pickle', 'rb') as h:
            pc_diag = pickle.load(h)
        ##############DATA PATH###############
        if model_type == 'multi':
            pat_data_path = 'datasets/pat_data.pkl' #multi
            if emb=='gpt':
                model_rs_path = 'output/rs_tf_gpt1536_dx.pickle' # risk scores from multi-label classification GPT embedding model including 0-3 month data 
            else:
                model_rs_path = 'output/rs_tf1536_dx.pickle'
            intervals = ['3m', '6m', '12m', '36m', '60m']
            month2idx = {'3m':0, '6m':1, '12m':2, '36m':3, '60m':4}
            la2cat = {0:'ctrl', 1:(36, np.inf), 2:(12,36), 3:(6,12), 4:(3,6), 5:(0,3)}
            gaps = [(0,3), (3,6), (6,12), (12,36), (36, np.inf)]
        if model_type == 'multi_rm03':
            pat_data_path = 'datasets/pat_data_rev.pkl' #rm03
            if emb=='gpt':
                model_rs_path = 'output/rs_gpt1536.pickle'
            else:
                model_rs_path = 'output/rs_baseline1536.pickle'
            intervals = ['6m', '12m', '36m', '60m']
            month2idx = {'6m':0, '12m':1, '36m':2, '60m':3}
            la2cat = {0:'ctrl', 1:(36, np.inf), 2:(12,36), 3:(6,12), 4:(3,6)}
            gaps = [(3,6), (6,12), (12,36), (36, np.inf)]
        if model_type in ['binary_3-6m', 'binary_6-12m', 'binary_12-36m']:
            months_prior = model_type.split('_')[-1]
            pat_data_path = 'datasets/pat_dat_aug'+str(months_prior)+'.pickle' #binary
            if emb == 'gpt':
                model_rs_path = 'output/gpt_binary1536_'+months_prior+'.pickle'
            else:
                model_rs_path = 'output/baseline_binary1536_'+months_prior+'.pickle'
            s = months_prior.split('-')[0]
            e = months_prior.split('-')[1].split('m')[0]
            gaps = [(int(s), int(e))]
        ##############TEST SET###################
        if os.path.exists(testset_path):
            print(testset_path+' exist')
            with open(testset_path, 'rb') as h:
                pat_dat = pickle.load(h)
        else:
            print('creating and saving testset')
            with open(pat_data_path, 'rb') as h:
                pat_dat = pickle.load(h)
            data_loader = DataLoader(pat_data_path, use_graph_embeddings = False)  
            data = data_loader.reidx_dat
            pat_c2i = data_loader.pat_c2i
            vocab_size = data_loader.vocab_size

            training = Training(data, save_path=None)
            train_ids, val_ids, test_ids = training.data_splits.split_ids()

            pat_dat = {i:pat_dat[i] for i in test_ids}
            with open(testset_path, 'wb') as h:
                pickle.dump(pat_dat, h)

        test_ids = list(pat_dat.keys())
        print('testset')
        if 'binary' in model_type:
            case  = [i for i in test_ids if pat_dat[i]['label']==1]
            ctrl  = [i for i in test_ids if pat_dat[i]['label']==0]
            print('case:', len(case), 'ctrl:', len(ctrl))
        else:
            test_ids_cat = {}
            for i in la2cat.values():
                test_ids_cat[i] = []
            for i in test_ids:
                la = sum(pat_dat[i]['label'])
                test_ids_cat[la2cat[la]].append(i)
            case = [v for k, v in test_ids_cat.items() if k!='ctrl']
            case = [item for sublist in case for item in sublist]
            print({k:len(v) for k,v in test_ids_cat.items()})
        ################RISK FACTOR#################
        rf_res = {}
        rf_tps, rf_fps = {}, {}
        for gap in gaps:
            one_rf_tps[emb][gap], rf_tps[gap] = [], []
        for rf_path in rf_paths:
            rf = pd.read_csv(rf_path)
            rf.columns = ['person_id', 'date', 'concept_id']
            rf['date'] = pd.to_datetime(rf['date'], errors = 'coerce')

            rf['sorted_date'] = rf.sort_values(by='date').groupby('person_id')['date'].transform('first')
            rf_dict = rf.drop_duplicates(subset='person_id').set_index('person_id')['sorted_date'].to_dict()

            before = defaultdict(list)
            after = []
            for k, v in rf_dict.items():
                if k in case:  # No need to convert to a list
                    diff = pc_diag[k] - v
                    for gap in gaps:
                        # Check if the difference in days falls within the current gap
                        # if gap[0] * 30 <= diff.days <= gap[1] * 30:
                        if gap[0] * 30 <= diff.days:
                            before[gap].append(k)
                            break
                    else:  # Only execute if the loop wasn't broken (i.e., diff.days < 0)
                        if diff.days < 0:
                            after.append(k)

            # print('risk factor:', rf_path)
            if 'binary' in model_type:
                fp = len(set(list(rf['person_id'])) & set(ctrl))
                rf_fps[rf_path] = set(list(rf['person_id'])) & set(ctrl)
            else:       
                fp = len(set(list(rf['person_id'])) & set(test_ids_cat['ctrl']))
                rf_fps[rf_path] = set(list(rf['person_id'])) & set(test_ids_cat['ctrl'])
            # print('ctrl:', fp)
            rf_name = rf_path.split('/')[-1].split('.')[0]
            rf_res[rf_name] = []

            for gap in gaps:
                if 'binary' in model_type:
                    tp = len(set(before[gap]) & set(case))
                    rf_tp_ids = list(set(before[gap]) & set(case))
                else:
                    tp = len(set(before[gap]) & set(test_ids_cat[gap]))
                    rf_tp_ids = list(set(before[gap]) & set(test_ids_cat[gap]))
                if len(rf_tps[gap])==0:
                    rf_tps[gap] = list(set(rf_tps[gap] + rf_tp_ids))
                else:
                    rf_tps[gap] = list(set(rf_tps[gap]) & set(rf_tp_ids))
                one_rf_tps[emb][gap] = list(set(one_rf_tps[emb][gap]+rf_tp_ids))
                # print(interval, tp)
                rf_res[rf_name] = rf_res[rf_name]+[tp, fp, tp/(tp+fp)]
            
        rf_fp_combined = []
        for k, v in rf_fps.items():
            if len(rf_fp_combined) == 0:
                rf_fp_combined.extend(v)
            else:
                rf_fp_combined = list(set(rf_fp_combined) & set(v))
            
        rf_fp_combined = list(set(rf_fp_combined))
        rf_res['combined'], case_ctrl = [], []
        for gap in gaps:
            rf_res['combined'].extend([len(rf_tps[gap]),len(rf_fp_combined),
                                       len(rf_tps[gap])/(len(rf_tps[gap])+len(rf_fp_combined))])
            if 'binary' in model_type:
                case_ctrl.extend([(len(case), len(ctrl))]*3)
            else:
                case_ctrl.extend([(len(test_ids_cat[gap]), len(test_ids_cat['ctrl']))]*3)

        rf_res['variant'] = []
        # rf_res['variant_rf_combined'] = []
        rf_variant_tps = {}
        for gap in gaps:
            if 'binary' in model_type:
                fp_list = list(set(person_ids_variant) & set(ctrl))
                fp = len(fp_list)
                tp = len(set(person_ids_variant) & set(case))
                variant_tp_ids = list(set(person_ids_variant) & set(case))
            else:
                fp_list = list(set(person_ids_variant) & set(test_ids_cat['ctrl']))
                fp = len(fp_list)
                tp = len(set(person_ids_variant) & set(test_ids_cat[gap]))
                variant_tp_ids = list(set(person_ids_variant) & set(test_ids_cat[gap]))
            if tp+fp==0:
                rf_res['variant'].extend([tp, fp, np.nan])
            else:
                rf_res['variant'].extend([tp, fp, tp/(tp+fp)])
            
            # rf_variant_tps[gap] = list(set(rf_tps[gap]+variant_tp_ids))
            # tp = len(rf_variant_tps[gap])
            # fp = len(set(fp_list+rf_fp_combined))
            # if tp+fp==0:
            #     rf_res['variant_rf_combined'].extend([tp, fp, np.nan]) 
            # else:
            #     rf_res['variant_rf_combined'].extend([tp, fp, tp/(tp+fp)])               
        #####################EHR MODEL#####################
        model_res = []
        model_tp_ids = {}
        with open(model_rs_path, 'rb') as handle:
            model_rs = pickle.load(handle)

        if 'binary' in model_type:
            ids = model_rs['ids']
            la = model_rs['labels']
            rs_list = model_rs['raw_scores']
            precision, recall, thresholds = precision_recall_curve(la, rs_list)

            if optimal_threshold_criteron == 'f1':
                optimal_threshold, max_f1_score= find_optimal_threshold(precision, recall, thresholds)
            elif optimal_threshold_criteron == 'ppv':
                optimal_threshold, _ = find_maxPPV_threshold(precision, recall, thresholds)

            predictions = [1 if score >= optimal_threshold else 0 for score in rs_list]

            # Calculate True Positives (TP) and False Positives (FP)
            model_tp_ids[gaps[0]] = [id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 1]
            tp = len(model_tp_ids[gaps[0]])
            fp = len([id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 0])
            model_res = [tp,fp,tp/(tp+fp)]
        else:
            for idx, m in enumerate(intervals):
                true_label = np.array(model_rs[m]['labels'])[:,month2idx[m]]
                model_score =np.array(model_rs[m]['raw_scores'])[:,month2idx[m]]
                ids = np.array(model_rs[m]['ids'])
                precision, recall, pr_thre = precision_recall_curve(true_label,model_score)

                if optimal_threshold_criteron == 'f1':
                    optimal_threshold, max_f1_score= find_optimal_threshold(precision, recall, pr_thre)
                elif optimal_threshold_criteron == 'ppv':
                    optimal_threshold, _ = find_maxPPV_threshold(precision, recall, thresholds)
                predictions = [1 if score >= optimal_threshold else 0 for score in model_score]
                model_tp_ids[gaps[idx]] = [id for pred, true, id in zip(predictions, true_label, ids) if pred == 1 and true == 1]
                tp = len(model_tp_ids[gaps[idx]])
                fp = len([id for pred, true, id in zip(predictions, true_label, ids) if pred == 1 and true == 0])
                # print(m)
                # print('TP:', tp)
                # print('FP:', fp)
                # print('PPV:', tp/(tp+fp))
                model_res.append([tp,fp,tp/(tp+fp)])
            model_res = [item for sublist in model_res for item in sublist]

        overlap_rf_model = []
        overlap_rf_variant = []
        overlap_model_variant = []
        for gap in gaps:
            overlap_rf_model.extend([len(set(one_rf_tps[emb][gap]) & set(model_tp_ids[gap])), np.nan, np.nan])
            overlap_rf_variant.extend([len(set(one_rf_tps[emb][gap]) & set(variant_tp_ids)), np.nan, np.nan])
            # print(gap, 'rf tp:', len(rf_tps[gap]), 'model tp:', len(model_tp_ids[gap]))
            # print('overlap:', len(set(rf_tps[gap]) & set(model_tp_ids[gap])))
            overlap_model_variant.extend([len(set(variant_tp_ids) & set(model_tp_ids[gap])), np.nan, np.nan])
        row_names = ['case and ctrl']+[model_type]+list(rf_res.keys())+['overlap_rf_model']+['overlap_rf_variant']+['overlap_variant_model']
        sub_cat = []
        for gap in gaps:
            for v in ['TP', 'FP', 'PPV']:
                sub_cat.append((gap,v))
        columns = pd.MultiIndex.from_tuples(sub_cat)
        df = pd.DataFrame([case_ctrl]+[model_res]+list(rf_res.values())+[overlap_rf_model]+[overlap_rf_variant]+[overlap_model_variant], columns=columns, index = row_names)

        df.to_csv('output/clinical_utility_'+model_type+'_'+emb+'.csv')

def FP2screen(thresholds_pr, la, ids, rs_list, percent_TP=0.5):
    print('Total TP:', sum(la))
    target_tp = sum(la) * percent_TP  # Precompute target TP for clarity
    print(f'Target TP: {percent_TP * 100:.0f}%, {target_tp:.0f}')

    rs_list = np.concatenate(rs_list)
    start = 0
    predictions = rs_list >= thresholds_pr[start]  # Vectorized thresholding
    tp = np.sum((predictions == 1) & (np.array(la) == 1))  # Count TP using NumPy

    while (tp-target_tp)>50:
        start = start+10000
        predictions = rs_list >= thresholds_pr[start]  # Vectorized thresholding
        tp_old = tp
        tp = np.sum((predictions == 1) & (np.array(la) == 1))  # Count TP using NumPy
        if tp_old!=tp:
            print('>50', tp)
    if (tp-target_tp)<0:
        start = start-10000
        tp = tp_old

    while (tp-target_tp)>10:
        start = start+2000
        predictions = rs_list >= thresholds_pr[start]  # Vectorized thresholding
        tp_old = tp
        tp = np.sum((predictions == 1) & (np.array(la) == 1))  # Count TP using NumPy
        if tp_old!=tp:
            print('>10', tp)
    if (tp-target_tp)<0:
        start = start-2000
        tp = tp_old

    while (tp-target_tp)>3:
        start = start+1000
        predictions = rs_list >= thresholds_pr[start]  # Vectorized thresholding
        tp_old=tp
        tp = np.sum((predictions == 1) & (np.array(la) == 1))  # Count TP using NumPy
        if tp_old!=tp:
            print('>3', tp)
    if (tp-target_tp)<0:
        start = start-1000
        tp = tp_old

    print('searching almost done')
    while (tp-target_tp)>=0:
        start = start+1
        predictions = rs_list >= thresholds_pr[start]  # Vectorized thresholding
        true_positive_mask = (predictions == 1) & (np.array(la) == 1)
        tp_old=tp
        tp = np.sum(true_positive_mask)  # Count TP using NumPy
        if tp_old!=tp:
            print('>0', tp)
    fp = np.sum((predictions == 1) & (np.array(la) == 0))
    tp_ids = ids[true_positive_mask]
    print('COMPLETE:', tp, tp+fp, (tp+fp)/len(la))
    
    return start, target_tp, tp, fp, tp_ids


data_paths= ['output/baseline_binary1536_6-12m.pickle',
             'output/gpt_binary1536_6-12m.pickle']
rows = []

for data_path in data_paths:
    with open(data_path, 'rb') as h:
        rs = pickle.load(h)

    ids = np.array(rs['ids'])
    la = rs['labels']
    rs_list = rs['raw_scores']
    emb_type = data_path.split('/')[1].split('_')[0][0:4]
    precision, recall, thresholds = precision_recall_curve(la, rs_list)
    s = 0
    name = data_path.split('/')[1].split('_')[0]
    rows.append([name])
    for p in [0.8, 0.5, 0.2]:
        start, target_tp, tp, fp, tp_ids = FP2screen(thresholds[s:], la, ids, rs_list, percent_TP=p)

        # print(rf_tps[(6,12)], type(rf_tps[(6,12)]))
        # print(tp_ids[0], type(tp_ids[0]))

        overlap_rf = len(set(list(tp_ids)) & set(one_rf_tps[emb_type][(6,12)]))
        overlap_gv = len(set(list(tp_ids)) & set(variant_tp_ids))
        start = start+s
        rows.append([p, target_tp, tp, overlap_rf, overlap_gv, tp+fp, (tp+fp)/len(la)])
    df = pd.DataFrame(rows)

df.columns = ['sensitivity', 'target_tp', 'tp', 'overlap_with_rf', 'overlap_with_gv', 'NNS', 'NNS per total population']
df.to_csv('output/NNS.csv', index = None)