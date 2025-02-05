import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

def find_optimal_threshold(precision, recall, thresholds):
    # Example: Choose the threshold that maximizes the F1 score
    f1_scores = 2*(precision*recall) / (precision+recall)
    f1_scores = np.nan_to_num(f1_scores)
    max_f1_index = np.argmax(f1_scores[:-1])  # exclude last value which corresponds to recall=0
    optimal_threshold = thresholds[max_f1_index]
    return optimal_threshold, f1_scores[max_f1_index]

data_paths = {}
###############################################################
# #CUMC
# data_paths['multi_class'] = ['output/prev/rs_tf_gpt1536dynamic_dx.pickle',
#                              'output/prev/test_rs_tf_gpt1536_dx.pickle',
#                              'output/prev/test_rs_tf_dx1536.pickle',
#                              'output/rs_tf_dx1536_fine_tune.pickle',
#                              'output/rs_tf_gpt_dx1536_fine_tune.pickle',
#                              'output/rs_baseline1536.pickle',
#                              'output/rs_gpt1536.pickle']
# data_paths['binary'] = ['output/baseline_binary1536_3-6m.pickle',
#                         'output/baseline_binary1536_6-12m.pickle',
#                         'output/baseline_binary1536_12-36m.pickle',
#                         'output/gpt_binary1536_3-6m.pickle',
#                         'output/gpt_binary1536_6-12m.pickle',
#                         'output/gpt_binary1536_12-36m.pickle']

#CSMC
data_paths['multi_class'] = ['output/rs_tf1536_dx.pickle',
                             'output/rs_tf_gpt1536_dx.pickle',
                             'output/rs_baseline1536.pickle',
                             'output/rs_gpt1536.pickle']
data_paths['binary'] = ['output/baseline_binary1536_3-6m.pickle',
                        'output/baseline_binary1536_6-12m.pickle',
                        'output/baseline_binary1536_12-36m.pickle',
                        'output/gpt_binary1536_3-6m.pickle',
                        'output/gpt_binary1536_6-12m.pickle',
                        'output/gpt_binary1536_12-36m.pickle']
rows = []
for model_type in ['multi_class', 'binary']:
    data_path_lst = data_paths[model_type]
    for data_path in data_path_lst:
        with open(data_path, 'rb') as h:
            rs = pickle.load(h)

        hr = {}
        if model_type == 'multi_class':
            month2idx = {list(rs.keys())[i]:i for i in range(len(rs))}
            for m in month2idx:

                ids = rs[m]['ids']
                la = [i[month2idx[m]] for i in rs[m]['labels']]
                rs_list = [i[month2idx[m]] for i in rs[m]['raw_scores']]
                precision, recall, thresholds = precision_recall_curve(la, rs_list)
                fpr, tpr, roc_thre = roc_curve(la, rs_list)

                optimal_threshold, max_f1_score = find_optimal_threshold(precision, recall, thresholds)
                predictions = [1 if score >= optimal_threshold else 0 for score in rs_list]

                # Calculate True Positives (TP) and False Positives (FP)
                tp = len([id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 1])
                fp = len([id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 0])

                # print(m, 'ppv: ', tp/(tp+fp), 'tp: ', tp, 'fp: ', fp)
                # print(m, 'sample size:', sum(la))
                sample_size = sum(la)
                row_data = {
                    "model_type": model_type,
                    "model_name": data_path,
                    "pred_window": m,
                    "ppv": tp/(tp+fp),
                    "tp": tp,
                    "fp": fp,
                    "sample_size": sample_size
                }
                rows.append(row_data)
        elif model_type == 'binary':
            ids = rs['ids']
            la = rs['labels']
            rs_list = rs['raw_scores']
            precision, recall, thresholds = precision_recall_curve(la, rs_list)

            optimal_threshold, max_f1_score= find_optimal_threshold(precision, recall, thresholds)

            predictions = [1 if score >= optimal_threshold else 0 for score in rs_list]

            # Calculate True Positives (TP) and False Positives (FP)
            tp = len([id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 1])
            fp = len([id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 0])

            # print('ppv: ', tp/(tp+fp), 'tp: ', tp, 'fp: ', fp)
            pred_window = data_path.split('_')[-1]
            sample_size = len(la)

            row_data = {
                "model_type": model_type,
                "model_name": data_path,
                "pred_window": pred_window,
                "ppv": tp/(tp+fp),
                "tp": tp,
                "fp": fp,
                "sample_size": sample_size
            }
            rows.append(row_data)
        hr['TP'] = [id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 1]
        hr['FP'] = [id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 0]
        name = data_path.split('/')
        name = name[-1].split('.')
        name = name[0]
        with open('output/'+'_HR_'+name, 'wb') as handle:
            pickle.dump(hr, handle)
df = pd.DataFrame(rows)
df.to_csv('output/clinical_utilty_eval.csv', index = None)
