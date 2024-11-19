import pickle
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

def find_optimal_threshold(precision, recall, thresholds):
    # Example: Choose the threshold that maximizes the F1 score
    f1_scores = 2*(precision*recall) / (precision+recall)
    f1_scores = np.nan_to_num(f1_scores)
    max_f1_index = np.argmax(f1_scores[:-1])  # exclude last value which corresponds to recall=0
    optimal_threshold = thresholds[max_f1_index]
    return optimal_threshold, f1_scores[max_f1_index]

###############################################################

# data_path = 'output/prev/rs_tf_gpt1536dynamic_dx.pickle'
data_path = 'output/rs_baseline1536.pickle'
model_type = 'multi_class' 

# data_path = '../12-36m model/output/rs_tf1536_gpt_dx.pickle'
# model_type = 'binary'

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

        print(m, 'ppv: ', tp/(tp+fp), 'tp: ', tp, 'fp: ', fp)
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

    print('ppv: ', tp/(tp+fp), 'tp: ', tp, 'fp: ', fp)

hr['TP'] = [id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 1]
hr['FP'] = [id for pred, true, id in zip(predictions, la, ids) if pred == 1 and true == 0]
# with open('output/rf_hr_'+model_type+'_12-36m.pickle', 'wb') as h:
#     pickle.dump(hr, h)