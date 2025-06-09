import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy import stats
import pickle
from tqdm import tqdm

print('load data')
base = 1536
gpt = 1536
binary_eval = 0
multi_eval = 1
emb_eval = 0
save_path = 'Table1_base32_and_binary_log'

# CUMC
if emb_eval ==1:
    with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/prev/test_rs_tf_dx1536.pickle', 'rb') as h:
        base_multi = pickle.load(h)
    with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/prev/test_rs_tf_gpt1536_dx.pickle', 'rb') as h:
        gpt_multi = pickle.load(h)
    with open('D:/2024/GNN exp/gpt manuscript/npj_code/datasets/test_rs_tf_gnn128.pickle', 'rb') as h:
        gnn = pickle.load(h)
    with open('D:/2024/GNN exp/gpt manuscript/npj_code/datasets/test_rs_tf_mixtral4096_dx.pickle', 'rb') as h:
        mistral = pickle.load(h)    
    
if multi_eval == 1:
    if base == 32:
        with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/prev/test_rs_tf_dx.pickle', 'rb') as h:
            base_multi = pickle.load(h)
        with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/rs_tf_dx32_fine_tune.pickle', 'rb') as h:
            base_multi_exl = pickle.load(h)
    else:
        # with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/prev/test_rs_tf_dx1536.pickle', 'rb') as h:
        #     base_multi = pickle.load(h)
        # with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/Cedars/output/prev/tableS2/rs_tf_dx1536_freeze.pickle', 'rb') as h:
        #     base_multi = pickle.load(h)
        # with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/rs_tf_dx1536_fine_tune.pickle', 'rb') as h:
        #     base_multi_exl = pickle.load(h)
        with open('D:/2024/GNN exp/gpt manuscript/npj_code/datasets/test_rs_tf_mixtral4096_dx.pickle', 'rb') as h:
            base_multi = pickle.load(h) 
    if gpt == 32:
        with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/prev/test_rs_tf_gpt32_dx.pickle', 'rb') as h:
            gpt_multi = pickle.load(h)   
        with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/rs_tf_gpt_dx32_fine_tune.pickle', 'rb') as h:
            gpt_multi_exl = pickle.load(h)
    else:
        # with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/prev/test_rs_tf_gpt1536_dx.pickle', 'rb') as h:
        #     gpt_multi = pickle.load(h)
        with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/prev/test_rs_tf_gpt32_dx.pickle', 'rb') as h:
            gpt_multi = pickle.load(h)   
        # with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/Cedars/output/prev/tableS2/test_rs_tf_gpt1536s_dx.pickle', 'rb') as h:
        #     gpt_multi = pickle.load(h)
        # with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/output/rs_tf_gpt_dx1536_fine_tune.pickle', 'rb') as h:
        #     gpt_multi_exl = pickle.load(h)


if binary_eval == 1:
    with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/Cedars/output/baseline_binary1536_3-6m.pickle', 'rb') as h:
        base3 = pickle.load(h)
    with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/Cedars/output/gpt_binary1536_3-6m.pickle', 'rb') as h:
        gpt3 = pickle.load(h)

    with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/Cedars/output/baseline_binary1536_6-12m.pickle', 'rb') as h:
        base6 = pickle.load(h)
    with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/Cedars/output/gpt_binary1536_6-12m.pickle', 'rb') as h:
        gpt6 = pickle.load(h)
    # with open('output/baseline_binary1536_6-12m.pickle', 'rb') as h:
    #     base6 = pickle.load(h)
    # with open('output/gpt_binary1536_6-12m.pickle', 'rb') as h:
    #     gpt6 = pickle.load(h)

    with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/Cedars/output/baseline_binary1536_12-36m.pickle', 'rb') as h:
        base12 = pickle.load(h)
    with open('D:/2024/GNN exp/gpt manuscript/plosMed revision code/Cedars/output/baseline_binary1536_12-36m.pickle', 'rb') as h:
        gpt12 = pickle.load(h)

# # # #CSMC
# with open('rs_tf_dx.pickle', 'rb') as h:
#     base_multi = pickle.load(h)
# # # with open('rs_tf1536_dx.pickle', 'rb') as h:
# # #     base_multi = pickle.load(h)
# with open('rs_tf_gpt1536_dx.pickle', 'rb') as h:
#     gpt_multi = pickle.load(h)

# # # with open('rs_baseline1536.pickle', 'rb') as h:
# # #     base_multi_exl = pickle.load(h)
# # # with open('rs_gpt1536.pickle', 'rb') as h:
# # #     gpt_multi_exl = pickle.load(h)

# with open('baseline_binary_3-6m.pickle', 'rb') as h:
#     base3 = pickle.load(h)
# with open('gpt_binary_3-6m.pickle', 'rb') as h:
#     gpt3 = pickle.load(h)

# with open('baseline_binary_6-12m.pickle', 'rb') as h:
#     base6 = pickle.load(h)
# with open('gpt_binary_6-12m.pickle', 'rb') as h:
#     gpt6 = pickle.load(h)

# with open('baseline_binary_12-36m.pickle', 'rb') as h:
#     base12 = pickle.load(h)
# with open('gpt_binary_12-36m.pickle', 'rb') as h:
#     gpt12 = pickle.load(h)

print('bootstrap testing')
def bootstrap_test(y_true, y_pred, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    auc_diffs, pr_diffs, f1_diffs = [], [], []
    aucs = []
    prs= []
    f1s = []

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_base = auc(recall,precision)
    f1 = 2*(precision*recall)/(precision+recall+1e-8)
    max_f1 = np.max(f1)

    positives = np.where(y_true == 1)[0]
    negatives = np.where(y_true == 0)[0]

    for _ in tqdm(range(n_bootstraps)):
        # Always keep all positives
        boot_pos_indices = rng.choice(positives, size=len(positives), replace=True)
        boot_neg_indices = rng.choice(negatives, size=len(negatives), replace=True)
        indices = np.concatenate([boot_pos_indices, boot_neg_indices])
        
        auc1 = roc_auc_score(y_true[indices], y_pred[indices])
        aucs.append(auc1)


        precision, recall, _ = precision_recall_curve(y_true[indices], y_pred[indices])
        pr = auc(recall,precision)
        prs.append(pr)
        f1 = 2*(precision*recall)/(precision+recall+1e-8)
        max_f1_1 = np.max(f1)
        f1s.append(max_f1_1)

    return {
        "auroc": auroc,
        "ci_lower_1_auc": np.percentile(aucs, 2.5),
        "ci_upper_1_auc": np.percentile(aucs, 97.5),
    }

def bootstrap_auc_diff(y_true, y_pred1, y_pred2, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    auc_diffs, pr_diffs, f1_diffs = [], [], []
    auc1s, auc2s = [], []
    pr1s, pr2s = [],[]
    f1s_1, f1s_2 = [], []

    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    auroc_base = roc_auc_score(y_true, y_pred1)
    auroc_gpt = roc_auc_score(y_true, y_pred2)

    precision, recall, _ = precision_recall_curve(y_true, y_pred1)
    pr_base = auc(recall,precision)
    f1 = 2*(precision*recall)/(precision+recall+1e-8)
    max_f1_base = np.max(f1)

    precision, recall, _ = precision_recall_curve(y_true, y_pred2)
    pr_gpt = auc(recall,precision)
    f1 = 2*(precision*recall)/(precision+recall+1e-8)
    max_f1_gpt = np.max(f1)

    # for _ in tqdm(range(n_bootstraps)):
    #     indices = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
    #     if len(np.unique(y_true[indices])) < 2:
    #         continue  # skip if resample has no pos/neg cases
    #     auc1 = roc_auc_score(y_true[indices], y_pred1[indices])
    #     auc2 = roc_auc_score(y_true[indices], y_pred2[indices])
    #     auc_diffs.append(auc1 - auc2)

    positives = np.where(y_true == 1)[0]
    negatives = np.where(y_true == 0)[0]

    for _ in tqdm(range(n_bootstraps)):
        # Always keep all positives
        boot_pos_indices = rng.choice(positives, size=len(positives), replace=True)
        boot_neg_indices = rng.choice(negatives, size=len(negatives), replace=True)
        indices = np.concatenate([boot_pos_indices, boot_neg_indices])
        
        auc1 = roc_auc_score(y_true[indices], y_pred1[indices])
        auc2 = roc_auc_score(y_true[indices], y_pred2[indices])
        auc_diffs.append(auc2 - auc1)  # note: auc_gpt - auc_base
        auc1s.append(auc1)
        auc2s.append(auc2)

        precision, recall, _ = precision_recall_curve(y_true[indices], y_pred1[indices])
        pr1 = auc(recall,precision)
        pr1s.append(pr1)
        f1 = 2*(precision*recall)/(precision+recall+1e-8)
        max_f1_1 = np.max(f1)
        f1s_2.append(max_f1_1)

        precision, recall, _ = precision_recall_curve(y_true[indices], y_pred2[indices])
        pr2 = auc(recall,precision)
        pr2s.append(pr2)
        f1 = 2*(precision*recall)/(precision+recall+1e-8)
        max_f1_2 = np.max(f1)
        f1s_1.append(max_f1_2)

        pr_diffs.append(pr1-pr2)
        f1_diffs.append(max_f1_1-max_f1_2)
        

    # Compute p-value: how often the difference is ≤ 0
    auc_diffs = np.array(auc_diffs)
    p_value_auc = np.mean(auc_diffs <= 0)

    pr_diffs = np.array(pr_diffs)
    p_value_pr = np.mean(pr_diffs <= 0)

    f1_diffs = np.array(f1_diffs)
    p_value_f1 = np.mean(f1_diffs <= 0)

    return {
        "p_value_auc": p_value_auc,
        "auroc_base": auroc_base,
        "ci_lower_1_auc": np.percentile(auc1s, 2.5),
        "ci_upper_1_auc": np.percentile(auc1s, 97.5),
        "auroc_gpt": auroc_gpt,
        "ci_lower_2_auc": np.percentile(auc2s, 2.5),
        "ci_upper_2_auc": np.percentile(auc2s, 97.5),
        "p_value_pr": p_value_pr,
        "pr_base": pr_base,
        "ci_lower_1_pr": np.percentile(pr1s, 2.5),
        "ci_upper_1_pr": np.percentile(pr1s, 97.5),
        "pr_gpt": pr_gpt,
        "ci_lower_2_pr": np.percentile(pr2s, 2.5),
        "ci_upper_2_pr": np.percentile(pr2s, 97.5),
        "p_value_f1": p_value_f1,
        "max_f1_base": max_f1_base,
        "ci_lower_1_f1": np.percentile(f1s_1, 2.5),
        "ci_upper_1_f1": np.percentile(f1s_1, 97.5),
        "max_f1_gpt": max_f1_gpt,
        "ci_lower_2_f1": np.percentile(f1s_2, 2.5),
        "ci_upper_2_f1": np.percentile(f1s_2, 97.5),
    }

def binary_label(lst):
    return np.array([1 if i.sum()>0 else 0 for i in lst ])
def pred_score(lst, ii=0):
    raw_scores = np.array([i[ii] for i in lst ])
    return 1/(1+np.exp(-raw_scores))

res = {}
if emb_eval==1:
    with open('output/'+save_path+'.txt', 'a') as f:
        print('multimodel incl 0-3m, 6-12m', file=f)
    m_index = {'3m':0, '6m':1, '12m':2, '36m':3, '60m':4}
    m = '12m'
    ii = m_index[m]

    result= bootstrap_auc_diff(binary_label(base_multi[m]['labels']), pred_score(base_multi[m]['raw_scores'], ii=ii), pred_score(gpt_multi[m]['raw_scores'], ii=ii))
    res['multi'] = result

    with open('output/'+save_path+'.txt', 'a') as f:
        print(f"AUC base: {result['auroc_base']:.4f}, AUC gpt: {result['auroc_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_auc']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_auc']:.4f}, {result['ci_upper_1_auc']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_auc']:.4f}, {result['ci_upper_2_auc']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"PR base: {result['pr_base']:.4f}, PR gpt: {result['pr_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_pr']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_pr']:.4f}, {result['ci_upper_1_pr']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_pr']:.4f}, {result['ci_upper_2_pr']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"F1 base: {result['max_f1_base']:.4f}, F1 gpt: {result['max_f1_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_f1']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_f1']:.4f}, {result['ci_upper_1_f1']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_f1']:.4f}, {result['ci_upper_2_f1']:.4f})", file=f)

        print('GNN, 6-12m')

    result= bootstrap_auc_diff(binary_label(gnn[m]['labels']), pred_score(gnn[m]['raw_scores'], ii=ii), pred_score(gpt_multi[m]['raw_scores'], ii=ii))
    res['gnn'] = result

    with open('output/'+save_path+'.txt', 'a') as f:
        print(f"AUC base: {result['auroc_base']:.4f}, AUC gpt: {result['auroc_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_auc']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_auc']:.4f}, {result['ci_upper_1_auc']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_auc']:.4f}, {result['ci_upper_2_auc']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"PR base: {result['pr_base']:.4f}, PR gpt: {result['pr_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_pr']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_pr']:.4f}, {result['ci_upper_1_pr']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_pr']:.4f}, {result['ci_upper_2_pr']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"F1 base: {result['max_f1_base']:.4f}, F1 gpt: {result['max_f1_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_f1']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_f1']:.4f}, {result['ci_upper_1_f1']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_f1']:.4f}, {result['ci_upper_2_f1']:.4f})", file=f)

        print('MISTRAL, 6-12m')

    result= bootstrap_auc_diff(binary_label(mistral[m]['labels']), pred_score(mistral[m]['raw_scores'], ii=ii), pred_score(gpt_multi[m]['raw_scores'], ii=ii))
    res['mistral'] = result

    with open('output/'+save_path+'.txt', 'a') as f:
        print(f"AUC base: {result['auroc_base']:.4f}, AUC gpt: {result['auroc_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_auc']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_auc']:.4f}, {result['ci_upper_1_auc']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_auc']:.4f}, {result['ci_upper_2_auc']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"PR base: {result['pr_base']:.4f}, PR gpt: {result['pr_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_pr']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_pr']:.4f}, {result['ci_upper_1_pr']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_pr']:.4f}, {result['ci_upper_2_pr']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"F1 base: {result['max_f1_base']:.4f}, F1 gpt: {result['max_f1_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_f1']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_f1']:.4f}, {result['ci_upper_1_f1']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_f1']:.4f}, {result['ci_upper_2_f1']:.4f})", file=f)



if multi_eval==1:
    with open('output/'+save_path+'.txt', 'a') as f:
        print('multimodel incl 0-3m', file=f)
    m_index = {'3m':0, '6m':1, '12m':2, '36m':3, '60m':4}
    res['multi'] = {}
    for m in m_index:
    # for m in ['12m', '60m']:
        ii = m_index[m]

        # rm_ids = list(set(base_multi[m]['ids'])-set(gpt_multi[m]['ids']))
        # indices = [i for i, val in enumerate(base_multi[m]['ids']) if val in rm_ids]
        # filtered_labels = [val for i, val in enumerate(base_multi[m]['labels']) if i not in indices]
        # filtered_scores = [val for i, val in enumerate(base_multi[m]['raw_scores']) if i not in indices]
        # result= bootstrap_auc_diff(binary_label(filtered_labels), pred_score(filtered_scores, ii=ii), pred_score(gpt_multi[m]['raw_scores'], ii=ii))
        # result2= bootstrap_test(binary_label(base_multi[m]['labels']), pred_score(base_multi[m]['raw_scores'], ii=ii))
        # with open('output/'+save_path+'.txt', 'a') as f:
        #     print(m, file=f)
        #     print(f"AUC base: {result['auroc_base']:.4f}, AUC gpt: {result['auroc_gpt']:.4f}", file=f)
        #     print(f"p-value: {result['p_value_auc']:.4f}", file=f)
        #     print(f"95% CI: ({result['ci_lower_1_auc']:.4f}, {result['ci_upper_1_auc']:.4f})", file=f)
        #     print(f"95% CI: ({result['ci_lower_2_auc']:.4f}, {result['ci_upper_2_auc']:.4f})", file=f)
            # print('just base 32m', file=f)
            # print(f"AUC base: {result2['auroc']:.4f}", file=f)
            # print(f"95% CI: ({result2['ci_lower_1_auc']:.4f}, {result2['ci_upper_1_auc']:.4f})", file=f)


        # res['multi'][m] = result

    # with open('output/'+save_path+'.txt', 'a') as f:
    #     print('multimodel excl 0-3m', file=f)
    # res['multi_excl'] = {}
    # m_index = {'6m':0, '12m':1, '36m':2, '60m':3}
    m_index = {'3m':0, '6m':1, '12m':2, '36m':3, '60m':4}
    for m in m_index:
        ii = m_index[m]

        # result= bootstrap_auc_diff(binary_label(base_multi_exl[m]['labels']), pred_score(base_multi_exl[m]['raw_scores'], ii=ii), pred_score(gpt_multi_exl[m]['raw_scores'], ii=ii))
        result= bootstrap_auc_diff(binary_label(base_multi[m]['labels']), pred_score(base_multi[m]['raw_scores'], ii=ii), pred_score(gpt_multi[m]['raw_scores'], ii=ii))
        print(f"AUC base: {result['auroc_base']:.4f}, AUC gpt: {result['auroc_gpt']:.4f}")
        print(f"p-value: {result['p_value_auc']:.4f}")
        print(f"95% CI: ({result['ci_lower_1_auc']:.4f}, {result['ci_upper_1_auc']:.4f})")
        print(f"95% CI: ({result['ci_lower_2_auc']:.4f}, {result['ci_upper_2_auc']:.4f})")
        # with open('output/'+save_path+'.txt', 'a') as f:
        #     print(m)
        #     print(f"AUC base: {result['auroc_base']:.4f}, AUC gpt: {result['auroc_gpt']:.4f}", file=f)
        #     print(f"p-value: {result['p_value_auc']:.4f}", file=f)
        #     print(f"95% CI: ({result['ci_lower_1_auc']:.4f}, {result['ci_upper_1_auc']:.4f})", file=f)
        #     print(f"95% CI: ({result['ci_lower_2_auc']:.4f}, {result['ci_upper_2_auc']:.4f})", file=f)
        # res['multi_excl'][m] = result

if binary_eval==1:
    
    res['binary'] = {}
    result= bootstrap_auc_diff(np.array(base3['labels']), pred_score(base3['raw_scores']), pred_score(gpt3['raw_scores']))
    with open('output/'+save_path+'.txt', 'a') as f:
        print('3-6m binary model', file=f)
        print(f"AUC base: {result['auroc_base']:.4f}, AUC gpt: {result['auroc_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_auc']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_auc']:.4f}, {result['ci_upper_1_auc']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_auc']:.4f}, {result['ci_upper_2_auc']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"PR base: {result['pr_base']:.4f}, PR gpt: {result['pr_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_pr']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_pr']:.4f}, {result['ci_upper_1_pr']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_pr']:.4f}, {result['ci_upper_2_pr']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"F1 base: {result['max_f1_base']:.4f}, F1 gpt: {result['max_f1_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_f1']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_f1']:.4f}, {result['ci_upper_1_f1']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_f1']:.4f}, {result['ci_upper_2_f1']:.4f})", file=f)
    res['binary']['6m'] =result

    
    result = bootstrap_auc_diff(np.array(base6['labels']), pred_score(base6['raw_scores']), pred_score(gpt6['raw_scores']))
    with open('output/'+save_path+'.txt', 'a') as f:
        print('6-12m binary model', file=f)
        print(f"AUC base: {result['auroc_base']:.4f}, AUC gpt: {result['auroc_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_auc']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_auc']:.4f}, {result['ci_upper_1_auc']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_auc']:.4f}, {result['ci_upper_2_auc']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"PR base: {result['pr_base']:.4f}, PR gpt: {result['pr_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_pr']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_pr']:.4f}, {result['ci_upper_1_pr']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_pr']:.4f}, {result['ci_upper_2_pr']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"F1 base: {result['max_f1_base']:.4f}, F1 gpt: {result['max_f1_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_f1']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_f1']:.4f}, {result['ci_upper_1_f1']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_f1']:.4f}, {result['ci_upper_2_f1']:.4f})", file=f)
    res['binary']['12m'] = result

    result = bootstrap_auc_diff(np.array(base12['labels']), pred_score(base12['raw_scores']), pred_score(gpt12['raw_scores']))
    with open('output/'+save_path+'.txt', 'a') as f:
        print('12-36m binary model', file=f)
        print(f"AUC base: {result['auroc_base']:.4f}, AUC gpt: {result['auroc_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_auc']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_auc']:.4f}, {result['ci_upper_1_auc']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_auc']:.4f}, {result['ci_upper_2_auc']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"PR base: {result['pr_base']:.4f}, PR gpt: {result['pr_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_pr']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_pr']:.4f}, {result['ci_upper_1_pr']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_pr']:.4f}, {result['ci_upper_2_pr']:.4f})", file=f)
        print('-------------------------------------------------------------', file=f)
        print(f"F1 base: {result['max_f1_base']:.4f}, F1 gpt: {result['max_f1_gpt']:.4f}", file=f)
        print(f"p-value: {result['p_value_f1']:.4f}", file=f)
        print(f"95% CI: ({result['ci_lower_1_f1']:.4f}, {result['ci_upper_1_f1']:.4f})", file=f)
        print(f"95% CI: ({result['ci_lower_2_f1']:.4f}, {result['ci_upper_2_f1']:.4f})", file=f)
    res['binary']['36m'] = result

# with open(save_path, 'wb') as h:
#     pickle.dump(res, h)