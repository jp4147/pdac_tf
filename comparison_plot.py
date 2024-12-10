import pickle
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

fpr, tpr = {}, {}
embedding_dim = 1536
months = ['3-6m', '6-12m', '12-36m']
print('load binary risk scores')
for mod in ['gpt', 'baseline']:
    fpr[mod], tpr[mod] = {},{}
    for months_prior in months:

        rs_path = 'output/'+mod+'_binary'+str(embedding_dim)+'_'+months_prior+'.pickle'

        with open(rs_path, 'rb') as handle:
            r = pickle.load(handle)
        fpr[mod][months_prior], tpr[mod][months_prior], _ = roc_curve(np.array(r['labels']), np.array(r['raw_scores']))


print('load multi-label risk scores')
rs_path_rm03 = 'output/rs_gpt'+str(embedding_dim)+'.pickle' # risk scores from multi-label classification GPT embedding model excluding 0-3 month data 
rs_path_multi = 'output/rs_tf_gpt'+str(embedding_dim)+'_dx.pickle' # risk scores from multi-label classification GPT embedding model including 0-3 month data 

with open(rs_path_multi, 'rb') as handle:
    gpt_multi = pickle.load(handle)
with open(rs_path_rm03, 'rb') as handle:
    rm03_gpt_multi = pickle.load(handle)

fpr['gpt_multi'], tpr['gpt_multi']= {},{}
fpr['rm03_gpt_multi'], tpr['rm03_gpt_multi']={},{}

fpr['gpt_multi']['3-6m'], tpr['gpt_multi']['3-6m'], _ = roc_curve(np.array(gpt_multi['6m']['labels'])[:,1], np.array(gpt_multi['6m']['raw_scores'])[:,1])
fpr['rm03_gpt_multi']['3-6m'], tpr['rm03_gpt_multi']['3-6m'], _ = roc_curve(np.array(rm03_gpt_multi['6m']['labels'])[:,0], np.array(rm03_gpt_multi['6m']['raw_scores'])[:,0])

fpr['gpt_multi']['6-12m'],tpr['gpt_multi']['6-12m'],_ = roc_curve(np.array(gpt_multi['12m']['labels'])[:,2], np.array(gpt_multi['12m']['raw_scores'])[:,2])
fpr['rm03_gpt_multi']['6-12m'],tpr['rm03_gpt_multi']['6-12m'],_ = roc_curve(np.array(rm03_gpt_multi['12m']['labels'])[:,1], np.array(rm03_gpt_multi['12m']['raw_scores'])[:,1])

fpr['gpt_multi']['12-36m'],tpr['gpt_multi']['12-36m'],_ = roc_curve(np.array(gpt_multi['36m']['labels'])[:,3], np.array(gpt_multi['36m']['raw_scores'])[:,3])
fpr['rm03_gpt_multi']['12-36m'],tpr['rm03_gpt_multi']['12-36m'],_ = roc_curve(np.array(rm03_gpt_multi['36m']['labels'])[:,2], np.array(rm03_gpt_multi['36m']['raw_scores'])[:,2])

print('save plots')

for m in months:
    for mod in ['baseline', 'gpt', 'gpt_multi', 'rm03_gpt_multi']:
        if mod == 'baseline':
            linewidth = 1.5
            alpha = 0.6
            color = 'k'
        elif mod =='gpt':
            linewidth = 1.5
            alpha = 1
            color = 'k'
        elif mod =='gpt_multi':
            linewidth = 1.5
            alpha = 1
            color = 'g'
        elif mod =='rm03_gpt_multi':
            linewidth = 1.5
            alpha = 0.6
            color = 'g'
        roc_auc = auc(fpr[mod][m], tpr[mod][m])
        if mod == 'rm03_gpt_multi':
            roc_label='AUROC '+m+',' +'gpt_multi(0-3 excl.)'+': {:.3f}'.format(roc_auc)
        elif mod=='base' or mod == 'gpt':
            roc_label='AUROC '+m+',' +mod+'_binary,'+': {:.3f}'.format(roc_auc)
        else:
            roc_label='AUROC '+m+',' +mod+': {:.3f}'.format(roc_auc)
        plt.plot(fpr[mod][m], tpr[mod][m], label = roc_label, color = color, linewidth = linewidth, alpha = alpha)
    plt.xlabel('False Positive Rate', fontsize =15)
    plt.ylabel('True Positive Rate', fontsize =15)
    plt.legend(loc = 'lower right', fontsize =12)
    plt.savefig(f'output/plot_{m}_comp.png')  # Save plot as a PNG file
    plt.clf()