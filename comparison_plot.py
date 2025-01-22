import pickle
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def sens_spec_range(fpr, tpr, spec_from = 0, spec_to = 1.0):
    sen_dict = {}
    spec_dict = {}
    for specAt in np.arange(spec_from, spec_to, 0.001):
        spec = (1-fpr)
        sen = tpr

        spec_reversed = spec[::-1]
        sen_reversed = sen[::-1]

        sen_95spec = np.interp(specAt, spec_reversed, sen_reversed)
        f = interpolate.interp1d(sen, spec, kind='linear', fill_value="extrapolate")
        spec_95sen = f(specAt)

        sen_dict[specAt] = sen_95spec
        spec_dict[specAt] = spec_95sen

    return sen_dict, spec_dict

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

        with open('model/train_losses_'+mod+str(embedding_dim)+'_'+months_prior+'.pickle', 'rb') as handle:
            train_losses = pickle.load(handle)
        print(mod+' '+ months_prior+' '+'num_epoch:', len(train_losses))
        with open('model/val_losses_'+mod+str(embedding_dim)+'_'+months_prior+'.pickle', 'rb') as handle:
            val_losses = pickle.load(handle) 

        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.savefig(f'output/plot_{months_prior}_'+mod+'_losses.png')
        plt.clf()

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
        elif mod=='baseline' or mod == 'gpt':
            roc_label='AUROC '+m+',' +mod+'_binary,'+': {:.3f}'.format(roc_auc)
        else:
            roc_label='AUROC '+m+',' +mod+': {:.3f}'.format(roc_auc)
        
        # plt.plot(fpr[mod][m], tpr[mod][m], label = roc_label, color = color, linewidth = linewidth, alpha = alpha)
        sen, spec = sens_spec_range(fpr[mod][m], tpr[mod][m])
        plt.plot(sen.keys(), sen.values(), label = roc_label, color = color, linewidth = linewidth, alpha = alpha)
    # plt.xlabel('False Positive Rate', fontsize =15)
    # plt.ylabel('True Positive Rate', fontsize =15)
    # plt.legend(loc = 'lower right', fontsize =12)
    plt.xlabel('Specificity', fontsize =15)
    plt.ylabel('Sensitivity', fontsize =15)
    plt.legend(loc = 'lower left', fontsize =12)
    plt.savefig(f'output/plot_{m}_comp.png')  # Save plot as a PNG file
    plt.clf()
