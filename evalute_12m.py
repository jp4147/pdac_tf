import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from scipy.interpolate import interp1d

with open('output/rs_tf_dx.pickle', 'rb') as handle:
    b32 = pickle.load(handle)
with open('output/rs_tf1536_dx.pickle', 'rb') as handle:
    b1536 = pickle.load(handle)
with open('output/rs_tf_gpt1536_dx.pickle', 'rb') as handle:
    gpt1536 = pickle.load(handle)

def sens_spec_range(res, spec_from = 0.80, spec_to = 1.0):
    label2month = {5:'3m', 4:'6m', 3:'12m', 2:'36m', 1:'60m'}  
    max_sum = 5    
    tpr_collect, fpr_collect = {}, {}
    for la_sum, m in label2month.items():
        fpr, tpr, roc_thre = roc_curve(np.array(res[m]['labels'])[:,max_sum -la_sum], np.array(res[m]['raw_scores'])[:,max_sum -la_sum])

        tpr_collect[m]=tpr
        fpr_collect[m]=fpr

    sen_dict = {}
    spec_dict = {}
 
    for m in label2month.values():
        sen_dict[m] = {}
        spec_dict[m] = {}
        for specAt in tqdm(np.arange(spec_from, spec_to, 0.001)):
            spec = (1-fpr_collect[m])
            sen = tpr_collect[m]

            spec_reversed = spec[::-1]
            sen_reversed = sen[::-1]

            sen_95spec = np.interp(specAt, spec_reversed, sen_reversed)
            f = interpolate.interp1d(sen, spec, kind='linear', fill_value="extrapolate")
            spec_95sen = f(specAt)

            sen_dict[m][specAt] = sen_95spec
            spec_dict[m][specAt] = spec_95sen
        
    return sen_dict, spec_dict, tpr_collect, fpr_collect


res_list = [b32, b1536, gpt1536]
mod_name = ['$Baseline_{\t{fine\_tune32}}$ ', '$Baseline_{\t{fine\_tune1536}}$ ', '$GPT_{\t{fine\_tune1536}}$ ']
color = ['k','gray','r']

m = '12m'
for plot_type in ['no_interp', 'interp']:
    i=0
    for res in res_list:
        sen, spec, tpr, fpr = sens_spec_range(res, spec_from = 0, spec_to = 1.0)

        specAt = 0.85
        current_spec = spec[m]
        current_sen = sen[m]

        # Convert the dictionary keys and values to arrays for processing
        spec_values = np.array(list(current_spec.keys()))  # Specificity keys
        sen_values = np.array(list(current_sen.values()))  # Sensitivity values
        # idx = np.where(spec_values == specAt)
        # print(mod_name[i], ':', sen_values[idx])

        interp_x = np.linspace(0,0.999,10)
        interp_y = interp1d(spec_values, sen_values, kind = 'linear')(interp_x)

        roc_auc = auc(fpr[m], tpr[m])
        label= mod_name[i] + '(AUROC'+': {:.3f}'.format(roc_auc)+')'
        # plt.scatter(spec_values, sen_values, color = color[i], label = label, s = 2)
        if plot_type == 'no_interp':
            plt.plot(interp_x, interp_y, color = color[i], label = label)
        else:
            plt.plot(spec_values, sen_values, color = color[i], label = label)
        i=i+1

    plt.xlabel('Specificity', fontsize = 15)
    plt.ylabel('Sensitivity', fontsize = 15) 
    plt.legend(fontsize=12)
    plt.savefig('output/'+plot_type+f'figure2_{m}.png')  # Save plot as a PNG file
    plt.clf()

