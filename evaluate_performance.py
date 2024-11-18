import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from scipy import interpolate
import os
import torch.nn as nn
import pickle

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class Evaluate:
    def __init__(self, test_data, ids, model, trained_model_path, device):
        self.device = device
        
        ##########################################################
        #Single GPU
        self.model = model
        self.model.load_state_dict(torch.load(trained_model_path, map_location=self.device))
        
        # # multi GPU processing
        # state_dict = torch.load(trained_model_path, map_location=self.device)
        # self.model = model.to(self.device)
        # self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2])
        # # Adjust state_dict keys if they have 'module.' prefix
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('module.'):
        #         new_state_dict[k[len('module.'):]] = v
        #     else:
        #         new_state_dict[k] = v
        # self.model.module.load_state_dict(new_state_dict)
        ##########################################################       
        
        self.test_data = test_data
        self.ids = ids
        # self.label2month = {5:'3m', 4:'6m', 3:'12m', 2:'36m', 1:'60m'}
        self.label2month = {4:'6m', 3:'12m', 2:'36m', 1:'60m'}
        self.max_sum = 4
        
        self._raw_scores = None
    
    def data_by_label(self):
        y = [i[2] for i in self.test_data]
        y_sum = np.array([sum(i) for i in y])
        
        data_by_label = {}
        id_by_label = {}
        # for label_sum in [0,1,2,3,4,5]:
        for label_sum in [0,1,2,3,4]:
            idx = np.where(y_sum == label_sum)[0]
            data_by_label[label_sum] = [self.test_data[i] for i in idx]
            id_by_label[label_sum] = [self.ids[i] for i in idx]
        return data_by_label, id_by_label
                    
    def raw_scores(self):
        if self._raw_scores is None:
            data_by_label, id_by_label = self.data_by_label()
            res = {}
            for la_sum in tqdm(self.label2month.keys()):
                res[self.label2month[la_sum]] = {}
                all_outputs, all_labels = [], []
                
                test_loader = DataLoader(data_by_label[la_sum]+data_by_label[0], batch_size=16, shuffle=False, collate_fn=self.collate_fn)
                ids = id_by_label[la_sum]+id_by_label[0]
                
                self.model.eval() 
                with torch.no_grad():
                    for sequences, age, labels in test_loader:
                        sequences, age, labels = sequences.to(self.device), age.to(self.device), labels.float().to(self.device)

                        # Forward pass
                        outputs = self.model(sequences, age)
                        all_outputs.extend(outputs.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())
                        

                res[self.label2month[la_sum]]['ids'] = ids
                res[self.label2month[la_sum]]['raw_scores'] = all_outputs
                res[self.label2month[la_sum]]['labels'] = all_labels
            self._raw_scores = res
        return self._raw_scores
      
    def collate_fn(self, batch):
        seq=  pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
        age=  pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
        label = torch.tensor([item[2] for item in batch])
        return seq, age, label
    
    def roc_pr(self, plot = 'yes'):
        res = self._raw_scores if self._raw_scores else self.raw_scores()
        colors = ['k', 'g', 'b', 'orange', 'r']
        tpr_collect, fpr_collect, pre_collect, rec_collect = {}, {}, {}, {}
        pr_thre_collect, roc_thre_collect = {}, {}
        for la_sum, m in self.label2month.items():
            fpr, tpr, roc_thre = roc_curve(np.array(res[m]['labels'])[:,self.max_sum-la_sum], np.array(res[m]['raw_scores'])[:,self.max_sum-la_sum])
            pre, rec, pr_thre = precision_recall_curve(np.array(res[m]['labels'])[:,self.max_sum-la_sum], np.array(res[m]['raw_scores'])[:,self.max_sum-la_sum])

            tpr_collect[m]=tpr
            fpr_collect[m]=fpr
            pre_collect[m]=pre
            rec_collect[m]=rec
            pr_thre_collect[m] = pr_thre
            roc_thre_collect[m] = roc_thre
        if plot=='yes':
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
            for la_sum, m in self.label2month.items():  
                roc_auc = auc(fpr_collect[m], tpr_collect[m])
                roc_label='AUROC '+m+': {:.3f}'.format(roc_auc)
                ax1.plot(fpr_collect[m], tpr_collect[m], color = colors[self.max_sum-la_sum], label = roc_label)
                
                pr_auc = auc(rec_collect[m], pre_collect[m])
                pr_label = 'AUPR ' + m + ': {:.3f}'.format(pr_auc)
                ax2.plot(rec_collect[m], pre_collect[m], color=colors[self.max_sum-la_sum], label=pr_label)
                
            # Set titles and labels
            ax1.set_title('ROC Curve')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.legend(loc="lower right")
            
            ax2.set_title('Precision-Recall Curve')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.legend(loc="lower left")

            # Show plot
            plt.tight_layout()
            plt.show()
        
        return tpr_collect, fpr_collect, pre_collect, rec_collect, roc_thre_collect, pr_thre_collect
    
    
    def sens_spec(self, specAt = 0.99, plot = 'yes'):
        sen_dict = {}
        spec_dict = {}
        tpr, fpr, pre, rec, _, _ = self.roc_pr(plot = plot)
        for m in self.label2month.values():
            v_sen, v_spec = [],[]
            spec = (1-fpr[m])
            sen = tpr[m]

            spec_reversed = spec[::-1]
            sen_reversed = sen[::-1]

            sen_95spec = np.interp(specAt, spec_reversed, sen_reversed)
            f = interpolate.interp1d(sen, spec, kind='linear', fill_value="extrapolate")
            spec_95sen = f(specAt)

            sen_dict[m] = sen_95spec
            spec_dict[m] = spec_95sen
            
        print('sensitivity at '+str(specAt)+' specificity:', sen_dict)
        print('specificity at '+str(specAt)+' sensitivity:', spec_dict)
        return sen_dict, spec_dict
    
def sens_spec_range(file_name, spec_from = 0.80, spec_to = 1.0):
    
    with open('output/'+file_name, 'rb') as handle:
        res = pickle.load(handle)
    
    # label2month = {5:'3m', 4:'6m', 3:'12m', 2:'36m', 1:'60m'}  
    label2month = {4:'6m', 3:'12m', 2:'36m', 1:'60m'}
    max_sum = 4    
    colors = ['k', 'g', 'b', 'orange', 'r']
    tpr_collect, fpr_collect = {}, {}
    for la_sum, m in label2month.items():
        fpr, tpr, roc_thre = roc_curve(np.array(res[m]['labels'])[:,max_sum -la_sum], np.array(res[m]['raw_scores'])[:,max_sum -la_sum])
        pre, rec, pr_thre = precision_recall_curve(np.array(res[m]['labels'])[:,max_sum -la_sum], np.array(res[m]['raw_scores'])[:,max_sum -la_sum])

        tpr_collect[m]=tpr
        fpr_collect[m]=fpr

    sen_dict = {}
    spec_dict = {}
 
    for m in label2month.values():
        sen_dict[m] = {}
        spec_dict[m] = {}
        for specAt in tqdm(np.arange(spec_from, spec_to, 0.001)):
            v_sen, v_spec = [],[]
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
    