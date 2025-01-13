import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import random
import torch
import numpy as np

from load_data import DataLoader
from training import Training
import os

mod = 'baseline'
months_prior = '3-6m' #'6-12m', '12-36m'
torch.cuda.set_device(1)
device = torch.device('cuda')
embedding_dim = 1536
output_dim = 1
lr = 0.0000001

pat_data_path = 'datasets/pat_dat_aug'+str(months_prior)+'.pickle'
emb_path = 'datasets/gpt_emb1536.pickle'
save_path = mod+str(embedding_dim)+'_'+months_prior
if os.path.exists(pat_data_path):
    print(pat_data_path +' exist')
else:
    print('create '+ pat_data_path)
    ############### DATA EXCLUDING 0-3m########################
    print('data loading...')
    with open('datasets/pat_data_rev.pkl', 'rb') as handle:
        pat_data = pickle.load(handle)

    with open('datasets/pc_diag.pickle', 'rb') as h:
        pc_diag = pickle.load(h)

    print('data augmentation starts')
    dat6 = {k:v for k,v in pat_data.items() if sum(v['label'])==4}
    dat12 = {k:v for k,v in pat_data.items() if sum(v['label'])==3}
    dat36 = {k:v for k,v in pat_data.items() if sum(v['label'])==2}

    dat_ctrl = {}
    for k,v in pat_data.items():
        if sum(v['label'])==0:
            dat_ctrl[k] = v
            dat_ctrl[k]['label'] = 0

    dat = {}
    case_rev = {}
    int_m = int(months_prior.split('-')[0])
    if int_m  == 3:
        case_rev.update(dat6)
    else:
        if int_m == 6:
            dat.update(dat6)
            dat.update(dat12)
        else:
            dat.update(dat6)
            dat.update(dat12)
            dat.update(dat36)

        for id in dat.keys():
            timestamps = pat_data[id]['timestamps']
            months_before_diagnosis = pc_diag[id] - pd.DateOffset(months=int_m)
            filtered_timestamps = [timestamp for timestamp in timestamps if timestamp <= months_before_diagnosis]
            if len(filtered_timestamps)>=5:
                case_rev[id] = {}
                case_rev[id]['concept_dx'] = pat_data[id]['concept_dx'][0:len(filtered_timestamps)]
                case_rev[id]['timestamps'] = filtered_timestamps
                case_rev[id]['age'] = pat_data[id]['age'][0:len(filtered_timestamps)]
                case_rev[id]['age_norm'] = pat_data[id]['age_norm'][0:len(filtered_timestamps)]
                for i in ['race','ethnicity','gender','age_at_diagnosis','birth_date']:
                    case_rev[id][i] = pat_data[id][i]

    dat_case = {}
    for k,v in case_rev.items():
        dat_case[k] = v
        dat_case[k]['label'] = 1

    print(months_prior, ' case:', len(dat_case), 'ctrl:', len(dat_ctrl))

    pat_dat_aug = {}
    pat_dat_aug.update(dat_case)
    pat_dat_aug.update(dat_ctrl)
    
    keys = list(pat_dat_aug.keys())
    random.shuffle(keys)

    shuffled_dict = {key: pat_dat_aug[key] for key in keys}

    print('save augmented data')
    with open(pat_data_path, 'wb') as handle:
        pickle.dump(shuffled_dict, handle)

########### Model training #########
print('model_type: ', mod)

data_loader = DataLoader(pat_data_path, use_graph_embeddings = False)  
pat_c2i = data_loader.pat_c2i
vocab_size = data_loader.vocab_size

data = data_loader.reidx_dat
# ########################Baseline 1536#####################################
print('training starts')
if mod == 'baseline':
    training = Training(data, save_path, vocab_size = vocab_size, embedding_dim = embedding_dim, output_dim = output_dim, device = device)

    best_model, train_loss, val_loss = training.training(lr = lr)

elif mod == 'gpt':
    with open(emb_path, 'rb') as handle:
        gpt_emb = pickle.load(handle)

    concept_emb_dict= {}
    for c, i in pat_c2i.items():
        concept_emb_dict[c] = gpt_emb[c]
        
    chat_gpt_emb = []
    for k, v in concept_emb_dict.items():
        chat_gpt_emb.append(v)
        
    chat_gpt_emb = torch.tensor(np.array(chat_gpt_emb))
    padding_embedding = torch.zeros(1, embedding_dim)
    final_embeddings = torch.cat((padding_embedding, chat_gpt_emb), dim=0)

    training = Training(data, save_path, vocab_size = vocab_size, embedding_dim = embedding_dim, 
                        output_dim = output_dim, final_embeddings=final_embeddings, device = device)

    best_model, train_loss, val_loss = training.training(lr = lr) 
