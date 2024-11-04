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

mod = 'gpt'
torch.cuda.set_device(1)
device = torch.device('cuda')
embedding_dim = 1536
output_dim = 4
lr = 0.0000001

pat_data_path = 'datasets/pat_data_rev.pkl'
emb_path = 'datasets/gpt_emb1536.pickle'
save_path = mod+str(embedding_dim)
if os.path.exists(pat_data_path):
    print('pat_data_rev.pickle exist')
else:
    print('create pat_data_rev.pickle excluding 0-3m data')
    ############### DATA EXCLUDING 0-3m########################
    print('load data')
    with open('datasets/pat_data.pkl', 'rb') as h:
        pat_data = pickle.load(h)

    print('load pc_diag.pickle')
    with open('datasets/pc_diag.pickle', 'rb') as h:
        pc_diag = pickle.load(h)

    labels = {5:'3m', 4:'6m', 3:'12m', 2:'36m', 1:'60m'}
    case = {}
    for idx, month in labels.items():
        case[month] = []
        for k, v in pat_data.items():
            if sum(v['label'])==idx:
                case[month].append(k)

    print('sample sizes')
    for idx, month in labels.items():
        print(month, len(case[month]))

    print('exclude 0-3m data')
    case_rev = {}
    for id_3m in case['3m']:
        timestamps = pat_data[id_3m]['timestamps']
        three_months_before_diagnosis = pc_diag[id_3m] - pd.DateOffset(months=3)
        filtered_timestamps = [timestamp for timestamp in timestamps if timestamp < three_months_before_diagnosis]
        if len(filtered_timestamps)>=5:
            case_rev[id_3m] = {}
            case_rev[id_3m]['concept_dx'] = pat_data[id_3m]['concept_dx'][0:len(filtered_timestamps)]
            case_rev[id_3m]['timestamps'] = filtered_timestamps
            case_rev[id_3m]['age'] = pat_data[id_3m]['age'][0:len(filtered_timestamps)]
            case_rev[id_3m]['age_norm'] = pat_data[id_3m]['age_norm'][0:len(filtered_timestamps)]
            for i in ['race','ethnicity','gender','age_at_diagnosis','birth_date']:
                case_rev[id_3m][i] = pat_data[id_3m][i]
            case_rev[id_3m]['label'] = [0,1,1,1,1]

    for m in ['6m', '12m', '36m', '60m']:
        case_rev.update({i:pat_data[i] for i in case[m]})

    ctrl = {}
    for k, v in pat_data.items():
        if sum(v['label'])==0:
            ctrl[k] = v

    data_rev = {}
    data_rev.update(case_rev)
    print('add contrl group')
    data_rev.update(ctrl)

    for k, v in data_rev.items():
        v['label'] = v['label'][1:]

    print('shuffle data')
    keys = list(data_rev.keys())
    random.shuffle(keys)

    shuffled_dict = {key: data_rev[key] for key in keys}

    print('revised sample size')
    la = []
    for k, v in data_rev.items():
        la.append(str(v['label']))

    print(Counter(la))

    print('save pat_data_rev.pkl')
    with open('datasets/pat_data_rev.pkl', 'wb') as h:
        pickle.dump(shuffled_dict, h, protocol=pickle.HIGHEST_PROTOCOL)

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

    torch.save(best_model, 'model/'+mod+str(embedding_dim)+'.pt')
    with open('model/train_losses_'+mod+str(embedding_dim)+'.pickle', 'wb') as handle:
        pickle.dump(train_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('model/val_losses_'+mod+str(embedding_dim)+'.pickle', 'wb') as handle:
        pickle.dump(val_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
    # best_model, train_loss, val_loss = training.training()

    torch.save(best_model, 'model/'+save_path+'.pt')
    with open('model/train_losses_'+save_path+'.pickle', 'wb') as handle:
        pickle.dump(train_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('model/val_losses_'+save_path+'.pickle', 'wb') as handle:
        pickle.dump(val_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)    