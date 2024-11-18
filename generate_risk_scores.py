import pickle
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

from training import Training
from load_data import DataLoader
from evaluate_performance import Evaluate

mod = 'gpt'
embedding_dim = 1536
output_dim = 4
torch.cuda.set_device(1)
device = torch.device('cuda')
pat_data_path = 'datasets/pat_data_rev.pkl'
emb_path = 'datasets/gpt_emb1536.pickle'
rs_path = 'output/rs_'+mod+str(embedding_dim)+'.pickle'

# with open('model/train_losses_'+mod+str(embedding_dim)+'.pickle', 'rb') as handle:
#     train_loss = pickle.load(handle)
# with open('model/val_losses_'+mod+str(embedding_dim)+'.pickle', 'rb') as handle:
#     val_loss = pickle.load(handle)

# plt.figure()
# plt.plot(train_loss)
# plt.plot(val_loss)
# plt.show()

############################Generate Risk Scores####################################
if os.path.exists(rs_path):
    print(rs_path+' exist')    
else:
    print('generate risk scores')
    data_loader = DataLoader(pat_data_path, use_graph_embeddings = False)  
    data = data_loader.reidx_dat
    pat_c2i = data_loader.pat_c2i
    vocab_size = data_loader.vocab_size
    if mod == 'baseline':
        training = Training(data, save_path = None, vocab_size = vocab_size, embedding_dim = embedding_dim, output_dim = output_dim, device = device)
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

        training = Training(data, save_path=None, vocab_size = vocab_size, embedding_dim = embedding_dim, 
                            output_dim = output_dim, final_embeddings=final_embeddings, device = device)
    
    model = training.model

    test_data = training.data_splits.test_data
    _, _, test_ids = training.data_splits.split_ids()

    ev = Evaluate(test_data, test_ids, model, 'model/'+mod+str(embedding_dim)+'.pt', device = device)

    label2month = {4:'6m', 3:'12m', 2:'36m', 1:'60m', 0:'ctrl'}
    test_labels = {}
    for la, m in label2month.items():
        test_labels[m] = 0
    for i in test_data:
        test_labels[label2month[sum(i[2])]] += 1

    print(test_labels)
    ev.sens_spec(specAt = 0.999)

    rs = ev.raw_scores()
    with open(rs_path, 'wb') as handle:
        pickle.dump(rs, handle)
######################################################################################