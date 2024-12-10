import pickle
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

from training import Training
from load_data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# mod = 'gpt' # 'gpt', 'baseline'
# months_prior = '6-12m' # '3-6m', '6-12m', '12-36m'
overwrite=1
embedding_dim = 1536
output_dim = 1

for mod in ['gpt', 'baseline']:
    for months_prior in ['3-6m','6-12m', '12-36m']:
        from load_data import DataLoader
        torch.cuda.set_device(1)
        device = torch.device('cuda')
        pat_data_path = 'datasets/pat_dat_aug'+str(months_prior)+'.pickle'
        emb_path = 'datasets/gpt_emb1536.pickle'
        save_path = mod+str(embedding_dim)+'_'+months_prior
        rs_path = 'output/'+mod+'_binary'+str(embedding_dim)+'_'+months_prior+'.pickle'

        # with open('model/train_losses_'+save_path+'.pickle', 'rb') as handle:
        #     train_loss = pickle.load(handle)
        # with open('model/val_losses_'+save_path+'.pickle', 'rb') as handle:
        #     val_loss = pickle.load(handle)

        # plt.figure()
        # plt.plot(train_loss)
        # plt.plot(val_loss)
        # plt.show()

        ############################Generate Risk Scores####################################
        if os.path.exists(rs_path):
            print(rs_path+' exist')    
        if overwrite==1:
            print('generate risk scores')
            data_loader = DataLoader(pat_data_path, use_graph_embeddings = False)  
            data = data_loader.reidx_dat
            pat_c2i = data_loader.pat_c2i
            vocab_size = data_loader.vocab_size
            if mod == 'baseline':
                training = Training(data, save_path, vocab_size = vocab_size, embedding_dim = embedding_dim, output_dim = output_dim, device = device)

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
            
            model = training.model

            test_data = training.data_splits.test_data
            _, _, test_ids = training.data_splits.split_ids()
            model.load_state_dict(torch.load('model/'+save_path+'.pt', map_location=device))

            def collate_fn(batch):
                seq=  pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
                age=  pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
                label = torch.tensor([item[2] for item in batch])
                return seq, age, label
            res = {}
            all_outputs, all_labels = [], []

            from torch.utils.data import DataLoader
            test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

            model.eval() 
            with torch.no_grad():
                for sequences, age, labels in test_loader:
                    sequences, age, labels = sequences.to(device), age.to(device), labels.float().to(device)

                    # Forward pass
                    outputs = model(sequences, age)
                    all_outputs.extend(outputs.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())
                    

            res['ids'] = test_ids
            res['raw_scores'] = all_outputs
            res['labels'] = all_labels

            import pickle
            with open(rs_path, 'wb') as handle:
                pickle.dump(res, handle)
        ######################################################################################

