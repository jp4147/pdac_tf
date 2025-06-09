import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

from training import Training
from load_data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

class MyLime: 
    def __init__(self, sequence_str_initial, model, age, device, la_idx):
        self.sequence_str_initial = sequence_str_initial
        self.model = model
        self.age = age
        self.device = device
        self.la_idx = la_idx
        
    def logits_to_probs(self, logits):
        return 1 / (1 + np.exp(-logits))
    
    def model4lime(self, text_inputs):
        predictions = []
        self.model.eval()
    
        # Ensure text_inputs is a list, even if it's just one sample
        if not isinstance(text_inputs, list):
            text_inputs = [text_inputs]
    
        for text_input in text_inputs:
            # Convert the string input back to a tensor
            sequence_list = [int(code) for code in text_input.split(' ') if code.strip()]
            seq_pad = len(self.sequence_str_initial.split(' '))-len(sequence_list)
            seq_tensor = torch.tensor([0]*seq_pad+sequence_list)
        
            # Add an extra dimension to match the input shape expected by the model
            seq_tensor = seq_tensor.unsqueeze(0).to(self.device)
            
            # Assuming age is fixed or predetermined for this explanation
            age_tensor = torch.tensor(self.age)
            age_tensor = age_tensor.unsqueeze(0).to(self.device)
    
            # Get model prediction for this input
            with torch.no_grad():
                prediction = self.model(seq_tensor, age_tensor)[0]           
    
            # Detach the prediction and move to CPU if necessary, then convert to NumPy
            prediction_np = prediction.detach().cpu().numpy()
            # pred_prob = self.logits_to_probs(prediction_np[self.la_idx])
            pred_prob = self.logits_to_probs(prediction_np)
            predictions.append(pred_prob)
    
            low_high_prediction = np.array([[1 - pred, pred] for pred in predictions])
            low_high_prediction = low_high_prediction.squeeze(2)
    
        # Convert the list of predictions to a NumPy array
        return low_high_prediction
    

explainer = LimeTextExplainer(class_names=["Low Risk", "High Risk"])

def seq2str(seq):
    seq_str = ' '.join(map(str, seq))
    return seq_str

def individual_test(pat_data_sample, model, show = 1):
    seq = seq2str(pat_data_sample[0])
    age = pat_data_sample[1]
    la_idx = pat_data_sample[2]
    my_lime = MyLime(seq, model, age, device, la_idx)
    exp = explainer.explain_instance(seq, my_lime.model4lime)
    if show == 1:
        exp.show_in_notebook(text=True)
    else:
        return exp.as_list()
    
def generate_feature_scores(pat_data, model):
    feature_importances = {}
    for patid, seq_age_i in tqdm(pat_data.items()):
        feature_importances[patid] = individual_test(seq_age_i, model, show = 0)
    return feature_importances


embedding_dim = 1536
output_dim = 1

for mod in ['gpt', 'baseline']:
    for months_prior in ['3-6m', '6-12m', '12-36m']:
        from load_data import DataLoader
        torch.cuda.set_device(0)
        device = torch.device('cuda')
        pat_data_path = 'datasets/pat_dat_aug'+str(months_prior)+'.pickle'
        emb_path = 'datasets/gpt_emb1536.pickle'
        save_path = 'model/'+mod+str(embedding_dim)+'_'+months_prior

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
        model.load_state_dict(torch.load(save_path+'.pt', map_location=device))

        print(mod, months_prior, 'load HR ids')
        with open('output/_HR_'+mod+'_binary1536_'+months_prior+'.pickle', 'rb') as h:
            hr = pickle.load(h)
        if len(hr['FP'])>300:
            hr_ids = hr['TP']+random.sample(hr['FP'], 300)
        else:
            hr_ids = hr['TP']+hr['FP']
        hr_data= {}
        for i in hr_ids:
            idx = test_ids.index(i)
            d = test_data[idx]
            hr_data[i] = (d[0].tolist(), d[1].tolist(), 0)

        hr_fea = generate_feature_scores(hr_data, model)

        with open('output/'+mod+'_binary1536_'+months_prior+'_fea.pickle', 'wb') as handle:
            pickle.dump(hr_fea, handle)