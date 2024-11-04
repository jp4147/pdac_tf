import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from tqdm import tqdm

class DataLoader:
    def __init__(self, pat_data_path, use_graph_embeddings=False):
        self.networkx_graph_path = 'graph/graph.pickle'
        self.node_embeddings_path = 'graph/node_embeddings.npy'
        self.rel_embeddings_path = 'graph/rel_embeddings.npy'
        
        # self.node_embeddings_path = 'graph/128 embedding/node_embeddings.npy'
        # self.rel_embeddings_path = 'graph/128 embedding/rel_embeddings.npy' 
        # self.nodes, self.nx_G = self.load_graph_data()
               
        self.pat_data_path = pat_data_path
        self.vocab_size = None
        
        with open(self.pat_data_path, 'rb') as f:
            self.pat_data = pickle.load(f)
        
        self.reidx_dat, self.pat_c2i  = self.data_reindex()
        
        if use_graph_embeddings:
            self.final_embeddings = self.generate_final_embeddings()
        
    def load_graph_data(self):
        with open(self.networkx_graph_path, 'rb') as f:
            nx_G = pickle.load(f)
        nodes = np.array(list(sorted(nx_G.nodes())))
        return nodes, nx_G  
    
    def data_reindex(self):
        patlist = list(self.pat_data.keys())
        seq = 'concept_dx' 
        age = 'age_norm'
        label = 'label'
        
        print('assign reindex to concepts')
        pat_seq = [v[seq] for k,v in self.pat_data.items()]
        unique_seq = set(idx for idx_lst in pat_seq for idx in idx_lst)
        pat_c2i = {c: i+1 for i, c in enumerate(unique_seq)}
        
        print('data reindexing')
        reidx_dat = {}
        for pat in tqdm(patlist):
            reidx_dat[pat] = {}
            reidx_dat[pat]['concept'] = [pat_c2i[i] for i in self.pat_data[pat][seq]]
            reidx_dat[pat]['age'] = self.pat_data[pat][age]
            reidx_dat[pat]['label'] = self.pat_data[pat][label]
            
        print('vocab_size:', len(list(pat_c2i.keys()))+1)
        self.vocab_size = len(list(pat_c2i.keys()))+1
        
        return reidx_dat, pat_c2i
    
    def generate_final_embeddings(self):
        node_embeddings = np.load(self.node_embeddings_path)  
        pat_dx_in_concepts = list(self.pat_c2i.keys())
        node2idx = {n: i for i, n in enumerate(self.nodes)}
        pat_gnn_emb = [node_embeddings[node2idx[c]] for c in pat_dx_in_concepts]
        pre_trained_embeddings = torch.tensor(np.array(pat_gnn_emb), dtype=torch.float32)
        padding_embedding = torch.zeros(1, self.embedding_dim)
        final_embeddings = torch.cat((padding_embedding, pre_trained_embeddings), dim=0)
        return final_embeddings