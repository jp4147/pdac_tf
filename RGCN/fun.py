import torch
import numpy as np
import json
class RandomCorruptionNegativeSampler(object):
    def __init__(self, n, use_cache=False):
        self.n = n

        self.use_cache = use_cache
        self.cache = {}

    def __call__(self, g, eids):
        all_nodes = g.nodes('_N')
        src, dst = g.find_edges(eids)

        repl_src = torch.where(torch.rand((len(src), self.n)) > 0.5, 1,
                               0).to(g.device)
        repl_dst = 1 - repl_src

        repl_node = all_nodes[torch.randint(len(all_nodes), repl_src.shape)]

        new_src = torch.where(repl_src == 1, repl_node,
                              torch.reshape(src, (-1, 1)).repeat([1, self.n]))
        new_dst = torch.where(repl_dst == 1, repl_node,
                              torch.reshape(dst, (-1, 1)).repeat([1, self.n]))

        return torch.reshape(new_src, [-1]), torch.reshape(new_dst, [-1])
    
def print_first_proc(*values, proc_id=0):
    if proc_id == 0:
        print(*values, )
        
def save_weights(model, G, n_rels, model_path):
    device = model.device
    G = G.to(device)
    node_embeddings = model.rgcn_block(G).detach().cpu().numpy()

    relation_embeddings = model.hake.rel_embedding(
        torch.arange(n_rels).to(device)).cpu().detach().numpy()

    np.save(model_path / 'node_embeddings.npy', node_embeddings)
    np.save(model_path / 'rel_embeddings.npy', relation_embeddings)

    lam = model.hake.lam.cpu().detach().numpy()
    lam2 = model.hake.lam2.cpu().detach().numpy()
    with open(model_path / 'lambda.json', 'w') as f:
        json.dump({'lambda': float(lam), 'lambda2': float(lam2)}, f)

    torch.save(model.state_dict(), model_path / 'saved_model_weights.pt')

    print(f"Node embeddings saved to {model_path / 'node_embeddings.npy'}")
    print(f"Relational matrices saved to {model_path / 'rel_embeddings.npy'}")
    print(
        f"Full model weights saved to {model_path / 'saved_model_weights.pb'}")
    
def cleanup():
    torch.distributed.destroy_process_group()