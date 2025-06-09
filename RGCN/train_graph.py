import torch
from pathlib import Path
import networkx as nx
import dgl
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from fun import RandomCorruptionNegativeSampler, print_first_proc, save_weights, cleanup
from model import create_model
from loss import AdverserialLoss
from train import train
import numpy as np
import pickle

RANDOM_SEED = 2
    
def train_graph_embeddings(proc_id,
                           devices,
                           nx_g_path,
                           model_output_path,
                           learning_rate=0.01,
                           batch_size=256,
                           epochs=10,
                           embedding_size=300,
                           n_layers=2,
                           negative_samples=128,
                           patience=5,
                           regularizer='basis',
                           basis=6,
                           dropout=0.0):
    # Initialize distributed training context.
    dev_id = devices[proc_id]
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(backend='gloo',
                                             init_method=dist_init_method,
                                             world_size=len(devices),
                                             rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(backend='gloo',
                                             init_method=dist_init_method,
                                             world_size=len(devices),
                                             rank=proc_id)
        
    # Load graph

    model_path = Path(model_output_path)

    if not model_path.exists():
        model_path.mkdir(exist_ok=True)

    print_first_proc("Converting networkx to dgl", proc_id=proc_id)
    with open(nx_g_path, 'rb') as handle:
        nx_G = pickle.load(handle)

    G = dgl.from_networkx(nx_G,
                          node_attrs=['concept_id'],
                          edge_attrs=['id', 'rel_type']).to(device)

    print_first_proc("Calculating norms", proc_id=proc_id)

    edge_rel_counts = {}
    dests = G.edges()[1]
    rel_types = G.edata['rel_type']

    
    edges_range = tqdm(range(len(dests))) if proc_id == 0 else range(
        len(dests))
    for i in edges_range:
        dest = int(dests[i])
        rel_type = int(rel_types[i])
        if dest not in edge_rel_counts:
            edge_rel_counts[dest] = {}
            edge_rel_counts[dest][rel_type] = 1
        else:
            if rel_type in edge_rel_counts[dest]:
                edge_rel_counts[dest][rel_type] += 1
            else:
                edge_rel_counts[dest][rel_type] = 1

    norms = []
    for i in range(len(dests)):
        dest = int(dests[i])
        rel_type = int(rel_types[i])
        norms.append(1.0 / max(edge_rel_counts[dest][rel_type], 1))
        
    norms = torch.Tensor(norms).to(device)
    G.edata['norm'] = torch.unsqueeze(norms, 1)

    print_first_proc("Generating samples", proc_id=proc_id)
    eids = G.edata['id']
    train_eids, val_eids = train_test_split(eids,
                                            test_size=0.2,
                                            random_state=RANDOM_SEED)

    val_eids, test_eids = train_test_split(val_eids,
                                           test_size=0.5,
                                           random_state=RANDOM_SEED)

    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    negative_sampler = RandomCorruptionNegativeSampler(negative_samples)

    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=negative_sampler)
    train_dataloader = dgl.dataloading.DataLoader(G,
                                                  train_eids,
                                                  sampler,
                                                  device=device,
                                                  use_ddp=True,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=False,
                                                  num_workers=0)

    val_dataloader = dgl.dataloading.DataLoader(G,
                                                val_eids,
                                                sampler,
                                                device=device,
                                                use_ddp=False,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=0)
    
    print_first_proc("Defining model", proc_id=proc_id)
    model_args = (device, G)
    model_kwargs = {
        'embedding_size': embedding_size,
        'n_layers': n_layers,
        'basis': basis,
        'dropout': dropout,
        'regularizer': regularizer
    }
    model, n_rels = create_model(*model_args, **model_kwargs)
    # model.load_state_dict(torch.load('./pt_output/saved_model_weights.pt'))

    model = model.to(device)

    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=None,
                                                          output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[device],
                                                          output_device=device)
        
    loss_fn = AdverserialLoss(margin=5.0, temp=2.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print_first_proc("Training model", proc_id=proc_id)
    
    history, this_best_mrr = train(model,
                                   model_output_path,
                                   proc_id,
                                   train_dataloader,
                                   val_dataloader,
                                   loss_fn,
                                   optimizer,
                                   negative_samples,
                                   epochs=epochs,
                                   early_stopping=True,
                                   patience=patience)
    
    outputdata = {
        'best_mrr': this_best_mrr.item(),
        # 'model_weights': model.state_dict(),
        # 'model_args': model_args,
        # 'model_kwargs': model_kwargs
    }

    print(f'Proc {proc_id} finished')

    outputs = [None for _ in range(len(devices))]

    # Waits for all gpus to be finished
    torch.distributed.all_gather_object(outputs, outputdata)

    all_best_mrrs = [output['best_mrr'] for output in outputs]
    best_mrr = np.max(all_best_mrrs)
    best_mrr_index = np.argmax(all_best_mrrs)

    if proc_id == best_mrr_index:

        print("Restoring best model weights")
        print(f'Best MRR: {best_mrr}')

        best_model = model
        save_weights(best_model.module, G, n_rels, model_path)
        with open(model_output_path+'/history.pickle', 'wb') as f:
            pickle.dump(history, f)

    cleanup()
