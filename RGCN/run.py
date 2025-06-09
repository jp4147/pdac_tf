import argparse
from train_graph import train_graph_embeddings
import os
os.environ['DGL_GRAPHBOLT_DISABLED'] = '1'
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def train_graph_embeddings_mp(nx_G,
                              model_output_path,
                              learning_rate=0.01,
                              batch_size=256,
                              epochs=10,
                              embedding_size=300,
                              n_layers=2,
                              negative_samples=128,
                              patience=5,
                              num_gpus=2,
                              regularizer='basis',
                              basis=2,
                              dropout=0.0):
    import torch.multiprocessing as mp
    print(f"Using {num_gpus} gpus")
    try:
        mp.spawn(train_graph_embeddings,
                 args=(list(range(num_gpus)), nx_G, model_output_path,
                       learning_rate, batch_size, epochs, embedding_size,
                       n_layers, negative_samples, patience, regularizer,
                       basis, dropout),
                 nprocs=num_gpus)
    except KeyboardInterrupt:
        cleanup()


if __name__ == '__main__':

    num_gpus = 3

    parser = argparse.ArgumentParser(
        description="Create embeddings for nodes in a networkx graph")
    parser.add_argument('nx_g_path', help='Pickled NetworkX graph')
    parser.add_argument('model_output_path',
                        help='Path for model output and weights')

    args = parser.parse_args()

    train_graph_embeddings_mp(args.nx_g_path,
                              args.model_output_path,
                              num_gpus=num_gpus,
                              embedding_size=32,
                              batch_size=256,
                              learning_rate=0.001,
                              negative_samples=512,
                              n_layers=2,
                              epochs=300,
                              patience=10,
                              basis = 2, 
                              regularizer='basis')