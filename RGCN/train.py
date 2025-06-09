import torch
from fun import print_first_proc, save_weights
from tqdm import tqdm
from contextlib import nullcontext
from torchmetrics.functional import retrieval_reciprocal_rank
import numpy as np
from torchmetrics import RetrievalMRR
from pathlib import Path
import pickle
# import torch.distributed as dist

def train(model,
          model_output_path,
          proc_id,
          train_dataloader,
          val_dataloader,
          loss_fn,
          optimizer,
          negative_samples,
          epochs=5,
          early_stopping=False,
          patience=5):
    def _get_triples(positive_graph, negative_graph):
        pos_srcs = positive_graph.edges()[0]
        pos_dests = positive_graph.edges()[1]

        pos_triples = torch.stack(
            [pos_srcs, positive_graph.edata['rel_type'], pos_dests], dim=-1)

        neg_srcs = negative_graph.edges()[0]
        neg_dests = negative_graph.edges()[1]
        neg_triples = torch.reshape(
            torch.stack([
                neg_srcs,
                torch.repeat_interleave(positive_graph.edata['rel_type'],
                                        negative_samples), neg_dests
            ],
                        dim=-1),
            (pos_triples.shape[0], -1, pos_triples.shape[1]))

        x = torch.cat([torch.unsqueeze(pos_triples, 1), neg_triples], dim=1)
        y = torch.cat([
            torch.ones((x.shape[0], 1)),
            torch.zeros((x.shape[0], neg_triples.shape[1]))
        ],
                      dim=-1)

        return x, y

    history = {
        "tr_loss": [],
        "tr_mrr": [],
        "val_loss": [],
        "val_mrr": [],
    }
    best_model_state = model.state_dict()
    best_mrr = 0
    epochs_without_improvement = 0
    for epoch in range(epochs):
        print_first_proc(f'Epoch {epoch+1}/{epochs}', proc_id=proc_id)
        model.train()

        batch_losses = []
        batch_mrrs = []

        with tqdm(train_dataloader) if proc_id == 0 else nullcontext() as td:
            td = td if proc_id == 0 else train_dataloader
            for step, (input_nodes, positive_graph, negative_graph,
                       blocks) in enumerate(td):

                x, y = _get_triples(positive_graph, negative_graph)

                scores = model((x, blocks))

                loss = loss_fn(scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 5 == 0:
                    sub_mrrs = torch.zeros(scores.shape[0])
                    for i in range(scores.shape[0]):
                        target_i = torch.tensor(y[i], device=scores[i].device)
                        # print(type(y[i]), isinstance(y[i], torch.Tensor))
                        # print('target_i type', type(target_i))
                        sub_mrr = retrieval_reciprocal_rank(scores[i], target_i)
                        # sub_mrr = retrieval_reciprocal_rank(scores[i], y[i])
                        sub_mrrs[i] = sub_mrr

                    mrr = torch.mean(sub_mrrs)

                    batch_losses.append(loss.item())
                    batch_mrrs.append(mrr.item())

                    if proc_id == 0:
                        td.set_postfix(
                            {
                                "loss": "%.03f" % np.mean(batch_losses),
                                "mrr": "%.03f" % np.mean(batch_mrrs),
                            },
                            refresh=False)
            history["tr_loss"].append(np.mean(batch_losses))
            history["tr_mrr"].append(np.mean(batch_mrrs))

        model.eval()

        val_mrr_metric = RetrievalMRR('error')
        val_batch_losses = []
        val_batch_mrrs = []
        val_batch_imrrs = []

        with tqdm(val_dataloader) if proc_id == 0 else nullcontext(
        ) as vd, torch.no_grad():
            vd = vd if proc_id == 0 else val_dataloader
            for input_nodes, positive_graph, negative_graph, blocks in vd:
                x, y = _get_triples(positive_graph, negative_graph)
                scores = model((x, blocks))

                loss = loss_fn(scores)

                sub_mrrs = torch.zeros(scores.shape[0])
                for i in range(scores.shape[0]):
                    target_i = torch.tensor(y[i], device=scores[i].device)
                    # print(type(y[i]), isinstance(y[i], torch.Tensor))
                    # print('target_i type', type(target_i))
                    sub_mrr = retrieval_reciprocal_rank(scores[i], target_i)
                    # sub_mrr = retrieval_reciprocal_rank(scores[i], y[i])
                    sub_mrrs[i] = sub_mrr

                mrr = torch.mean(sub_mrrs)

                val_batch_losses.append(loss.item())
                val_batch_mrrs.append(mrr.item())

            val_loss = torch.tensor(np.mean(val_batch_losses), device=model.device)
            val_mrr = torch.tensor(np.mean(val_batch_mrrs),
                                   device=model.device)

            print_first_proc(
                {
                    "val_loss": "%.03f" % val_loss,
                    "val_mrr": "%.03f" % val_mrr,
                },
                proc_id=proc_id)

            history["val_loss"].append(val_loss)
            history["val_mrr"].append(val_mrr)

            torch.distributed.all_reduce(val_mrr)
            val_mrr /= torch.distributed.get_world_size()
            torch.distributed.all_reduce(val_loss)
            val_loss /= torch.distributed.get_world_size()
            print_first_proc(
                {
                    "avg_loss": "%.03f" % val_loss,
                    "avg_mrr": "%.03f" % val_mrr,
                },
                proc_id=proc_id)

            if val_mrr > best_mrr:
                best_mrr = val_mrr
                best_model_state = model.state_dict()
                epochs_without_improvement = 0

                if (epoch + 1) % 2 == 0 and proc_id == 0:
                    print("Checkpointing...")
                    save_weights(model.module, model.module.G,
                                 model.module.n_rels, Path('./'+model_output_path+'/'))
            else:
                epochs_without_improvement += 1

            if early_stopping and epochs_without_improvement == patience:
                print(f"Early stopping proc_id {proc_id} at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)

    return history, best_mrr