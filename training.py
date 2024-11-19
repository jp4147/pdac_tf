from split_data import DataSplit

from model import TransformerModel
import torch
import torch.nn as nn
import copy
import pickle

class Training:
    def __init__(self, data, save_path, vocab_size=100, embedding_dim=32, hidden_size=64, output_dim=5, final_embeddings = None, device: torch.device = torch.device('cpu')):
        self.data = data
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.final_embeddings = final_embeddings
        self.device = device 
        self.save_path = save_path
        
        self.data_splits = DataSplit(data)
        self.train_loader = self.data_splits.train_loader
        self.val_loader = self.data_splits.val_loader
        
        self.model = TransformerModel(self.vocab_size, self.embedding_dim, self.hidden_size, num_heads=2, 
                                      num_encoder_layers=2, output_dim=self.output_dim, pre_trained_weights = final_embeddings, 
                                      dropout = 0.5, device = self.device).to(self.device)
        
    def training(self, lr = 0.00001):
        criterion = nn.BCEWithLogitsLoss()  # or any other appropriate loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # or any other optimizer

        num_epochs = 500
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model = None

        epochs_without_improvement = 0
        patience = 10

        for epoch in range(num_epochs):
            epoch_train_losses = []  # to store losses for this epoch

            self.model.train()  # ensure the model is in train mode
            for sequences, age, labels in self.train_loader:
                sequences, age, labels = sequences.to(self.device), age.to(self.device), labels.float().to(self.device)

                # Forward pass
                if self.output_dim ==1:
                    outputs = self.model(sequences, age).squeeze(1)
                else:
                    outputs = self.model(sequences, age)
                loss = criterion(outputs, labels)
                epoch_train_losses.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}')

            # Evaluate on validation set
            self.model.eval()  # ensure the model is in eval mode
            with torch.no_grad():
                val_losses_this_epoch = []
                for sequences, age, labels in self.val_loader:
                    sequences, age, labels = sequences.to(self.device), age.to(self.device), labels.float().to(self.device)
                    
                    if self.output_dim ==1:
                        outputs = self.model(sequences, age).squeeze(1)
                    else:
                        outputs = self.model(sequences, age)
                    loss = criterion(outputs, labels)
                    val_losses_this_epoch.append(loss.item())

                avg_val_loss = sum(val_losses_this_epoch) / len(val_losses_this_epoch)
                val_losses.append(avg_val_loss)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss}')

            # if this model is the best so far, save it
            if avg_val_loss < best_val_loss and epoch>10:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0

                torch.save(best_model, 'model/'+self.save_path+'.pt')
            else:
                if epoch>10:
                    epochs_without_improvement +=1
                if epochs_without_improvement == patience:
                    print("Stopping training due to lack of improvement in validation loss.")
                    break
            with open('model/train_losses_'+self.save_path+'.pickle', 'wb') as handle:
                pickle.dump(train_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('model/val_losses_'+self.save_path+'.pickle', 'wb') as handle:
                pickle.dump(val_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)    

        return best_model, train_losses, val_losses