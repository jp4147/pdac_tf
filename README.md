1. Prepare the Dataset
Run make_datasets script: This will generate and save the following pickle files in the dataset folder:
- pc_diag.pickle: Contains pancreatic cancer diagnosis dates.
- pat_data.pickle: Contains patient data.
  
2. Train the Model
Execute model_training script: This script performs the following actions:
- Trains the transformer model.
- Saves the trained model.
- Plots training and validation loss to confirm model training.
- Evaluates the model performance using the test set.
- Saves the predicted probabilities for the test set.
