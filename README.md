1. Prepare the Dataset

Run make_datasets.ipynb script: This will generate and save the following pickle files in the dataset folder:
- pc_diag.pickle: Contains pancreatic cancer diagnosis dates.
- pat_data.pickle: Contains patient data.

2. Evaluate baseline characteristics of the prepared dataset

3. Train the Model (multi-classification model)

Run model_training.ipynb script: This script performs the following actions:
- Trains the transformer model.
- Saves the trained model.
- Plots training and validation loss to confirm model training.
- Evaluates the model performance using the test set.
- Saves the predicted probabilities for the test set.

4. Evaluate the Model Performance
- Run evaluate.ipynb script: Generates plot showing sensitivities at ranges of specificities

###### Additional Analyses #####
1. Model training
- run_3m_exl: multi-classification model training excluding 0-3 month data
- run_binary.py: binary model training

2. Generate risk scores
3. Bootstrap testing
4. LIME analyses (interpretation.py, imp_fea_plot.py)
5. EHR model evaluation against traditional risk factors (rf_vs_EHR_testset.py)
