from load_data import DataLoader
import pickle
import pandas as pd

import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def ci95(data):

    # Sample mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    n = len(data)

    # Calculate the 95% confidence interval
    confidence_level = 0.95
    alpha = 1 - confidence_level
    df = n - 1  # degrees of freedom

    t_critical = stats.t.ppf(1 - alpha/2, df)
    margin_of_error = t_critical * (std_dev / np.sqrt(n))

    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    return (ci_lower, ci_upper)

def fea_imp_bargraph(patids, fea_model, pat_i2c, save_path = ''):
    agg_fea = {}

    for p in patids:
        v = fea_model[p]
        for info in v:
            feature = info[0]
            score = info[1]
            if feature not in agg_fea:
                agg_fea[feature] = []
            agg_fea[feature].append(score)

    # Filter out features with too few scores
    agg_fea = {feature: scores for feature, scores in agg_fea.items() if len(scores) > 1}   
    agg_importances = {feature: [np.mean(importances), ci95(importances)] for feature, importances in agg_fea.items()}
    freq_fea = {feature: len(importances) for feature, importances in agg_fea.items()}

    # sorted_features = sorted(freq_fea.items(), key=lambda x: x[1], reverse=True)
    # common_fea = [concept_dict[pat_i2c[int(i[0])]] for i in sorted_features[0:15]]
    # common_fea_scores = [agg_fea[i[0]] for i in sorted_features[0:15]]
    # num_fea = [i[1] for i in sorted_features[0:15]]
    
    sorted_features = sorted(agg_importances.items(), key=lambda x: x[1][0], reverse=True)
    common_fea = [concept_dict[pat_i2c[int(i[0])]] for i in sorted_features[0:15]]
    common_fea_scores = [agg_fea[i[0]] for i in sorted_features[0:15]]
    num_fea = [freq_fea[i[0]] for i in sorted_features[0:15]]

    weighted_freq = {feature: mean_importance[0] * freq_fea[feature] for feature, mean_importance in agg_importances.items()}
    
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(6, 10))

    ax1.boxplot(common_fea_scores, positions=list(range(len(common_fea))), widths=0.6)

    # Set the x-axis labels
    ax1.set_xticks(list(range(len(common_fea_scores))))
    # # 36m x labels
    # common_fea[-4] = "Lung disease with Sjogren's Disease Symptoms"
    ax1.set_xticklabels(common_fea, rotation = 90)


    # Set the title and labels
    ax1.set_ylabel('Feature weights', fontsize = 12)

    ax2 = ax1.twinx()
    ax2.plot(list(range(len(common_fea))), num_fea, color='r', marker='o', linestyle='-', label='', alpha = 0.5)   
    ax2.set_ylabel('# of patients having the feature', color = 'r', fontsize = 12)
    # plt.show()

    plt.tight_layout()  # Optional: helps prevent label cut-off
    if len(save_path)==0:
        print("ERROR: No save_path")
    plt.savefig('output/'+save_path+'.png', dpi=300, bbox_inches='tight')  # Change file name/format as needed
    
    return weighted_freq

print('load OMOP concept')
concept = pd.read_csv('datasets/concept.csv', low_memory=False)
concept_dict = dict(zip(concept['concept_id'], concept['concept_name']))

print('collect feature index information')

pat_i2c = {}
for months_prior in ['3-6m', '6-12m', '12-36m']:
    fea_index_path = 'datasets/pat_i2c_'+months_prior+'.pickle'
    if os.path.exists(fea_index_path):
        with open(fea_index_path, 'rb') as h:
            pat_i2c[months_prior] = pickle.load(h)
    else:
        pat_data_path = 'datasets/pat_dat_aug'+str(months_prior)+'.pickle'
        data_loader = DataLoader(pat_data_path, use_graph_embeddings = False)  
        pat_c2i = data_loader.pat_c2i

        pat_i2c[months_prior] = {v:k for k, v in pat_c2i.items()}

        with open('datasets/pat_i2c_'+months_prior+'.pickle', 'wb') as h:
            pickle.dump(pat_i2c[months_prior], h)

print('load LIME features and save plots')

for mod in ['gpt', 'baseline']:
    for months_prior in ['3-6m', '6-12m', '12-36m']:
        with open('output/'+mod+'_binary1536_'+months_prior+'_fea.pickle', 'rb') as handle:
            fea = pickle.load(handle)

        save_path = 'fea_'+mod+'_'+months_prior
        wf = fea_imp_bargraph(list(fea.keys()), fea, pat_i2c[months_prior], save_path)
    