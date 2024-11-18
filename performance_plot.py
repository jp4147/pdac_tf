import pickle
from evaluate_performance import sens_spec_range
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import pandas as pd

# m = '60m'
results = []
for m in ['6m', '12m', '36m', '60m']:
    m_rev = {'6m':'3-6m', '12m':'6-12m', '36m':'12-36m', '60m':'36-60m'}
    mods = ['baseline', 'gpt']
    embedding_dim = 1536
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    for mod in mods:
        rs_path = 'rs_'+mod+str(embedding_dim)+'.pickle'
        sen, spec, tpr, fpr = sens_spec_range(rs_path, spec_from = 0, spec_to = 1.0)

        specAt = 0.999
        current_spec = spec[m]
        current_sen = sen[m]

        # Convert the dictionary keys and values to arrays for processing
        spec_values = np.array(list(current_spec.keys()))  # Specificity keys
        sen_values = np.array(list(current_sen.values()))  # Sensitivity values
        # spec_reversed = spec_values[::-1]
        # sen_reversed = sen_values[::-1]

        sen_95spec = np.interp(specAt, spec_values, sen_values)

        roc_auc = auc(fpr[m], tpr[m])
        label='AUROC '+m_rev[m]+','+mod+': {:.3f}'.format(roc_auc)
        ax1.plot(fpr[m], tpr[m], label = label)
        ax2.plot(sen[m].keys(), sen[m].values(), label = label)

        # Save results to the list
        results.append({
            'Interval': m_rev[m],
            'Model': mod,
            'AUROC': roc_auc,
            'Sen@Spec=0.999': sen_95spec
        })

    ax1.set_xlabel('False Positive Rate', fontsize =15)
    ax1.set_ylabel('True Positive Rate', fontsize =15)
    ax1.legend(loc="lower right", fontsize =12)

    ax2.set_xlabel('Specificity', fontsize =15)
    ax2.set_ylabel('Sensitivity', fontsize =15)
    ax2.legend(loc="lower left", fontsize =12)

    # Show plot
    plt.tight_layout()
    plt.savefig(f'output/plot_{m}.png')  # Save plot as a PNG file

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results table to a CSV file
    results_df.to_csv('output/sen_95spec_results.csv', index=False)    
    
