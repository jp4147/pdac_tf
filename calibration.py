import numpy as np
from netcal.binning import HistogramBinning
from netcal.scaling import BetaCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from netcal.metrics import ECE
from scipy.special import logit, expit
import pickle

mod = 'gpt' #gpt, baseline
months_prior = '3-6m' #'3-6m', '6-12m', '12-36m'

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_reliability_diagram(y_true, probs_uncalibrated, probs_calibrated, n_bins=20):
    # Compute calibration curves
    frac_pos_uncal, mean_pred_uncal = calibration_curve(y_true, probs_uncalibrated, n_bins=n_bins, strategy='quantile')
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, probs_calibrated, n_bins=n_bins, strategy='quantile')

    x_max = np.max(list(mean_pred_uncal)+list(mean_pred_cal))+0.001
    y_max = np.max(list(frac_pos_uncal)+list(frac_pos_cal))+0.001
    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.plot(mean_pred_uncal, frac_pos_uncal, 's-', label='Before Calibration', color='gray')
    plt.plot(mean_pred_cal, frac_pos_cal, 'o-', label='After Calibration', color='black')

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    # plt.xlim([0,0.3])
    # plt.ylim([0,0.3])
    # plt.xscale("log")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim([0, x_max])
    plt.ylim([0, y_max])
    # plt.show()
    plt.savefig('output/'+save_path+'.png', dpi=300, bbox_inches='tight')  # Change file name/format as needed

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
import numpy as np

def evaluate_metrics(true_labels, probs_before, probs_after):
    # AUROC
    auroc_before = roc_auc_score(true_labels, probs_before)
    auroc_after = roc_auc_score(true_labels, probs_after)
    
    print("==== Performance Metrics ====")
    print(f"AUROC before calibration: {auroc_before:.4f}")
    print(f"AUROC after calibration : {auroc_after:.4f}")

    # AUPRC
    auprc_before = average_precision_score(true_labels, probs_before)
    auprc_after = average_precision_score(true_labels, probs_after)
    print(f"AUPRC before calibration: {auprc_before:.4f}")
    print(f"AUPRC after calibration : {auprc_after:.4f}")

    # Max F1 Score
    prec_b, recall_b, thresh_b = precision_recall_curve(true_labels, probs_before)
    f1s_b = 2 * (prec_b * recall_b) / (prec_b + recall_b + 1e-8)
    max_f1_b = np.max(f1s_b)

    prec_a, recall_a, thresh_a = precision_recall_curve(true_labels, probs_after)
    f1s_a = 2 * (prec_a * recall_a) / (prec_a + recall_a + 1e-8)
    max_f1_a = np.max(f1s_a)

    print(f"Max F1 before calibration: {max_f1_b:.4f}")
    print(f"Max F1 after calibration : {max_f1_a:.4f}")

for mod in ['gpt', 'baseline']:
    for months_prior in ['3-6m','6-12m', '12-36m']:
        embedding_dim = 1536
        val_rs_path = 'output/'+mod+'_binary'+str(embedding_dim)+'_'+months_prior+'_val.pickle'
        test_rs_path = 'output/'+mod+'_binary'+str(embedding_dim)+'_'+months_prior+'.pickle'
        save_path = 'calibration_binary_model_'+mod+months_prior

        with open(val_rs_path, 'rb') as h:
            scores_val = pickle.load(h)
        with open(test_rs_path, 'rb') as h:
            scores_test = pickle.load(h)
        plot_bins = 30
        num_bins = 30  # You can adjust the number of bins as needed
        # calibrator = HistogramBinning(bins=num_bins)
        # calibrator = BetaCalibration()
        # calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator = LogisticRegression()

        epsilon = 1e-15
        val_scores = np.array(scores_val['raw_scores']).flatten()
        val_labels = np.array(scores_val['labels'])
        val_probs = 1 / (1 + np.exp(-val_scores))  # sigmoid
        val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

        # calibrator.fit(val_probs_clipped, val_labels)
        calibrator.fit(val_scores.reshape(-1, 1), val_labels)

        test_scores = np.array(scores_test['raw_scores']).flatten()
        test_labels = np.array(scores_test['labels'])
        test_probs = 1 / (1 + np.exp(-test_scores))  # sigmoid
        test_probs_clipped = np.clip(test_probs, epsilon, 1 - epsilon)
        # calibrated_probs = calibrator.transform(test_probs_clipped)
        calibrated_probs = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]

        # Initialize ECE metric
        ece = ECE(bins=num_bins)

        # Compute ECE before calibration
        ece_before = ece.measure(test_probs, test_labels)

        # Compute ECE after calibration
        ece_after = ece.measure(calibrated_probs, test_labels)

        with open('output/calibration_binary_model_log.txt', 'a') as f:
            print(mod, months_prior, file = f)
            print(f"ECE before calibration: {ece_before}", file = f)
            print(f"ECE after calibration: {ece_after}", file = f)
            print('------------------------------------------------', file = f)

        # Example usage
        # probs_before = predicted probabilities before calibration
        # probs_after = calibrated predicted probabilities
        # true_labels = binary ground-truth labels

        evaluate_metrics(test_labels, test_probs, calibrated_probs)
        plot_reliability_diagram(test_labels, test_probs, calibrated_probs, n_bins = plot_bins)
