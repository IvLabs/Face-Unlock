import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_metrics(threshold, dist, psame):
    pdiff = np.logical_not(psame)
    preds = np.less(dist, threshold)
    
    true_accepts  = np.sum(np.logical_and(preds, psame))
    false_accepts = np.sum(np.logical_and(preds, pdiff))
    # false_rejects = np.sum(np.logical_and(np.logical_not(preds), psame))
    true_rejects  = np.sum(np.logical_and(np.logical_not(preds), pdiff))

    val = true_accepts  / (psame.sum() + 1e-16) # sensitivity, recall
    far = false_accepts / (pdiff.sum() + 1e-16) # (1-specificity)
    # frr = false_rejects / (psame.sum() + 1e-16)
    # frr = 1 - val
    # trr = true_rejects  / (pdiff.sum() + 1e-16) # specificity 
    # trr = 1 - far

    precision = true_accepts / (true_accepts + false_accepts + 1e-16)
    # a ratio of correctly predicted positive observations to the total predicted positive observations
    accuracy = (true_accepts + true_rejects) / (psame.sum() + pdiff.sum() + 1e-16)
    # a ratio of correctly predicted observation to the total observations

    return val, far, precision, accuracy

def evaluate(distances, labels, roc_curve=False):
    # mask = np.logical_not(np.eye(labels.shape[0], dtype='bool'))
    # Psame = labels[mask]
    # distances = distances[mask]
    thresholds = np.arange(0,4,0.01)
    num_thresholds = len(thresholds)
    vals = np.zeros(num_thresholds)
    fars = np.zeros(num_thresholds)
    youdens = np.zeros(num_thresholds)
    for idx, threshold in tqdm(enumerate(thresholds)):
        vals[idx], fars[idx], _, _ = calculate_metrics(threshold, distances, labels)
        youdens[idx] = vals[idx] - fars[idx] # sensitivity + specificity — 1
    best_threshold_idx = np.argmax(youdens) 

    fig = None
    if roc_curve:
        fig, ax = plt.subplots()
        ax.plot(fars, vals, marker = '.', c='r')
        ax.plot([0,1], [0,1], linestyle='--', label='random')
        ax.scatter(fars[best_threshold_idx], vals[best_threshold_idx], c='black', label='Optimal Threshold')
        ax.grid()
        ax.title.set_text('ROC Curve')
        ax.set_xlabel('False Accept Rate (FAR)')
        ax.set_ylabel('True Accept Rate (VAL)')
        ax.legend()
        # ax.savefig('ROC_curve')
        # ax.show()


    true_accept_rate, false_accept_rate, precision, accuracy = \
         calculate_metrics(thresholds[best_threshold_idx], distances, labels)

    return thresholds[best_threshold_idx], true_accept_rate, false_accept_rate, precision, accuracy, fig

