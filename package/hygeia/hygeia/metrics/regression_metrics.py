import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np

def create_mask(data):
    """Masks zeros"""
    return (data != 0)

def correlation_coefficient(preds, ground_truth, mask_zeros=True, method='pearson'):
    """
    Compute the correlation coefficient between two arrays based on the specified method.

    Parameters
    ------------
        preds: np array
            Predicted tensor of shape (n_samples, n_features)

        ground_truth: Torch tensor
            Ground truth tensor of shape (n_samples, n_features)
        
    Returns
    ------------
        Dictionary with computer mtrics at feature and sample level
    """
    if mask_zeros:
        preds = preds[ground_truth != 0]
        ground_truth = ground_truth[ground_truth != 0]
    
    if method == 'pearson':
        mean_x = np.mean(preds)
        mean_y = np.mean(ground_truth)
        numerator = np.sum((preds - mean_x) * (ground_truth - mean_y))
        denominator = np.sqrt(np.sum((preds - mean_x)**2) * np.sum((ground_truth - mean_y)**2))
        if denominator == 0:
            return 0.0
        return numerator / denominator
    elif method == 'spearman':
        spearman_corr, _ = spearmanr(preds, ground_truth)
        return spearman_corr

def get_manual_correlations(preds, ground_truth, mask_zeros = False):
    """
    Compute the average Pearson, Spearman, and R2 for each feature and sample across two tensors.
    Computed without using scipy for pearson (to solve nan issues)

    Parameters
    ------------
        preds: np array
            Predicted tensor of shape (n_samples, n_features)
        ground_truth: Torch tensor
            Ground truth tensor of shape (n_samples, n_features)
        mask_zeros: Bool
            Whether to mask ground truth zero instance before computing correlations
        
    Returns
    ------------
        Dictionary with computer metrics at feature and sample level
    
    """
    if preds.shape != ground_truth.shape:
        raise ValueError("The two tensors must have the same shape.")

    if mask_zeros:
        mask = create_mask(ground_truth)
        preds = preds[mask]
        ground_truth = ground_truth[mask]

    n_samples, n_features = preds.shape

    metrics = {
        'Average pearson gene lvl': np.mean([correlation_coefficient(preds[:, i], ground_truth[:, i], mask_zeros, method='pearson') for i in range(n_features)]),
        'Average spearman gene lvl': np.mean([correlation_coefficient(preds[:, i], ground_truth[:, i], mask_zeros, method='spearman') for i in range(n_features)]),
        'Average R2 gene lvl': np.mean([r2_score(ground_truth[:, i], preds[:, i]) for i in range(n_features)]),
        'Average pearson sample lvl': np.mean([correlation_coefficient(preds[i, :], ground_truth[i, :], mask_zeros, method='pearson') for i in range(n_samples)]),
        'Average spearman sample lvl': np.mean([correlation_coefficient(preds[i, :], ground_truth[i, :], mask_zeros, method='spearman') for i in range(n_samples)]),
        'Average R2 sample lvl': np.mean([r2_score(ground_truth[i, :], preds[i, :]) for i in range(n_samples)])
    }

    return metrics


def eval_regression_metrics(ground_truth, preds, return_average = True):
    """
    Compute pearson, R2 and spearman at the gene and sample level
    Computed using scipy functions
    
    Parameters
    ------------
    ground_truth: torch.Tensor
        n_samples x n_features ground truth
    
    preds: torch.Tensor
        n_samples x n_features predictions
    
    return_average: Bool
        Whether to return scores as a list (for each gene / for each sample)
        or only the average
    
    Returns
    ------------
    metrics: Dict
        Dictionary with average pearson, spearman, R2, both at the gene
        and sample level.
    """
    if preds.shape[1] == 1:
        pearson_corr, _ = pearsonr(preds.squeeze(), ground_truth.squeeze())
        r2 = r2_score(ground_truth.squeeze(), preds.squeeze())
        spearman_corr,_ = spearmanr(preds.squeeze(), ground_truth.squeeze())

        metrics = {
            'Pearson': pearson_corr,
            'R2': r2,
            'Spearman': spearman_corr
        }
        return metrics

    r2_scores_gene = []
    pearson_correlations_gene = []
    spearman_correlations_gene = []
    for i in range(preds.shape[1]):
        y_true = []
        y_pred = []

        y_true.extend(ground_truth[:, i])
        y_pred.extend(preds[:, i])

        r2 = r2_score(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)

        r2_scores_gene.append(r2)
        pearson_correlations_gene.append(pearson_corr)
        spearman_correlations_gene.append(spearman_corr)

    avg_pearson_gene_lvl = np.array(pearson_correlations_gene).mean()
    avg_r2_gene_lvl = np.array(r2_scores_gene).mean()
    avg_spearman_gene_lvl = np.array(spearman_correlations_gene).mean()

    r2_scores_sample = []
    pearson_correlations_sample = []
    spearman_correlations_sample = []
    
    for i in range(len(preds)):
        pred_i = preds[i]
        r2 = r2_score(pred_i, ground_truth[i])
        pearson_corr, _ = pearsonr(pred_i, ground_truth[i])
        spearman_corr, _ = spearmanr(pred_i, ground_truth[i])
     
        r2_scores_sample.append(r2)
        pearson_correlations_sample.append(pearson_corr)
        spearman_correlations_sample.append(spearman_corr)
    
    avg_pearson_sample_lvl = np.array(pearson_correlations_sample).mean()
    avg_r2_sample_lvl = np.array(r2_scores_sample).mean()
    avg_spearman_sample_lvl = np.array(spearman_correlations_sample).mean()
    
    if return_average:
        metrics = {
            'Average pearson gene lvl': avg_pearson_gene_lvl,
            'Average spearman gene lvl': avg_spearman_gene_lvl,
            'Average R2 gene lvl': avg_r2_gene_lvl,
            'Average pearson sample lvl': avg_pearson_sample_lvl,
            'Average spearman sample lvl': avg_spearman_sample_lvl,
            'Average R2 sample lvl': avg_r2_sample_lvl
        }
    else:
        metrics = {
            'Pearson gene lvl': np.array(pearson_correlations_gene),
            'Spearman gene lvl': np.array(spearman_correlations_gene),
            'R2 gene lvl': np.array(r2_scores_gene),
            'Pearson sample lvl': np.array(pearson_correlations_sample),
            'Spearman sample lvl': np.array(spearman_correlations_sample),
            'R2 sample lvl': np.array(r2_scores_sample)
        }
    
    return metrics