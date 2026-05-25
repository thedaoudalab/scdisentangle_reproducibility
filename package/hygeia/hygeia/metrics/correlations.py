import numpy as np

def create_mask(data):
    return (data != 0)

def pearson_correlation(reconstructions, inputs, mask_zeros=True, nz=False):
    """Compute the Pearson correlation coefficient between two arrays."""
   
    if mask_zeros:
        reconstructions = reconstructions[inputs != 0]
        inputs = inputs[inputs != 0]
        

    mean_x = np.mean(reconstructions)
    mean_y = np.mean(inputs)
    
    numerator = np.sum((reconstructions - mean_x) * (inputs - mean_y))
    denominator = np.sqrt(np.sum((reconstructions - mean_x)**2) * np.sum((inputs - mean_y)**2))
    
    # To prevent division by zero in the case where the arrays are constant
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def average_pearson_correlation(reconstructions, inputs, mask_zeros = False):
    """Compute the average Pearson correlation for each feature across two tensors."""
    # Check if shapes match
    #from icecream import ic
    if reconstructions.shape != inputs.shape:
        raise ValueError("The two tensors must have the same shape.")

    # if mask_zeros:
    #     A = A * create_mask(B)

    n_samples, n_features = reconstructions.shape
    
    # Compute Pearson correlation for each feature
    sample_correlations = [pearson_correlation(reconstructions[i, :], inputs[i, :], mask_zeros) for i in range(n_samples)]
    gene_correlations = [pearson_correlation(reconstructions[:, i], inputs[:, i], mask_zeros) for i in range(n_features)]
    
    return np.mean(sample_correlations), np.mean(gene_correlations)