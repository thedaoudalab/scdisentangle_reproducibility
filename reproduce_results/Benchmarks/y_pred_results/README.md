# How to skip train and inference

To visualize results without training + inference, your can use pre-computed metrics, and skip the `cm.compute_metrics` cells (jump directly to plotting and statistical testing cells). 
We provide the pre-computed metrics: `results.zip`. Unzip in the same folder:
   ```bash
   unzip weights.zip && rm weights.zip
   ```
