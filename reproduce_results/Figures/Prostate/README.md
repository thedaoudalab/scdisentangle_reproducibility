The UMAP embeddings in **Fig. 4a**, and PHATE embeddings **Fig. 4g** are sensitive to the exact build of
`umap-learn`, `numba`, `pynndescent`, `llvmlite`, and `graphtools`. Different builds can shift the embedding even with fixed seed. To reproduce exactly, add the code below to the first cell of `disentangle.ipynb`, and of `OOD_transition/streamline_plot.ipynb` notebooks.

```python
import sys, subprocess, tempfile
d = tempfile.mkdtemp()
subprocess.run([sys.executable, "-m", "pip", "install", "--target", d, "--no-deps", "--quiet",
                "umap-learn==0.5.9.post2", "numba==0.60.0", "pynndescent==0.5.13",
                "llvmlite==0.43.0", "graphtools==1.5.3"], check=True)
sys.path.insert(0, d)
```

The transition analysis additionally requires `scvelo==0.3.3`
