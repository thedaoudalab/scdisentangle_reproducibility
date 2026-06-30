The UMAP embeddings in **Fig. 4a** are sensitive to the exact builds of
`umap-learn`, `numba`, `pynndescent`, or `llvmlite`. Different builds can shift the embedding even with fixed seed. To reproduce exactly, add the following code to the first cell of `disentangle.ipynb` 

```python
import sys, subprocess, tempfile
d = tempfile.mkdtemp()
subprocess.run([sys.executable, "-m", "pip", "install", "--target", d, "--no-deps", "--quiet",
                "umap-learn==0.5.9.post2", "numba==0.60.0", "pynndescent==0.5.13",
                "llvmlite==0.43.0"], check=True)
sys.path.insert(0, d)
```
