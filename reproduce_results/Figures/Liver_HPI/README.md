The UMAP embeddings are sensitive to the exact build of
`umap-learn`, `numba`, `pynndescent`, and `llvmlite`. Different builds can shift the embedding even with fixed seed. To reproduce exactly, add the code below to the first cell of `UMAP.ipynb` notebook.

```python
import sys, subprocess, tempfile
d = tempfile.mkdtemp()
subprocess.run([sys.executable, "-m", "pip", "install", "--target", d, "--no-deps", "--quiet",
                "umap-learn==0.5.9.post2", "numba==0.60.0", "pynndescent==0.5.13",
                "llvmlite==0.43.0", "graphtools==1.5.3"], check=True)
sys.path.insert(0, d)
```

