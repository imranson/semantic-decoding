import numpy as np
from pathlib import Path

results_dir = Path("results")

for npz_path in results_dir.rglob("*.npz"):
    words = np.load(npz_path, allow_pickle=True)["words"]
    txt_path = npz_path.with_suffix(".txt")
    txt_path.write_text(" ".join(words))
    print(f"{npz_path} -> {txt_path}")
