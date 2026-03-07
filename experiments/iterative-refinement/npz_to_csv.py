import numpy as np
import pandas as pd
from pathlib import Path

results_dir = Path("results")

for npz_path in results_dir.rglob("*.npz"):
    data = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame({k: data[k] for k in data.keys()})
    csv_path = npz_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"{npz_path} -> {csv_path}")
