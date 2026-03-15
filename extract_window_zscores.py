import numpy as np
import csv
from pathlib import Path

scores_dir = Path("scores/S1")
out_dir = Path("csv_results")
out_dir.mkdir(exist_ok=True)

stories = {
    "imagined_speech": "alpha_repeat-1",
    "perceived_movie": "laluna",
    "perceived_multispeaker": "attend-F",
    "perceived_speech": "wheretheressmoke",
}

variables = {
    "gpt_layer": [6, 7, 8, 9, 10],
    "gpt_words": [3, 4, 5, 8, 10],
}

for var_name, values in variables.items():
    for task, story in stories.items():
        # Collect rows
        rows = []
        max_windows = 0
        for val in values:
            npz_path = scores_dir / task / f"{var_name}_{val}" / f"{story}.npz"
            data = np.load(npz_path, allow_pickle=True)
            wz = data["window_zscores"].item()
            bert_key = (story, "BERT")
            bert_zscores = wz[bert_key].tolist()
            max_windows = max(max_windows, len(bert_zscores))
            rows.append((val, bert_zscores))

        # Write CSV
        header = [var_name] + [f"BERT_window_zscore_{i}" for i in range(max_windows)]
        fname = out_dir / f"{var_name}_{task}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for val, zscores in rows:
                writer.writerow([val] + zscores)
        print(f"Wrote {fname} ({len(rows)} rows, {max_windows} window columns)")
