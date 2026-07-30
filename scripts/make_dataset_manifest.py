"""Create a dataset manifest for the leakage-safe rerun."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, object]] = []

    for task_dir in ["Binary", "Multiclass"]:
        for dataset_dir in sorted((root / task_dir).iterdir()):
            if not dataset_dir.is_dir():
                continue
            csv_path = dataset_dir / f"{dataset_dir.name}.csv"
            if not csv_path.exists():
                continue

            y = pd.read_csv(csv_path, usecols=["CLASS"])["CLASS"]
            counts = y.astype(str).value_counts()
            rows.append(
                {
                    "task": task_dir,
                    "dataset": dataset_dir.name,
                    "path": csv_path.relative_to(root).as_posix(),
                    "n_samples": int(len(y)),
                    "n_classes": int(counts.shape[0]),
                    "min_class_count": int(counts.min()),
                    "class_counts": ";".join(f"{label}:{count}" for label, count in counts.sort_index().items()),
                }
            )

    out_path = root / "outputs" / "dataset_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
