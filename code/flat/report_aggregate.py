import os, json, pandas as pd

def main(run_dir):
    rows = []
    for root, _, files in os.walk(run_dir):
        for fn in files:
            if fn.endswith("_metrics.json"):
                path = os.path.join(root, fn)
                with open(path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                split = "dev" if "dev" in fn else "test" if "test" in fn else "unknown"
                rows.append({"split": split, **m})
    df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, "metrics_summary.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main("./run")
