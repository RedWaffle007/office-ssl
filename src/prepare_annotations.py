from pathlib import Path
import pandas as pd
import json

def main():
    BASE = Path.home() / "projects/office-ssl/data/labeled"
    IMG_DIR = BASE / "images"
    LABEL_DIR = BASE / "labels"

    # === 1. Load class list ===
    classes_path = BASE / "classes.txt"
    classes = [c.strip() for c in open(classes_path, "r") if c.strip()]
    label_map = {name: i for i, name in enumerate(classes)}

    print(f"Loaded {len(classes)} classes:")
    print(classes)

    # Save mapping for reproducibility
    with open(BASE / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # === 2. Parse label files ===
    records = []
    for lbl_file in sorted(LABEL_DIR.iterdir()):
        if lbl_file.suffix not in (".txt", ".json"):
            continue

        # Base image name (assuming .jpg)
        fname = lbl_file.stem + ".jpg"
        img_path = IMG_DIR / fname
        if not img_path.exists():
            png_path = IMG_DIR / (lbl_file.stem + ".png")
            if png_path.exists():
                fname = lbl_file.stem + ".png"
            else:
                print(f" Missing image for {lbl_file.name}, skipping.")
                continue

        labels = set()

        # --- A) YOLO .txt ---
        if lbl_file.suffix == ".txt":
            lines = [l.strip() for l in open(lbl_file) if l.strip()]
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(classes):
                        labels.add(classes[class_id])
                    else:
                        print(f" Invalid class id {class_id} in {lbl_file.name}")
                except ValueError:
                    pass

        # --- C) JSON (name field) ---
        elif lbl_file.suffix == ".json":
            try:
                data = json.load(open(lbl_file))
                if isinstance(data, dict):
                    if "name" in data:
                        labels.add(data["name"].strip())
                    elif "objects" in data:
                        for obj in data["objects"]:
                            if "name" in obj:
                                labels.add(obj["name"].strip())
            except Exception as e:
                print(f" Failed to parse {lbl_file.name}: {e}")

        if not labels:
            print(f" No labels found in {lbl_file.name}")
            continue

        records.append({"filename": fname, "labels": ";".join(sorted(labels))})

    # === 3. Save final annotations ===
    df = pd.DataFrame(records)
    out_path = BASE / "annotations.csv"
    df.to_csv(out_path, index=False)
    print(f"\n Saved {len(df)} entries to {out_path}")

# Allow running as a script
if __name__ == "__main__":
    main()
