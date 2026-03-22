import os, shutil, random
from pathlib import Path

# --- config ---
DATASET_DIR = r"C:\Users\sonal\Projectpr\notebooks\Preprocessing\Preprocessed_datasets\PlantVillage"
OUTPUT_DIR  = r"C:\Users\sonal\Projectpr\src\client"
NUM_CLIENTS = 5
SPLIT_RATIO = 0.8
SEED = 42
# --------------

random.seed(SEED)

# 🔥 Traverse full structure
for category in os.listdir(DATASET_DIR):
    category_path = Path(DATASET_DIR) / category

    if not category_path.is_dir():
        continue

    for disease in os.listdir(category_path):
        disease_path = category_path / disease

        for plant in os.listdir(disease_path):
            plant_path = disease_path / plant

            if not plant_path.is_dir():
                continue

            # 🔥 use .npz files
            images = list(plant_path.glob("*.npz"))

            if len(images) == 0:
                continue

            random.shuffle(images)

            # split into clients
            chunks = [images[i::NUM_CLIENTS] for i in range(NUM_CLIENTS)]

            class_name = f"{category}_{disease}_{plant}"

            for client_id, chunk in enumerate(chunks):
                split = int(len(chunk) * SPLIT_RATIO)

                splits = {
                    "train": chunk[:split],
                    "val": chunk[split:]
                }

                for split_name, files in splits.items():
                    dest = Path(OUTPUT_DIR) / f"client_{client_id}" / split_name / class_name
                    dest.mkdir(parents=True, exist_ok=True)

                    for f in files:
                        shutil.copy(f, dest / f.name)

print("✅ Done. Client folders created at:", OUTPUT_DIR)