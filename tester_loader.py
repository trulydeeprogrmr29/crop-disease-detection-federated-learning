from src.data.loader import get_dataloader

loader, classes = get_dataloader("dataset/raw/PlantVillage")

print("Number of classes:", len(classes))
print("Classes:", classes)
