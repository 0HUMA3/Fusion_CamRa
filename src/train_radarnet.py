import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.radarfusion_dataset import RadarFusionDataset
from src.models.radarnet import RadarNet

# --------------------------- CONFIG ---------------------------
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4
OUT_CLASSES = 5  # change based on your dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------- DATASET ---------------------------
train_ds = RadarFusionDataset("data/data/preprocessed_data/train_meta_fixedpaths_with_radar.json")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --------------------------- MODEL ---------------------------
model = RadarNet(out_classes=OUT_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# --------------------------- TRAIN LOOP ---------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for batch in pbar:
        image = batch["image"].to(DEVICE)
        radar = batch["radar"].to(DEVICE)
        dacc = batch["dacc"].to(DEVICE)

        # NOTE: Replace this with your actual labels later
        # For now, create dummy labels for testing
        labels = torch.randint(0, OUT_CLASSES, (image.size(0),), device=DEVICE)

        optimizer.zero_grad()
        outputs = model(image, radar, dacc)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{running_loss/len(pbar):.4f}"})

    scheduler.step()

    print(f"✅ Epoch {epoch+1}/{EPOCHS} | Avg Loss: {running_loss/len(train_loader):.4f}")

    # Save checkpoint every epoch
    torch.save(model.state_dict(), f"checkpoints/radarnet_epoch{epoch+1}.pth")
    print(f"💾 Model saved to checkpoints/radarnet_epoch{epoch+1}.pth")
