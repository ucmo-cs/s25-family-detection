import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import sys

#put all output bullshit to this file, since the pycharm terminal buffer will overflow or fill over
#or idc the term whatever
sys.stdout = open("../Logs/train_log.txt", "a")

from siamese_model import SiameseNetwork, ContrastiveLoss
from DataProcessingFiles.kinship_dataset import KinshipPairDataset

# --- Config --ys
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MARGIN = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the data
train_data = KinshipPairDataset(
    csv_file="../ActualData/pairs.csv",
    root_dir="../FIDs/FIDs"
)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

#actual model
model = SiameseNetwork().to(DEVICE)
criterion = ContrastiveLoss(margin=MARGIN)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#pickup from a checkpoint if there is one present
checkpoints = sorted([f for f in os.listdir() if f.startswith("checkpoint_epoch_")])
if checkpoints:
    last_ckpt = checkpoints[-1]
    epoch_start = int(last_ckpt.split("_")[-1].split(".")[0])
    model.load_state_dict(torch.load(last_ckpt))
    print(f"🔁 Resuming from {last_ckpt} (epoch {epoch_start})")
else:
    epoch_start = 0

for epoch in range(epoch_start, NUM_EPOCHS):
    print(f"\n🚀 Starting Epoch {epoch + 1}/{NUM_EPOCHS}")
    model.train()
    total_loss = 0.0
    running_correct = 0
    total_preds = 0

    for step, (img1, img2, label) in enumerate(train_loader):
        img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        #tracking our accuracy
        with torch.no_grad():
            distances = torch.nn.functional.pairwise_distance(out1, out2)
            preds = (distances < 0.7).float()
            correct = (preds == label).sum().item()
            running_correct += correct
            total_preds += label.size(0)

        #print progress through epoch
        if step % 500 == 0:
            print(f"🟢 Epoch {epoch+1} | Step {step}/{len(train_loader)}")

    avg_loss = total_loss / len(train_loader)
    accuracy = running_correct / total_preds
    print(f"✅ Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    #save checkpoint
    ckpt_path = f"checkpoint_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")

    # Save metrics
    with open("../ActualData/metrics.csv", "a") as f:
        f.write(f"{epoch+1},{avg_loss:.4f},{accuracy:.4f}\n")

# Final model
torch.save(model.state_dict(), "../models/siamese_kinship_model.pt")
print("✅ Final model saved to siamese_kinship_model.pt")
