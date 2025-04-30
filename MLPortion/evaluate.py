import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import sys

from siamese_model import SiameseNetwork
from DataProcessingFiles.kinship_dataset import KinshipPairDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.stdout = open("../Logs/eval_log1.txt", "a")

#load the mode l
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load("../models/siamese_kinship_model.pt"))
model.eval()

#load the eval data
eval_dataset = KinshipPairDataset(
    csv_file="../ActualData/eval.csv",
    root_dir="../FIDs/FIDs"
)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

#this commented stuff is apart of me figuring out the most accurate model thereshold to use
'''all_preds = []
all_labels = []
all_scores = []'''
all_distances = []
all_labels = []

with torch.no_grad():
    for img1, img2, label in eval_loader:
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        out1, out2 = model(img1, img2)
        dist = torch.nn.functional.pairwise_distance(out1, out2)

        all_distances.append(dist.item())
        all_labels.append(label.item())
        #same as the previously commented out junk
        '''img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        out1, out2 = model(img1, img2)
        dist = torch.nn.functional.pairwise_distance(out1, out2)

        pred = (dist < 0.7).float().cpu().numpy()[0]
        score = (1 - dist).cpu().numpy()[0]

        all_preds.append(pred)
        all_scores.append(score)
        all_labels.append(label.item())'''

#acc = accuracy_score(all_labels, all_preds)
#roc = roc_auc_score(all_labels, all_scores)

thresholds = np.arange(0.2, 1.5, 0.05)
best_acc = 0
best_thresh = 0

print("\n📊 Threshold Sweep Results:")
for thresh in thresholds:
    preds = [(d < thresh) for d in all_distances]
    acc = accuracy_score(all_labels, preds)
    print(f"Threshold {thresh:.2f} → Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"\n✅ Best Threshold: {best_thresh:.2f}")
print(f"🎯 Accuracy at Best Threshold: {best_acc:.4f}")

roc = roc_auc_score(all_labels, [1-d for d in all_distances])
print(f"✅ Evaluation Complete")
#print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC:  {roc:.4f}")