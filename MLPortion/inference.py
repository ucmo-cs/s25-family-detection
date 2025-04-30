import torch
from PIL import Image
from torchvision import transforms

from MLPortion.siamese_model import SiameseNetwork

#load the model we trained
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load("models/siamese_kinship_model.pt"))
model.eval()

#try to preprocess image and make it cleaner to analyze and scan
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

#image paths chosen for testing inference.py on its own
img1_path = "TestPhotosCleaned/jackson-solo-img1_face_0.jpg"
img2_path = "TestPhotosCleaned/mom-solo_face_0.jpg"

#img1_path = "./TestPhotosCleaned/jackson-solo-img1.jpg"
#img2_path = "./TestPhotosCleaned/jackson-solo-img.jpg"

#function to run when we're trying to hit an endpoint
def check_kinship(file1, file2, thresehold=0.2):
    img1 = preprocess(file1)
    img2 = preprocess(file2)

    output1, output2 = model(img1, img2)
    dist_tensor = torch.nn.functional.pairwise_distance(output1, output2)
    dist = float(dist_tensor.item())

    return {
        "distance": round(float(dist), 4),
        "related": dist < thresehold
    }

#run inference on its own
img1 = preprocess(img1_path)
img2 = preprocess(img2_path)
with torch.no_grad():
    out1, out2 = model(img1, img2)
    dist = torch.nn.functional.pairwise_distance(out1, out2)

#determine best threshold for comparison
print(f"\nDistance between images: {dist.item():.4f}")
if dist.item() < 0.2:
    print("👨‍👩‍👧 The model says: They might be KIN!")
else:
    print("🚫 The model says: Probably NOT kin.")