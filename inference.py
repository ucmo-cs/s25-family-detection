import torch
from PIL import Image
from torchvision import transforms

from siamese_model import SiameseNetwork

# Load your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load("models/siamese_kinship_model.pt"))
model.eval()

# Preprocessing function (resize, tensor, etc.)
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

# Set your image file paths here
img1_path = "./TestPhotos/the-man-himself.png"
img2_path = "./TestPhotos/jackson-solo-img1.png"

#img1_path = "./TestPhotosCleaned/jackson-solo-img1.jpg"
#img2_path = "./TestPhotosCleaned/jackson-solo-img.jpg"

# Preprocess the images
img1 = preprocess(img1_path)
img2 = preprocess(img2_path)

# Run inference
with torch.no_grad():
    out1, out2 = model(img1, img2)
    dist = torch.nn.functional.pairwise_distance(out1, out2)

# Choose a threshold based on your eval results (e.g., 0.7)
print(f"\nDistance between images: {dist.item():.4f}")
if dist.item() < 0.2:
    print("👨‍👩‍👧 The model says: They might be KIN!")
else:
    print("🚫 The model says: Probably NOT kin.")