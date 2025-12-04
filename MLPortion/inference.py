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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

#image paths chosen for testing inference.py on its own
#img1_path = "TestPhotosCleaned/fiw_style/jackson-solo-img1_face_0.jpg"
#img2_path = "TestPhotosCleaned/fiw_style/mom-solo_face_0.jpg"
#img2_path = "TestPhotosCleaned/fiw_style/sonichu_face_0.jpg"
#img2_path = "TestPhotosCleaned/fiw_style/schizoid_face_0.jpg"
#img1_path = "FIDs/FIDs/F0011/MID1/P00114_face3.jpg"
#img2_path = "FIDs/FIDs/F0011/MID3/P00114_face1.jpg"
#img2_path = "FIDs/FIDs/F0011/MID4/P00119_face1.jpg"
img2_path = "FIDs/FIDs/F0011/MID4/P00121_face1.jpg"
#img2_path = "FIDs/FIDs/F0011/MID1/P00117_face1.jpg"
#img2_path = "FIDs/FIDs/F0020/MID1/P00200_face2.jpg"
img1_path = "FIDs/FIDs/F0012/MID2/P00123_face4.jpg"
#img1_path = "TestPhotosCleaned/fiw_style/adam-solo_face_0.jpg"
#img1_path = "TestPhotosCleaned/fiw_style/emma-solo-1_face_0.jpg"
#img2_path = "TestPhotosCleaned/fiw_style/another_rando_face_0.jpg"
#img2_path = "TestPhotosCleaned/fiw_style/rando3_face_0.jpg"
#img2_path = "TestPhotosCleaned/fiw_style/random_guy_face_0.jpg"

#img1_path = "TestPhotosCleaned/fiw_style/mom-solo_face_0.jpg"
#img2_path = "TestPhotosCleaned/fiw_style/emma-solo_face_0.jpg"

#img1_path = "TestPhotosCleaned/fiw_style/adam-solo_face_0.jpg"
#img2_path = "TestPhotosCleaned/fiw_style/jackson-solo-img1_face_0.jpg"


img1 = preprocess(img1_path)
img2 = preprocess(img2_path)

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

print(f"\nDistance between images: {dist.item():.4f}")
if dist.item() < 0.7391:
    print("👨‍👩‍👧 The model says: They might be KIN!")
else:
    print("🚫 The model says: Probably NOT kin.")