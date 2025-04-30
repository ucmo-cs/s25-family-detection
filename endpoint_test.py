import requests

url = "http://127.0.0.1:5000/check-kinship"
files = {
    "image1": open("/home/kirk/Desktop/Projects/s25-family-detection/TestPhotos/jackson-solo-img.jpg", "rb"),
    "image2": open("/home/kirk/Desktop/Projects/s25-family-detection/TestPhotos/mom-solo.jpg", "rb")
}

response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Raw text:", response.text)