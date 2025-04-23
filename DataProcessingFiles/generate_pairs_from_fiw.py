import os
import random
import csv
from glob import glob

def generate_pairs(data_dir, output_csv, num_neg_pairs=10000):
    positive_pairs = []
    negative_pairs = []

    families = [f for f in os.listdir(data_dir) if f.startswith("F")]
    print(f"Found {len(families)} families.")

    #create positive pairs, every family member for family id
    for family in families:
        family_path = os.path.join(data_dir, family)
        mids = [mid for mid in os.listdir(family_path) if mid.startswith("MID")]
        mid_to_imgs = {}

        for mid in mids:
            img_files = glob(os.path.join(family_path, mid, "*.jpg"))
            if img_files:
                mid_to_imgs[mid] = img_files

        mid_keys = list(mid_to_imgs.keys())

        #pair every mid for the family
        for i in range(len(mid_keys)):
            for j in range(i + 1, len(mid_keys)):
                img_list_1 = mid_to_imgs[mid_keys[i]]
                img_list_2 = mid_to_imgs[mid_keys[j]]
                for img1 in img_list_1:
                    for img2 in img_list_2:
                        img1_rel = os.path.relpath(img1, start=os.path.abspath(data_dir))
                        img2_rel = os.path.relpath(img2, start=os.path.abspath(data_dir))
                        img1_rel = img1_rel.replace("FIDs/FIDs/", "").lstrip("/")
                        img2_rel = img2_rel.replace("FIDs/FIDs/", "").lstrip("/")

                        positive_pairs.append((img1_rel, img2_rel, 1))

    print(f"Generated {len(positive_pairs)} positive pairs.")

    #create neg pairs, random people from random families
    all_imgs = []
    for family in families:
        for mid in os.listdir(os.path.join(data_dir, family)):
            all_imgs.extend(glob(os.path.join(data_dir, family, mid, "*.jpg")))

    seen_pairs = set()
    attempts = 0
    max_attempts = num_neg_pairs * 10

    while len(negative_pairs) < num_neg_pairs and attempts < max_attempts:
        img1_abs, img2_abs = random.sample(all_imgs, 2)

        #conver to rel path
        img1 = os.path.relpath(img1_abs, start=data_dir)
        img1 = img1.lstrip("/").replace("FIDs/FIDs/", "")
        img2 = os.path.relpath(img2_abs, start=data_dir)
        img2 = img2.lstrip("/").replace("FIDs/FIDs/", "")

        fam1 = os.path.normpath(img1).split(os.sep)[0]
        fam2 = os.path.normpath(img2).split(os.sep)[0]

        if fam1 != fam2:
            pair_key = tuple(sorted((img1, img2)))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                if "unrelated_and_nonfaces" not in img1 and "unrelated_and_nonfaces" not in img2:
                    negative_pairs.append((img1, img2, 0))

        attempts += 1

    print(f"Generated {len(negative_pairs)} negative pairs.")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img1", "img2", "label"])
        for row in positive_pairs + negative_pairs:
            writer.writerow(row)

    print(f"Saved pairs to {output_csv}")

generate_pairs("../FIDs/FIDs", output_csv="pairs.csv")