import os
from PIL import Image
import numpy as np

face_dir = "../source_data/adv_data112"
faces = os.listdir(face_dir)

num = 0
score = 0
for face in faces:
    face_path = os.path.join(face_dir, face)
    origin_img = Image.open(face_path).convert('RGB')
    origin_img = np.array(origin_img, dtype=float)
    num+=1
    print(origin_img.shape)
    if origin_img.shape[0]>150:
        score +=1

print(num)
print(score)