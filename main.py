from face_recognition import models
import cv2
import numpy as np
from mxnet import ndarray as nd
import os

models.show_avai_models()
config_path = 'configs/server_api.yaml'

retina_model = models.build_model('retina-r50',config_path)
arcface = models.build_model('arc-face',config_path)

print (retina_model)
print (arcface)

img =  cv2.imread('imgs/PhanVanTinh.jpg')

# pre-processing image
H, W = img.shape[:2]
if (W > 1000) or (H > 1000):
    scale = min(1000 / W, 1000 / H)
    resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
else:
    resized_img = img

resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
net_data = resized_rgb.transpose(2, 0, 1)
net_data = np.expand_dims(net_data, axis=0)
net_data = nd.array(net_data)


# get the embedding from person
faces, landmarks = retina_model.detect_fast(
                        net_data,
                        resized_img.shape,
                        0.8,
                        [1],
                        do_flip= False)

if faces is not None:
    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
        print('score', faces[i][4])
        box = faces[i].astype(np.int)
        color = (0,0,255)
        cv2.rectangle(resized_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        if landmarks is not None:
            landmark5 = landmarks[i].astype(np.int)
            print(landmark5.shape)
            for l in range(landmark5.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(resized_img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
if not os.path.exists("logs/outputs"):
    os.makedirs("logs/outputs")

cv2.imwrite("logs/outputs/test.png",resized_img)
