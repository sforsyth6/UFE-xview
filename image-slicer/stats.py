import numpy as np
import cv2
import os

path = '/data/big_train/'
direcs = os.listdir(path)
size = 224
num = 0

mean, std = [0 for i in range(2)]
i = 0
for pic in direcs:
    img = cv2.imread(path + pic, -1)
    mean += np.mean(img, axis=tuple(range(img.ndim-1)))
    num += 1
    print (img.dtype)
    break


print ((mean/num)/255)
