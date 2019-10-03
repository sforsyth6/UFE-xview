import os 
import cv2
import numpy as np

#train = '/raid/etegent/xview/images_rgb/train/'
train = '/data/val/'

big_train = '/data/big_val/'

slice_sz = 224

for image in os.listdir(train):
        name = image.split('.')[0]
        print (name)
        try:
            img = cv2.imread(train + image, -1)
#            cv2.imwrite(big_train + 'out.tif', img)
            shape = img.shape
            num_x_slices = int(shape[0]/slice_sz)
            num_y_slices = int(shape[1]/slice_sz)

            for xslice in range(num_x_slices):
                for yslice in range(num_y_slices):
                    sliced = img[xslice*slice_sz:(xslice+1)*slice_sz, yslice*slice_sz:(yslice+1)*slice_sz, :]
                    cv2.imwrite(big_train + '{0}-{1}-{2}.tif'.format(name,xslice,yslice), sliced)
                    #if there is extra image after discretizing, create a 224x224 image from that edge of the image (y-axis)
                    if yslice == (num_y_slices - 1) and (num_y_slices % slice_sz) != 0:
                        sliced = img[xslice*slice_sz:(xslice+1)*slice_sz, (shape[1] - 224):, :]
                        cv2.imwrite(big_train + '{0}-{1}-{2}-yedge.tif'.format(name,xslice,(yslice+1)), sliced)

                #if there is extra image after discretizing, create a 224x224 image from that edge of the image (x-axis)
                if xslice == (num_x_slices - 1) and (num_x_slices % slice_sz) != 0:
                    for yslice in range(num_y_slices):
                        sliced = img[(shape[0] - 224): , yslice*slice_sz:(yslice+1)*slice_sz, :]
                        cv2.imwrite(big_train + '{0}-{1}-{2}-xedge.tif'.format(name,(xslice+1),yslice), sliced)
                        #if there is extra image after discretizing, create a 224x224 image from that edge of the image(y-axis)
                        if yslice == (num_y_slices - 1) and (num_y_slices % slice_sz) != 0:
                            sliced = img[(shape[0] - 224):, (shape[1] - 224):, :]
                            cv2.imwrite(big_train + '{0}-{1}-{2}-xyedge.tif'.format(name,(xslice+1),(yslice+1)), sliced)

        except:
            print ('FUCK: ' + name)
