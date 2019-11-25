import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2 as cv
import os
from PIL import Image

image_path = "C:/Users/xwen2/Desktop/YOLOv3/VOCdevkit/VOC2019/images/train2014/"
anota_path = "C:/Users/xwen2/Desktop/YOLOv3/VOCdevkit/VOC2019/labels/"

#image_path = "C:/Users/xwen2/Desktop/New folder/"
#anota_path = "C:/Users/xwen2/Desktop/New folder (2)/"
aug = 20

imagelist = os.listdir(image_path)

for image_name in imagelist:
    (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_name))
    image = cv.imread(image_path+image_name)
    image_size =image.shape
    height = image_size[0]
    width = image_size[1]

    images = np.zeros((1, height, width, 3), dtype=np.uint8) 
    images[:, :, :, :] = image

    yolo = np.loadtxt(anota_path+nameWithoutExtention+".txt")

    center_x = width*yolo[1]
    center_y = height*yolo[2]
    box_width = width*yolo[3]
    box_height = height*yolo[4]

    x = center_x-0.5*box_width
    y = center_y-0.5*box_height
    X = center_x+0.5*box_width
    Y = center_y+0.5*box_height

    bbs = [
        [ia.BoundingBox(x1=x, y1=y, x2=X, y2=Y, label= yolo[0])]
    ]

    #print(bbs[0])

    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.Affine(scale=(0.75, 0.9),mode='constant', cval=(125, 126)), 
            iaa.Affine(scale=(1.1, 1.25),mode='constant', cval=(125, 126)), 
        ]),
        iaa.OneOf([
            iaa.Affine(rotate=(-25, -5),mode='constant', cval=(125, 126)), 
            iaa.Affine(rotate=(5, 25),mode='constant', cval=(125, 126))
        ]),
        iaa.OneOf([
            iaa.Affine(translate_percent={"x": (-0.0, 0.0), "y": (-0.0, 0.0)}, mode='constant', cval=(125, 126)),
            iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, mode='constant', cval=(125, 126)),
        ]),
        iaa.GaussianBlur(sigma=(2.0, 5.0)),
        iaa.OneOf([
            iaa.Add((20, 50)),
            iaa.Add((-50, -20))
        ])
    ])

    for i in range(aug):
        images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
        image_save=images

        
        #yolo_aug = np.arange(5)
        aug_yolo_label = bbs_aug[0][0].label
        aug_center_x = 0.5*(bbs_aug[0][0].x1+bbs_aug[0][0].x2)/width
        aug_center_y = 0.5*(bbs_aug[0][0].y1+bbs_aug[0][0].y2)/height

        aug_box_x = (bbs_aug[0][0].x2 - bbs_aug[0][0].x1)/width
        aug_box_y = (bbs_aug[0][0].y2 - bbs_aug[0][0].y1)/height

        list_aug = [(aug_yolo_label, aug_center_x, aug_center_y, aug_box_x, aug_box_y)]
        yolo_aug = np.asarray(list_aug) 

        print("augmented:", yolo_aug)
        
        
        np.savetxt(anota_path+nameWithoutExtention+"_aug"+str(i)+".txt", yolo_aug, fmt='%6e')





        cv.imwrite(image_path+nameWithoutExtention+"_aug"+str(i)+".JPG", images_aug[0])
        #print(bbs_aug[0])
        #cv.rectangle(images_aug[0], (int(bbs_aug[0][0].x1), int(bbs_aug[0][0].y1)), (int(bbs_aug[0][0].x2), int(bbs_aug[0][0].y2)), (55,255,155), 10 )
        #images_show = cv.resize(images_aug[0],(600, 500) )
        #cv.imshow("0", images_show)
        #cv.waitKey()
