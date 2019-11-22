import cv2
import os
from PIL import Image

path = "C:/Users/xwen2/Desktop/YOLOv3/VOCdevkit/VOC2019/images/train2014/"
save_path ="C:/Users/xwen2/Desktop/YOLOv3/VOCdevkit/VOC2019/"
imagelist = os.listdir(path)

for image_name in imagelist:

    print(image_name)
    image = Image.open(path+image_name)

    if hasattr(image, '_getexif'):
        dict_exif = image._getexif()
        print(dict_exif[274])

        if dict_exif[274] == 3:
            new_img = image.rotate(-90, 0, 1)
        elif dict_exif[274] == 6:
            new_img = image.rotate(-90, 0, 1)
        else:
            new_img = image
        
    else:
        new_img = image

    new_img.save(save_path+image_name)
