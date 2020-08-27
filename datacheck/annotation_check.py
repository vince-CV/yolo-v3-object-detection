import cv2 as cv
import os
import numpy as np
from PIL import Image

image_path = "C:/Users/xwen2/Google Drive/Mask Detector/data/"
anota_path = "C:/Users/xwen2/Google Drive/Mask Detector/data/"


imagelist = os.listdir(image_path)
                     
for image_name in imagelist:
        
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_name))

        if (extention != ".txt"): 
            print(image_path+image_name)
            image = cv.imread(image_path+image_name)

            yolo = np.loadtxt(anota_path+nameWithoutExtention+".txt")

            dim = np.array(yolo)
            print("dim: ", dim.ndim)

            number_of_box = len(yolo)
            image_size =image.shape
            height = image_size[0]
            width = image_size[1]


            if (dim.ndim == 1):
                center_x = int(width*yolo[1])
                center_y = int(height*yolo[2])
                box_width = int(width*yolo[3])
                box_height = int(height*yolo[4])
                x = int(center_x-0.5*box_width)
                y = int(center_y-0.5*box_height)
                X = int(center_x+0.5*box_width)
                Y = int(center_y+0.5*box_height)
                if (yolo[0] == 0):
                    cv.rectangle(image,(x, y),(X, Y),(55,255,155),5)
                else:
                    cv.rectangle(image,(x, y),(X, Y),(55,55,255),5)
            else:
                for item in range(number_of_box):
                    center_x = int(width*yolo[item][1])
                    center_y = int(height*yolo[item][2])
                    box_width = int(width*yolo[item][3])
                    box_height = int(height*yolo[item][4])
                    x = int(center_x-0.5*box_width)
                    y = int(center_y-0.5*box_height)
                    X = int(center_x+0.5*box_width)
                    Y = int(center_y+0.5*box_height)
                    if (yolo[item][0] == 0):
                        cv.rectangle(image,(x, y),(X, Y),(55,255,155),5)
                    else:
                        cv.rectangle(image,(x, y),(X, Y),(55,55,255),5)
            
            image = cv.resize(image, (600, 600))

            cv.imshow("annatation", image)
            cv.waitKey()

    


