import cv2 as cv

image_path = "C:/Users/xwen2/Desktop/109.JPG"


image = cv.imread(image_path)
image_size = image.shape

print(image_size)