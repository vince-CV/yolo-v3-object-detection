import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random

classes=["aryllatag_woven15"]


def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('C:/Users/xwen2/Desktop/YOLOv3/VOCdevkit/VOC2019/Annotations/%s.xml' %image_id)
    out_file = open('C:/Users/xwen2/Desktop/YOLOv3/VOCdevkit/VOC2019/labels/%s.txt' %image_id, 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


wd = "C:/Users/xwen2/Desktop/YOLOv3/"

work_sapce_dir = os.path.join(wd, "VOCdevkit/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)

work_sapce_dir = os.path.join(work_sapce_dir, "VOC2019/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)

annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
if not os.path.isdir(annotation_dir):
    os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)

image_dir = os.path.join(work_sapce_dir, "images/train2014/")
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
clear_hidden_files(image_dir)

VOC_file_dir = os.path.join(work_sapce_dir, "ImageSets/")
if not os.path.isdir(VOC_file_dir):
    os.mkdir(VOC_file_dir)

VOC_file_dir = os.path.join(VOC_file_dir, "Main/")
if not os.path.isdir(VOC_file_dir):
    os.mkdir(VOC_file_dir)

train_file = open(os.path.join(wd, "2019_train.txt"), 'w')
test_file = open(os.path.join(wd, "2019_test.txt"), 'w')
train_file.close()
test_file.close()

VOC_train_file = open(os.path.join(work_sapce_dir, "ImageSets/Main/train.txt"), 'w')
VOC_test_file = open(os.path.join(work_sapce_dir, "ImageSets/Main/test.txt"), 'w')
VOC_train_file.close()
VOC_test_file.close()

if not os.path.exists('C:/Users/xwen2/Desktop/YOLOv3/VOCdevkit/VOC2019/labels'):
    os.makedirs('C:/Users/xwen2/Desktop/YOLOv3/VOCdevkit/VOC2019/labels')

train_file = open(os.path.join(wd, "2019_train.txt"), 'a')
test_file = open(os.path.join(wd, "2019_test.txt"), 'a')
VOC_train_file = open(os.path.join(work_sapce_dir, "ImageSets/Main/train.txt"), 'a')
VOC_test_file = open(os.path.join(work_sapce_dir, "ImageSets/Main/test.txt"), 'a')

list = os.listdir(image_dir) # list image files

probo = random.randint(1, 100)
print("Probobility: %d" % probo)

for i in range(0,len(list)):
    path = os.path.join(image_dir,list[i])
    print(image_dir)
    print(path)
    if os.path.isfile(path):
        image_path = image_dir + list[i]
        voc_path = list[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)

    probo = random.randint(1, 100)
    print("Probobility: %d" % probo)
    if(probo < 85):
        #if os.path.exists(annotation_path):
        train_file.write(image_path + '\n')
        VOC_train_file.write(voc_nameWithoutExtention + '\n')
            #convert_annotation(nameWithoutExtention)
    else:
        #if os.path.exists(annotation_path):
        test_file.write(image_path + '\n')
        VOC_test_file.write(voc_nameWithoutExtention + '\n')
            #convert_annotation(nameWithoutExtention)

train_file.close()
test_file.close()
VOC_train_file.close()
VOC_test_file.close()
