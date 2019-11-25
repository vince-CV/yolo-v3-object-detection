# YOLO-v3 Object Detection

After install the darknet with cuda, opencv. Run the demo using webcam:
```
darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
```
Run the demo using picture:
```
darknet detector cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

### Install annotation tool
Install `LabelImg`, dependency `Python 3.x` + `Qt 5`.

### Add custom classes
Pre-define the classes in `labelImg/data/predefined_classes.txt`.

### Annotate custom data set
Generate file form:
1. PASCAL VOC: .xml (Top left, bottom right points);
2. YOLO: .txt (class_id, x, y, w, h) ratio form;<br>
Convert data files using: `datacheck/gen_anchors.py`.

### Reform data set


### Configrations for data & model
```
darknet.exe detector calc_anchors cfg/voc-at.data -num_of_clusters 9 -width 256 -height 256
```

### Train
```
darknet detector train cfg/voc-at.data cfg/yolov3-voc-at.cfg darknet53.conv.74
```

### Test
