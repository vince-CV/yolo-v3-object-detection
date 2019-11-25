# YOLO-v3 Object Detection

```
darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
```


### Install annotation tool
Install `LabelImg`, dependency `Python 3.x` + `Qt 5`

### Add custom classes

### Annotate custom data set
Generate file form:
1. PASCAL VOC: .xml (Top left, bottom right points)
2. YOLO: .txt (class_id, x, y, w, h) ratio form

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
