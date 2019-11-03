# Data Maps

Below shows the steps to get these data maps:

- `imagenet/val_map.txt`: Follow the "*ImageNet 2012 validation dataset*" section in the [README.md](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/README.md) in the reference repository.
- `imagenet/cal_map.txt`: Downloaded from [cal_image_list_option_1.txt](https://github.com/mlperf/inference/blob/master/calibration/ImageNet/cal_image_list_option_1.txt) in the reference repository.
- `coco/val_map.txt`: Run the following Python script:
```
import json
with open("build/data/coco/annotations/instances_val2017.json") as f:
    annotations = json.load(f)
with open("data_maps/coco/val_map.txt", "w") as f:
    print("\n".join([i["file_name"] for i in annotations["images"]]), file=f)
```
- `coco/cal_map.txt`: Downloaded from [coco_cal_images_list.txt](https://github.com/mlperf/inference/blob/master/calibration/COCO/coco_cal_images_list.txt) in the reference repository.
