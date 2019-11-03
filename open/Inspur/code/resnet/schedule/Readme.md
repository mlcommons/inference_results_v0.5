The standard 8-bit quantization steps of TensorRT were used for ResNet50/ MobileNet/ SSD-Mobilenet / SSD-Resnet34. Following are the details about the quantization. 
1.	We used¡°symmetric linear quantization¡±in TensorRT and used the calibration method to determine activation ranges for each tensor in the networks.
2.	500  validation images of imagenet dataset were used for calibrating ResNet50 and MobileNet, and 500 validation images of COCO2017 dataset ware used for calibrating SSD-MobileNet and SSD-Resnet34.  
