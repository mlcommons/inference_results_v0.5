# Compile a pretrained `mobilenet` TFLite model for Renegade

- compile tflite models using dgc (internal compiler)
``` shell
$ NPU_COMPILER_CONFIG_PATH=remove_lower.yml ../dgc imagenet_224x224_mobilenet_v1_uint8_quantization-aware-trained_dm_1.0.tflite -o mobilenet_v1.tflite
```

- run `mobilenet_v1.tflite` in SUT
