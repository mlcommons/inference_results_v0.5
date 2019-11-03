# Compile a pretrained `ssd-mobilenet` TFLite model for Renegade

- optimize concat using iredit (internal tool to edit IR)
``` shell
$ ../iredit optimize_concat mscoco_300x300_ssd_mobilenet_v1_uint8_quantization-aware-trained.tflite
```

- compile tflite models using dgc (internal compiler)
``` shell
$ ../dgc mscoco_300x300_ssd_mobilenet_v1_uint8_quantization-aware-trained_concat_optimized.tflite -o ssd300_mobilenet.tflite
```

- run `ssd300_mobilenet.tflite` in SUT
