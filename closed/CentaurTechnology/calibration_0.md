NCore Quantization
--------------------

NCore supports quantization-aware and post-training quantization schemes as implemented by TF-Lite. NCore software can derive the quantization parameters from the quantized TF-Lite graphs.

__**Quantization-aware Training (MobileNetV1, MobileNetV1-SSD)**__

For TensorFlow's quantization-aware training, NCore supports asymmetric per-tensor quantization for both the weight and data tensors. TF-Lite's TOCO tool uses the dynamic ranges specified by the FakeQuant nodes present in the training graph to produce a quantized TF-Lite graph that can be compiled for NCore.

__**Post-training Quantization (ResNet50)**__

For TF-Lite's calibration-based quantization flow, NCore supports per-channel symmetric weight tensors and per-tensor asymmetric data tensors. The reference MLPerf TensorFlow ResNet50 graph is first converted to a fp32 TF-Lite graph. Then TF-Lite performs calibration on the fp32 graph using the calibration dataset and determines the dynamic range of the data tensors based on the min/max values observed. The weights are quantized by on their per-channel dynamic range.

