# Habana MLPerf Quantization

Post-training quantization requires a dynamic range for each weight and activation tensor. 
Quantization is per-tensor asymmetric for both and uses an 8-bit integer as its numerical precision. 

## Weights

A per-tensor asymmetric dynamic range is used. The ranges were defined according to the absolute maximum and minimum value of each of the weight tensors. 
While the weight tensors were quantized to 8-bit the bias tensors were quantized to int32. 

## Activations

A per-tensor asymmetric dynamic range is used. The ranges were extracted, by invoking the model on a set of inputs (from the mlperf calibration set) in FP32 precision and calculate the average, over a set of mini-batches, of the absolute maximum and minmum values. 
Based on the dynamic range the activation tensor is clamped and quantized.  

## Additional Details

Some elementwise operations such as softmax were kept in int16. 

## Quantization in Plugins

Habana’s closed division submissions uses our proprietary software stack named SynapseAI, which implements the scheme described above. 

## Open Division

Our Open Division submissions uses exactly the same calibration snd quantization setting. 
