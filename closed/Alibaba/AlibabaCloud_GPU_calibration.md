# Alibaba Cloud ECS Bare Metal GPU T4 instance
Post-training quantization requires a dynamic range for each weight and activation tensor. Quantization is symmetric for both.

## Weights
Dynamic range values are generally per-channel (or per-row for matrix multiply). In a few cases, a per-tensor value is used. We find the maximum absolute value t of any element of the channel or tensor, and the dynamic range is then [-t,t]. 

## Activations

For each activation tensor, we use a distinct dynamic range that applies across the entire tensor. We invoke the model on a set of representative inputs in FP32 precision, and create a per-tensor histogram of absolute values. The histogram initially uses 1024 equal-range bins whose range is set by the initial image batch, but dynamically resizes by doubling the number of bins as necessary to accommodate the range of subsequent batches. Call this histogram, which has a power-of-2 bins, where all data elements are guaranteed to fall into one of the bins, the “starting histogram”. We then apply one of two methods, as chosen by the application.

* Minmax: as with weights, we compute the maximum absolute value of the tensor.
* Entropy: for each bin B in the starting histogram, we compute a divergence value as follows:  
  * Create a truncated histogram where each bin has the same range and count as the original, except that all elements in bins beyond B are considered to be in B, and all bins beyond B are removed. 
  * Create a coarse histogram by discretizing the truncated histogram into 127 bins of equal range between 0 and the midpoint of B, placing all elements in the final bin of the truncated histogram into the final bin of the coarse histogram. 
  * Compute the KL-divergence between the distributions represented by the coarse histogram and the truncated histogram.
  The dynamic range chosen is the center of the bin which minimizes divergence.

## Additional Details
A number of minor modifications are applied to this basic algorithm, including discarding the first bin in the histogram (which typically contains a huge number of noise activations) immediately after it has been built how empty bins are treated when computing divergence. For some operations which are not expected to change dynamic range (e.g. max-pooling, concatenation) we propagate dynamic range from the output to the input(s).

## Quantization in Plugins
The submissions primarily use TensorRT, which implements the scheme described above. Where plugins are used, weight quantization is performed as described above, and activation quantization uses dynamic range values computed using TensorRT on the original network. The plugins access these values through TensorRT’s calibration cache.
