# Hailo Post-Training Quantization


## Method

We follow the commonly used quantization scheme described in Jacob et al. (2017) (colloquially known as GEMMLOWP). Briefly, integer quantization consists of approximating real values with integers according to Xq = X / scale where scale = (max(x) − min(x))/2^N and N is the number of bits used in the approximation. Each layer’s weights and activations are given a different scale according to their extremum values. Activations are encoded using 8-bit unsigned integers and weights were encoded using 8-bit integers. Biases were encoded using either 16-bits integers or less. 

## Post-Training Manipulation

We use passive quantization, meaning that no retrain was used and there is no need for labeled data. For all experiments we extract the activation extremum values based on a random calibration set of varying size (usually 64 images). 

## Further Improvements

To improve quantization performance we employ, when needed, equalization (Meller et al. 2019) or IBC (Finkelstein and Almog, 2019) or both. 

## References
[1] Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, Jacob et al. (2017), https://arxiv.org/pdf/1712.05877.pdf
[2] GEMMLOWP reference
[3] Same, Same but Different, Meller at al., 2019, https://arxiv.org/pdf/1902.01917.pdf
[4] Fighting Quantization Bias with Bias, Finkelstein & Almog et al., 2019, https://arxiv.org/pdf/1906.03193.pdf
