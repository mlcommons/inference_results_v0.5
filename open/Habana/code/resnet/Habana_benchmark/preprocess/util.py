import numpy as np

DTYPE_TO_BITS = {"invalid": None, "int8": 8, "float16": 32, "float32": 32, "int16": 16, "int32": 32}
DTYPE_TO_NP_DTYPE = {"invalid": None, "int8": np.int8, "float16": np.float16, "float32": np.float32, "int16": np.int16, "int32": np.int32}

def round_away_from_zero(t):
    abs_t = np.abs(t)
    rnd = np.sign(t) * np.floor(abs_t + 0.5)
    return rnd

def quantize(data, zp, scale, dtype):
    assert (isinstance(data, np.ndarray))
    if dtype == "float32":
        return data
    if dtype != "int8":
        assert (zp == 0)
    quantized_data = data / scale + zp
    quantized_data_rounded = round_away_from_zero(quantized_data)
    return quantized_data_rounded.astype(np.dtype(dtype))

def dequantize(data, zp, scale, dtype):
    if dtype == "float32":
        return data

    data = data.astype(np.dtype(dtype))
    return (data - zp) * scale

