# Habana's open submission

Habana's open submission aim to demonstrate Goya low latency capabilities. Our open submission follows the closed submission rules with one exception - strict the latency constrains for Multi-Stream scenario.

* ResNet50: we managed to reach 20 Samples-Per-Query (SPQ) under latency constrain of 2ms and 40 SPQ under 3.3ms latency constrain. Thus, up to 25 fasters than the required latency for the closed division (50ms)
* SSD-large: we managed to reach 4 SPQ under latency constrain of 16.8ms and 8 SPQ under 30.8ms latency constrain. Up to than 4 time faster than the require latency for closed division (66ms)

We present our results using two submissions:
 * _Goya_fast_latency_,  for 2ms (resent) and 16.8ms (ssd-large) constrains.
 * _Goya_medium_latency_,  for 3.3ms (resent) and 30.8ms (ssd-large) constrains.