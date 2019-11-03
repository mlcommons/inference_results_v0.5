# MLPerf Inference 0.5 Self-Certification Checklist

Name of Certifying Engineer(s): Brian Pharris

Email of Certifying Engineer(s): bpharris@nvidia.com

Name of System(s) Under Test: T4x8, T4x20, TitanRTXx4, Xavier

Division (check one):
- [ ] Open
- [X] Closed

Category (check one):
- [X] Available
- [ ] Preview
- [ ] Research, Development, and Internal (RDI)

Benchmark (check one):
- [X] MobileNet
- [X] SSD-MobileNet
- [X] ResNet
- [X] SSD-1200
- [X] NMT
- [ ] Other, please specify:

Please fill in the following tables adding lines as necessary:
97%-tile latency is required for NMT only. 99%-tile is required for all other models.

### Single Stream Results Table
| SUT Name | Benchmark | Query Count | Accuracy |
|----------|-----------|-------------|----------|
| T4x8     | SSD-1200  | 13771       | 20.067%  |
| T4x8     | NMT       | 2445        | 23.8     |
| TitanRTXx4| SSD-1200 | 28325       | 20.067%  |
| TitanRTXx4| NMT      | 4392        | 23.7     |
| Xavier   | MobileNet | 178166      | 70.814%  |
| Xavier   | SSD-MobileNet| 74030    | 22.900%  |
| Xavier   | ResNet    |52795        | 76.034%  |
| Xavier   | SSD-1200  |4072         | 20.067%  |

### Multi-Stream Results Table
| SUT Name | Benchmark | Query Count |  Accuracy | 97%-tile Latency | 99%-tile Latency |
|----------|-----------|-------------|-----------|------------------|------------------|
| T4x8     |MobileNet  |270336       |70.814%    |                  |45660495|
| T4x8     |SSD-MobileNet| 270336    |22.912% |                     |46681233 |
| T4x8     |ResNet     |270336       |76.034%    |                  |46439806  |
| T4x8     |SSD-1200   |270336       | 20.067%   |                  |57354372  |
| TitanRTXx4|MobileNet  |270336      |70.814%   |                   | 49592780       |
| TitanRTXx4|SSD-MobileNet|270336    | 22.911%    |                 |47245056         |
| TitanRTXx4|ResNet     | 270336     | 76.034%  |                   | 46011127        |
| TitanRTXx4|SSD-1200   | 270336     | 20.067%  |                   |62579073     |
| Xavier   |MobileNet  | 270336      | 70.774%  |                   |47670221  |
| Xavier   |SSD-MobileNet|270336     |  22.936%  |                  |45184057         |
| Xavier   |ResNet     |270336       | 76.042%  |                   | 47422247     |
| Xavier   |SSD-1200   |270336       | 20.067%   |                  | 57478177    |

### Server Results Table
| SUT Name | Benchmark | Query Count | Accuracy | 97%-tile Latency | 99%-tile Latency |
|----------|-----------|-------------|----------|------------------|------------------|
|T4x8      |MobileNet  |8104382      |70.610%   |                  |4626978      |
|T4x8      |SSD-MobileNet| 3397203   |22.912%   |                  | 7389314       |
|T4x8      |ResNet     |2492800      |76.034%   |                  |10159013         |
|T4x8      |SSD-1200   |270336       |20.067%   |                  | 66471489       |
|T4x8      |NMT        |94874        |23.7      |222674220         |                  |
|T4x20     |SSD-MobileNet|7743409    | 22.912%  |                  | 9255124     |
|T4x20     |ResNet     |6211930      |  76.034% |                  |14676235                  |
|T4x20     |SSD-1200   |270336       | 20.067%  |                  | 76475046     |
|T4x20     |NMT        |226565       | 23.8     |237471490         |                  |
|TitanRTXx4|MobileNet  |11945898     |70.610%   |                  |7355068      |
|TitanRTXx4|SSD-MobileNet|4920468    |22.911%   |                  | 7726985        |
|TitanRTXx4|ResNet     |3601836      | 76.034%  |                  |12941389        |
|TitanRTXx4|SSD-1200   |270336       | 20.067%  |                  |42369296     |
|TitanRTXx4|NMT        |154862       | 23.8     | 210258342        |                  |

### Offline Results Table
| SUT Name | Benchmark | Sample Count | Accuracy | 
|----------|-----------|--------------|----------|
| T4x8     |MobileNet  |9517332     |70.814%    |
| T4x8     |SSD-MobileNet|3940464   |22.912%   |
| T4x8     |ResNet     | 2918784    |76.034%  |
| T4x8     |SSD-1200   |70752     |20.067%    |
| T4x8     |NMT        |179454      |23.8          |
| T4x20    |SSD-MobileNet|9851160   |22.912%    |
| T4x20    |ResNet     |7296960     |76.034%   |
| T4x20    |SSD-1200   |176880      | 20.067%         |
| T4x20    |NMT        |452100      |23.8          |
| TitanRTXx4|MobileNet |14552142    |70.814%   |
| TitanRTXx4|SSD-MobileNet|5979336  |22.912%   |
| TitanRTXx4|ResNet    |4300956     | 76.034%         |
| TitanRTXx4|SSD-1200  |107448      |20.067%          |
| TitanRTXx4|NMT       | 274164     | 23.7         |
| Xavier    |MobileNet |431772      |70.778%   |
| Xavier    |SSD-MobileNet|163548   |22.926%     |
| Xavier    |ResNet    |143946      |76.010%    |
| Xavier    |SSD-1200  |24576       |20.057%   |

Scenario (check all that apply):
- [X] Single-Stream
- [X] Multi-Stream
- [X] Server
- [X] Offline

For each SUT, does the submission meet the latency target for each
combination of benchmark and scenario? (check all that apply)
- [X] Yes (Single-Stream and Offline no requirements)
- [X] Yes (MobileNet x Multi-Stream 50 ms @ 99%)
- [X] Yes (MobileNet x Server 10 ms @ 99%)
- [X] Yes (SSD-MobileNet x Multi-Stream 50 ms @ 99%)
- [X] Yes (SSD-MobileNet x Server 10 ms @ 99%)
- [X] Yes (ResNet x Multi-Stream 50 ms @ 99%)
- [X] Yes (ResNet x Server 15 ms @ 99%)
- [X] Yes (SSD-1200 x Multi-Stream 66 ms @ 99%).
- [X] Yes (SSD-1200 x Server 100 ms @ 99%)
- [ ] Yes (NMT x Multi-Stream 100 ms @ 97%)
- [X] Yes (NMT x Server 250 ms @ 97%)
- [ ] No

For each SUT, is the appropriate minimum number of queries or samples
met, depending on the Scenario x Benchmark? (check all that apply)
- [X] Yes (Single-Stream 1,024 queries)
- [X] Yes (Offline 24,576 samples)
- [X] Yes (NMT Server and Multi-Stream 90,112 queries)
- [X] Yes (Image Models Server and Multi-Stream 270,336 queries)
- [ ] No

For each SUT and scenario, is the benchmark accuracy target met?
(check all that apply)
- [X] Yes (MobileNet 71.68% x 98%)
- [X] Yes (SSD-MobileNet 0.22 mAP x 99%)
- [X] Yes (ResNet 76.46% x 99%)
- [X] Yes (SSD-1200 0.20 mAP x 99%)
- [X] Yes (NMT 23.9 BLEU x 99%)
- [ ] No


For each SUT and scenario, did the submission run on the whole
validation set in accuracy mode? (check one)
- [X] Yes
- [ ] No

How many samples are loaded into the QSL in performance mode?
- resnet: 1024
- mobilenet: 1024
- ssd-small: 256
- ssd-large: 64
- gnmt: 3903900

For each SUT and scenario, does the number of loaded samples in the
QSL in performance mode meet the minimum requirement?  (check all that
apply)
- [X] Yes (ResNet and MobileNet 1,024 samples)
- [X] Yes (SSD-MobileNet 256 samples)
- [X] Yes (SSD-1200 64 samples)
- [X] Yes (NMT 3,903,900 samples)
- [ ] No

For each SUT and scenario, is the experimental duration greater than
or equal to 60 seconds?  (check one)
- [X] Yes
- [ ] No

Does the submission use LoadGen? (check one)
- [X] Yes
- [ ] No

Is your loadgen commit from one of these allowed commit hashes?
- [X] 61220457dec221ed1984c62bd9d382698bd71bc6
- [ ] 5684c11e3987b614aae830390fa0e92f56b7e800
- [ ] 55c0ea4e772634107f3e67a6d0da61e6a2ca390d
- [ ] d31c18fbd9854a4f1c489ca1bc4cd818e48f2bc5
- [ ] 1d0e06e54a7d763cf228bdfd8b1e987976e4222f
- [ ] Other, please specify:

Do you have any additional change to Loadgen? (check one)
- [ ] Yes, please specify:
- [X] No

Does the submission run the same code in accuracy and performance
modes? (check one)
- [X] Yes
- [ ] No

Where is the LoadGen trace stored? (check one)
- [X] Host DRAM
- [ ] Other, please specify:

For the submitted result, what is the QSL random number generator seed?
- [X] 0x2b7e151628aed2a6ULL (3133965575612453542)
- [ ] Other, please specify:

For the submitted results, what is the sample index random number generator seed?
- [X] 0x093c467e37db0c7aULL (665484352860916858)
- [ ] Other, please specify:

For the submitted results, what is the schedule random number generator seed?
- [X] 0x3243f6a8885a308dULL (3622009729038561421)
- [ ] Other, please specify:

For each SUT and scenario, is the submission run the correct number of
times for the relevant scenario? (check one)
- [X] Yes (Accuracy 1x Performance 1x Single-Stream, Multi-Stream,
Offline)
- [X] Yes (Accuracy 1x Performance 5x Server)
- [ ] No

Are the weights calibrated using data outside of the calibration set?
(check one)
- [ ] Yes
- [X] No

What untimed pre-processing does the submission use? (check all that apply)
- [X] Resize
- [X] Reorder channels or transpose
- [X] Pad
- [X] A single crop
- [X] Mean subtraction and normalization
- [X] Convert to whitelisted format
- [ ] No pre-processing
- [ ] Other, please specify:

What numerics does the submission use? (check all that apply)
- [ ] INT4
- [X] INT8
- [ ] INT16
- [ ] UINT8
- [ ] UINT16
- [ ] FP11
- [X] FP16
- [ ] BF16
- [X] FP32
- [ ] Other, please specify:

Which of the following techniques does the submission use? (check all
that apply)
- [ ] Wholesale weight replacement
- [ ] Weight supplements
- [ ] Discarding non-zero weight elements
- [ ] Pruning
- [ ] Caching queries
- [ ] Caching responses
- [ ] Caching intermediate computations
- [ ] Modifying weights during the timed portion of an inference run
- [ ] Weight quantization algorithms that are similar in size to the
non-zero weights they produce
- [ ] Hard coding the total number of queries
- [ ] Techniques that boost performance for fixed length experiments but
are inapplicable to long-running services except in the offline
scenario
- [ ] Using knowledge of the LoadGen implementation to predict upcoming
lulls or spikes in the server scenario
- [ ] Treating beams in a beam search differently. For example,
employing different precision for different beams
- [ ] Changing the number of beams per beam search relative to the reference
- [ ] Incorporating explicit statistical information about the performance or accuracy sets
- [ ] Techniques that take advantage of upsampled images.
- [ ] Techniques that only improve performance when there are identical samples in a query.
- [X] None of the above

Is the submission congruent with all relevant MLPerf rules?
- [X] Yes
- [ ] No

For each SUT, does the submission accurately reflect the real-world
performance of the SUT?
- [X] Yes
- [ ] No
