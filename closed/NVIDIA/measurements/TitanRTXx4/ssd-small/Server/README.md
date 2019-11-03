Run the following commands to generate TensorRT engines and run the harness:

```
make generate_engines RUN_ARGS="--benchmarks=ssd-small --scenarios=Server"
make run_harness RUN_ARGS="--benchmarks=ssd-small --scenarios=Server --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=ssd-small --scenarios=Server --test_mode=PerformanceOnly"
```

For more details, please refer to the README.md located in `closed/NVIDIA`.
