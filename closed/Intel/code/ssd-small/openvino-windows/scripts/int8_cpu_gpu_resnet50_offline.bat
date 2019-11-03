@echo OFF

call <\path\to\Release>\ov_mlperf.exe --scenario Offline ^
	--mode Performance ^
	--mlperf_conf_filename "<\path\to\mlperf.conf" ^
	--user_conf_filename "<\path\to\resnet50>\user.conf" ^
	--total_sample_count 50000 ^
	--data_path "<\path\to\CK-TOOLS>\dataset-imagenet-ilsvrc2012-val\" ^
	--model_path "<\path\to\resnet50>\resnet50_int8.xml" ^
	--model_name resnet50 ^
	--batch_size 1 ^
	--nwarmup_iters 10 ^
	--dataset imagenet ^
	--device MULTI:CPU,GPU ^
	--multi_device_streams CPU:4,GPU:4 ^
	--nireq 8 ^
	--nthreads 8 ^
	--nstreams 4
