@echo OFF

call <\path\to\Release>\ov_mlperf.exe --scenario SingleStream ^
	--mode Performance ^
	--mlperf_conf_filename "<\path\to\mlperf.conf" ^
	--user_conf_filename "<\path\to\resnet50>\user.conf" ^
	--total_sample_count 50000 ^
	--data_path "<\path\to\CK-TOOLS>\dataset-imagenet-ilsvrc2012-val\" ^
	--model_path "<\path\to\resnet50>\resnet50_int8.xml" ^
	--model_name resnet50 ^
	--batch_size 1 ^
	--nwarmup_iters 50 ^
	--dataset imagenet ^
	--device CPU ^
	--nireq 1 ^
	--nthreads 4 ^
	--nstreams 1
