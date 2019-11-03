@echo OFF

call <\path\to\Release>\ov_mlperf.exe --scenario Offline ^
	--mode Performance ^
	--mlperf_conf_filename "<\path\to\mlperf.conf" ^
	--user_conf_filename "<\path\to\ssd-mobilenet>\user.conf" ^
	--total_sample_count 50000 ^
	--data_path "<\path\to\CK-TOOLS>\dataset-coco-2017-val\" ^
	--model_path "<\path\to\ssd-mobilenet>\ssd-mobilenet_int8.xml ^
	--model_name ssd-mobilenet ^
	--batch_size 1 ^
	--nwarmup_iters 50 ^
	--dataset coco ^
	--device CPU ^
	--nireq 4 ^
	--nthreads 4 ^
	--nstreams 4
