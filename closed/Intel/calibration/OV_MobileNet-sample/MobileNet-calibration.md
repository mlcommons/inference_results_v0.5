1. Install latest OpenVINO (https://software.intel.com/en-us/openvino-toolkit/choose-download)

2. Run setup_vars.sh
	source /opt/intel/openvino/bin/setup_vars.sh

3. Download MLPerf calibration image set from this [location](https://github.com/mlperf/inference/tree/master/calibration/ImageNet). From one of this list, prepare a test file as shown in this  [file](./imagenet_mlperf/converted_mlperf_list.txt)

4. Download MobileNet FP32 model https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz

5. Create annotation files for the calibration image set downloaded from step 3

    1. convert_annotation tool is available in /opt/intel/openvino/deployment_tools/tools/accuracy_checker_tool)

    2. $convert_annotation imagenet --annotation_file ./imagenet_mlperf/converted_mlperf_list.txt --labels_file ./imagenet_mlperf/synset_words.txt --has_background True

    3. After this step you will see two files (*.pickle and *.json)

6. Edit [MobileNet YAML file](mobilenet_v1_cal_list_1.yml) file in this folder and set paths to dataset, annotation and models
    1. If calibrating to generate INt8 model files for CPU, keep the models->launchers->framework>mo_params->data_type to FP32.
    2. If  calibrating to generate INT8 model files for CPU and GPU, keep models->launchers->framework>mo_params->data_type to FP16 (as existing Intel integrated Gfx supports FP16, it is required to calibrate from FP16 weights)

7. Run calibration script to generate int8 models

    1. cd /opt/intel/openvino/deployment_tools/tools/calibration_tool

    2. python3 calibrate.py -c <Path_to_config> -M /opt/intel/openvino/deployment_tools/model_optimizer -e /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64 -tf dlsdk -td CPU --output_dir <Output_dir>

    3. This step will generate OpenVINO IR files for MobileNet v1 ( *.bin and *.xml files )
