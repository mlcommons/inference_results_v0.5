import  numpy as np
import  mxnet as mx
from mxnet.gluon.data.vision import transforms
import os
import sys
import re
import habana as hb
from onnx import  helper
import onnx
from util import *
from onnx import AttributeProto, TensorProto, GraphProto
from  dataset import  pre_process_vgg,pre_process_coco_resnet34,pre_process_coco_resnet34_tf
import cv2
import json
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO


def resnet50_prepare_calib_dataset(img_list_file_name,img_path=''):
    if(img_path==''):
        print('path parameters is empty')
        return
    if(img_list_file_name == ''):
        print('nned image list file name')
        return
    first = True
    with open(img_list_file_name, 'r') as f:
        for image_name in f:
            input_img_full_name = os.path.join(img_path, image_name.strip())
            img = cv2.imread(input_img_full_name)
            #img = mx.image.imread(input_img_full_name)
            img = pre_process_vgg(img,[224,224,3])
            img = img.transpose((2, 0, 1))  # transpose input
            img = np.expand_dims(img, axis=0)  # batchify
            if(first):
                dataset = img.copy()
                first = False
            else:
                dataset=np.ma.concatenate([dataset,img.copy()],axis=0)
    return dataset


def mlperf_resnet50_compile_recipe(model_dir, onnx_model_name, calib_image_list_file, image_dir, BATCH_SIZE):
    print('loading onnx file: ', onnx_model_name)
    print('onnx model directory path: ', model_dir)
    base_name = os.path.splitext(os.path.basename(onnx_model_name))[0]
    RECIPE_NAME = base_name + '_batch' + str(BATCH_SIZE) +'.recipe'
    NEW_ONNX   = base_name + '_batch' + str(BATCH_SIZE) +'.onnx'
    RECIPE_NAME = os.path.join(model_dir,RECIPE_NAME)
    NEW_ONNX = os.path.join(model_dir,NEW_ONNX)
    onnx_model_name = os.path.join(model_dir,onnx_model_name)
    input_shape = [BATCH_SIZE, 3, 224, 224]
    print('input_shape: ',input_shape)
    onnx_model = onnx.load(onnx_model_name)
    print('Removing softmax from onnx and adding globalAveragePool')
    for i,node in zip(range(len(onnx_model.graph.node)),onnx_model.graph.node):
        if (node.name == 'graph_outputs_Identity__6'):
            softmax_identity = node
        if (node.op_type == 'ReduceMean'):
            ri = i
        if (node.name == 'graph_outputs_Identity__4'):
            argmax_identity = node
        # bypass for identity issue
        if (node.op_type == 'ArgMax'):
            node.output[0] = 'ArgMax:0'
    onnx_model.graph.node.remove(softmax_identity)
    #bypass for identity issue
    onnx_model.graph.node.remove(argmax_identity)
    for node in onnx_model.graph.node:
        if (node.op_type == 'Softmax'):
            onnx_model.graph.node.remove(node)
    for output in onnx_model.graph.output:
        if (output.name == 'softmax_tensor:0'):
            onnx_model.graph.output.remove(output)
    onnx_model.graph.node.insert(ri+1, helper.make_node('GlobalAveragePool', [onnx_model.graph.node[ri].input[0]],
                                                     [onnx_model.graph.node[ri].output[0]],
                                                     'GlobalAveragePool'))

    onnx_model.graph.node.remove(onnx_model.graph.node[ri])
    input_name = ['input_tensor:0']
    output_name = ['ArgMax:0']
    profile = hb.profile.Performance()
    profile.num_low_precision_layers_to_raise = 0
    profile.inputs_as_nhwc = True
    hb_model = hb.model.create(profile)
    print('saving new onnx model file: ', NEW_ONNX)
    onnx.save(onnx_model, NEW_ONNX)
    print('Genarating Habana model from onnx model: ',NEW_ONNX)
    hb.model.from_onnx(hb_model, NEW_ONNX)
    print('Preparing calibration dataset')
    data_load = resnet50_prepare_calib_dataset(calib_image_list_file,image_dir)
    mxnet_iter = mx.io.NDArrayIter(data=data_load, label=None, batch_size=32,data_name=input_name[0])
    print('Running calibration')
    dynamic_range = hb.model.calibrate.measure_mxnet(hb_model, mx.cpu(), mxnet_iter, 16)
    print('Running Habana model quantization')
    hb.model.calibrate.quantize(hb_model, dynamic_range)
    hb.model.set_inputs_info(hb_model, {input_name[0]: {"shape": input_shape, "batch_pos": 0}})
    hb.model.set_outputs_info(hb_model, {output_name[0]: {"batch_pos": 0}})
    print('compiling recipe file: ',RECIPE_NAME)
    hb.model.compile_recipe(hb_model, RECIPE_NAME, compile_params={"visualization": 1})
    print('DONE')






def preprocess_imagenet_images(image_count,image_dir,model_dir,habana_recipe=None,BATCH_SIZE=10):
    ocnt = image_count;
    base,last_dir = os.path.split(image_dir)
    habana_recipe = os.path.join(model_dir,habana_recipe)
    output_file_dir = os.path.join(base, "imagenet_habana")
    os.makedirs(output_file_dir, exist_ok=True)
    full_input_list_name = os.path.join(image_dir, "val_map.txt")
    full_output_list_name = os.path.join(output_file_dir, "val_map.txt")
    output_list_pid = open(full_output_list_name,'w')
    input_name = 'input_tensor:0'
    input_shape = [BATCH_SIZE, 3, 224, 224]
    profile = hb.profile.Performance()
    profile.num_low_precision_layers_to_raise = 0
    profile.inputs_as_nhwc = True
    hb_model = hb.model.create(profile)
    device_id = hb.device.acquire()
    if(os.path.exists(habana_recipe) == False):
        print("didn't find the recipe file: ",habana_recipe)
        return
    topology_id = hb.device.load(device_id, habana_recipe)
    hb.device.activate(device_id, topology_id)
    tensors_info = hb.device.get_inputs_outputs_info(device_id, topology_id)
    input_info = tensors_info['inputs'][input_name]
    if(image_count <= 0):
        return
    with open(full_input_list_name, 'r') as f:
        for s in f:
            image_name, label = re.split(r"\s+", s.strip())
            output_file_name = os.path.splitext(os.path.basename(image_name))[0];
            output_file_name = output_file_name + ".bin"
            input_img_full_name = os.path.join(image_dir, image_name)
            output_img_full_name = os.path.join(output_file_dir, output_file_name)
            img = cv2.imread(input_img_full_name)
            #img = mx.image.imread(input_img_full_name)
            img = pre_process_vgg(img,[224,224,3])
            input_data =  img #img.transpose((1, 2, 0)).copy()  # transpose input
            input_data = quantize(input_data, zp=input_info['zp'], scale=input_info['scale'], dtype=input_info['data_type'])
            output_fid = open(output_img_full_name,'w')
            input_data.tofile(output_fid)
            output_fid.close()
            line = output_file_name + " " + str(label) + "\n"
            output_list_pid.write(line)
            image_count=image_count-1
            if (image_count == 0):
                break
            if(((ocnt-image_count) % 500) == 0):
                print('processed:',ocnt-image_count,' images')
    output_list_pid.close()


def print_option1_command_line():
    print("options[1] - compiling Habana recipe file")
    print('param[1] - True')
    print('param[2] - Model directory full path')
    print('param[3] - Onnx model file name')
    print('param[4] - Calib image list file name')
    print('param[5] - Source image directory')
    print('param[6] - Batch size')

def print_option2_command_line():
    print("option[2]- pre process the imagenet database")
    print('param[1] - False')
    print('param[2] - Number of images to be preprocessed')
    print('param[3] - Source image directory name')
    print('param[4] - Model directory full path')
    print('param[5] - Habana recipe file name')
    print('param[6] - Batch size')


def main():
    isPrepareRecipe = False
    print(len(sys.argv))
    if (len(sys.argv) < 7):
        print_option1_command_line()
        print_option2_command_line()
    else:
        if(sys.argv[1] == 'True'):
            isPrepareRecipe = True
        elif(sys.argv[1] == 'False'):
            isPrepareRecipe = False
        else:
            print('Need to choose between two options:')
            print('options 1 - Habana recipe compilation')
            print_option1_command_line()
            print('options 2 - image dataset preprocess')
            print_option2_command_line()
            return
        if(isPrepareRecipe):
            run_func = mlperf_resnet50_compile_recipe
            if(len(sys.argv) == 7):
                batch_size = int(sys.argv[6])
                run_func(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5], batch_size)
            else:
                print_option1_command_line()
        else:
            run_func = preprocess_imagenet_images
            if(len(sys.argv) == 7):
                batch_size = int(sys.argv[6])
                image_count = int(sys.argv[2])
                run_func(image_count,sys.argv[3], sys.argv[4],sys.argv[5], batch_size)
            else:
               print_option2_command_line()

if __name__ == "__main__":
    main()
