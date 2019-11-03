import  numpy as np
import argparse
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
import mxnet as mx
import numpy as np
import onnx 
from onnx import numpy_helper
from torchvision import transforms
from mx_r34_v1_ssd_1200 import bind_mxnet_model

class PyTorchPreProcess(object):
    def __init__(self, img_size=None):
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def preprocess(self, raw_input_batch):        
        inp = []
        for img in raw_input_batch:
            tmp_ = self.preprocess_transform(img).numpy()
            inp.append(tmp_)
        return inp

class import_weight_and_compile_ssd_resnet():
    def __init__(self,img_size,coco_cal_images_list, coco_base,recipe_name):
        hb.synapse.synapse_init()
        self._device_id = hb.device.acquire()
        self._recipe_name = recipe_name
        self.img_size = img_size
        self.preprocess_transform = PyTorchPreProcess(self.img_size)
        self.coco_cal_images_list = coco_cal_images_list
        self.coco_base = coco_base
    def compile(self,mx_model_path):
        input_shape = [1, 3, 1200, 1200]
        data_name = 'data'

        # path to Mxnet model
        file_path = './'
        mxnet_model = os.path.join(file_path, mx_model_path)

        # Set multibox loc pred layer to fp32
        user_quant_params = {}
        user_quant_params['multibox_loc_pred_output'] = {'zp': 0, 'scale': 1, 'dtype': "float32"}

        # init profile
        profile = hb.profile.Performance()
        profile.num_low_precision_layers_to_raise = 0
        profile.user_quantization = user_quant_params
        profile.inputs_as_nhwc = True

        hb_model = hb.model.create(profile)
        hb.model.from_mxnet(hb_model, mxnet_model, 0)

        input_data = self.get_inputs_for_mes()
        data_iter = mx.io.NDArrayIter(input_data, label=None, data_name=data_name, batch_size=16)
        print('Running calibration and quantization...')
        dynamic_range = hb.model.calibrate.measure_mxnet(hb_model, mx.cpu(), data_iter)
        hb.model.calibrate.quantize(hb_model, dynamic_range)

        hb.model.set_inputs_info(hb_model, {"data": {"shape": input_shape, "batch_pos": 0}})
        hb.model.set_outputs_info(hb_model, {"multibox_loc_pred_output": {"batch_pos": 0}, "cls_pred_concat_output": {"batch_pos": 0}})
        print('Compiling recipe ...')
        hb.model.compile_recipe(hb_model, self._recipe_name, compile_params={})

        topology_id = hb.device.load(self._device_id, self._recipe_name)
        hb.device.activate(self._device_id, topology_id)

        hb.device.release_device(self._device_id)
        hb.synapse.destroy()

    def get_inputs_for_mes(self):
        desc = [mx.io.DataDesc('data', (1,) + (3, 1200, 1200))]
        inputs_data = self.data_for_mes(desc)
        return inputs_data

       
    
    def data_for_mes(self, desc):
        all_data = None
    
        image_path = self.coco_base + "calibration_dataset"
        coco_annot = self.coco_base + 'annotations/instances_train2017.json'
        im_ids = []
        with open(self.coco_cal_images_list, 'rb') as f:
            cal_file_names = f.readlines()
        for l in cal_file_names:
            l = int(l.lstrip(b'0')[:-5])
            im_ids.append(l)
    
        batch_size = desc[0].shape[0]
        tot_num_batch = len(im_ids) // batch_size 
        cocoGt = COCO(annotation_file=os.path.join(coco_annot))
        for n_b in range(tot_num_batch):
            raw_inp = []
            valid_image_info = []
            im_ids_ = im_ids[n_b * batch_size:(n_b + 1) * batch_size]
    
            for k, im_id in enumerate((im_ids_)):
                img = Image.open(os.path.join(image_path, cocoGt.loadImgs([im_id])[0]['file_name'])).convert("RGB")
                raw_inp.append(img)
                valid_image_info.append((im_id, img.height, img.width))
    
            after_preprocess = mx.nd.array(self.preprocess_transform.preprocess(raw_inp))

            if n_b%100==0:
                print("finish: ", n_b)
            if all_data is None:
                all_data = after_preprocess
            else:
                all_data = mx.nd.concat(all_data, after_preprocess, dim=0)
        print("finish: ", n_b)    
        return all_data.asnumpy()
    
    def extract_weights_from_onnx(self,mode_onnx_path):
        model=onnx.load(mode_onnx_path)
        weight_dict={}
        INTIALIZERS=model.graph.initializer
        for initializer in INTIALIZERS:
            w= numpy_helper.to_array(initializer)
            weight_dict[initializer.name]=w
        return weight_dict
    
    def export_pytorch_params(self,onnx_params_path,output_path=None):
        mx_params = mx.nd.load(output_path)
        onnx_weight_dict = self.extract_weights_from_onnx(onnx_params_path)
        target={}
        missing=[]
        matched=set()
        for key in onnx_weight_dict.keys():
            if 'num_batches_tracked' in key:
                continue
            
            mx_key = 'aux:'+key if 'running' in key else 'arg:'+key
            mx_key = mx_key.replace('.running','_moving').replace('.weight','_weight').replace('.bias','_bias')
            bn_key=key.replace('.bias','.weight')    
            if 'bn' in key or (bn_key in onnx_weight_dict.keys() and  len(onnx_weight_dict[bn_key].shape)==1):            
                mx_key = mx_key.replace('_weight','_gamma').replace('_bias','_beta')
            target[mx_key]=mx.nd.array(onnx_weight_dict[key])
            if mx_key in mx_params.keys():
                matched.add(key)
            else:
                missing.append(key)
        assert len(missing)==0, 'missing keys in dict'
    
        print('saving ported params to', output_path)
        mx.nd.save(output_path,target)
    
def preprocess_coco_images(image_count,image_dir, model_dir,habana_recipe=None,BATCH_SIZE=10):
    ocnt = image_count; #ocnt used for printouts
    base,last_dir = os.path.split(image_dir)
    habana_recipe = os.path.join(model_dir, habana_recipe)
    output_file_dir = os.path.join(base, "coco_habana")
    os.makedirs(output_file_dir, exist_ok=True)
    full_input_list_name = os.path.join(image_dir, "annotations/instances_val2017.json")
    full_output_list_name = os.path.join(output_file_dir, "val_map.txt")
    input_name = 'data'
    input_shape = [BATCH_SIZE, 3, 1200, 1200]
    profile = hb.profile.Performance()
    profile.num_low_precision_layers_to_raise = 0
    profile.inputs_as_nhwc = True
    hb_model = hb.model.create(profile)
    device_id = hb.device.acquire()
    if (os.path.exists(habana_recipe) == False):
        print("didn't find the recipe file: ", habana_recipe)
        return
    topology_id = hb.device.load(device_id, habana_recipe)
    hb.device.activate(device_id, topology_id)
    tensors_info = hb.device.get_inputs_outputs_info(device_id, topology_id)
    input_info = tensors_info['inputs'][input_name]
    preprocess_transform = PyTorchPreProcess(1200)
    
    images = []
    output_list_pid = open(full_output_list_name, 'w')
    coco_db = COCO(annotation_file=full_input_list_name)
    im_ids = []
    for i in coco_db.getCatIds():
        im_ids += coco_db.catToImgs[i]
    im_ids = list(set(im_ids))
    image_list = [coco_db.imgs[id]["file_name"] for id in im_ids]
    for image_name in image_list:
        output_file_name = os.path.splitext(os.path.basename(image_name))[0];
        output_file_name = output_file_name + ".bin"
        input_img_full_name = os.path.join(image_dir, "val2017")
        input_img_full_name = os.path.join(input_img_full_name, image_name)
        output_img_full_name = os.path.join(output_file_dir, output_file_name)
        if (os.path.exists(input_img_full_name) == False):
            print("didn't find the image file: ", input_img_full_name)
            return
        img = Image.open(input_img_full_name).convert('RGB')
        input_data = preprocess_transform.preprocess(img)
        input_data = input_data.transpose((1, 2, 0)).copy()
        input_data = quantize(input_data, zp=input_info['zp'], scale=input_info['scale'], dtype=input_info['data_type'])
        output_fid = open(output_img_full_name,'w')
        input_data.tofile(output_fid)
        output_fid.close()
        line = output_file_name + " " + str(1) + "\n"
        output_list_pid.write(line)
        image_count=image_count-1
        if (image_count == 0):
            break
        if(((ocnt-image_count) % 250) == 0):
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
    print("option[2]- pre process the coco database")
    print('param[1] - False')
    print('param[2] - Number of images to be preprocessed')
    print('param[3] - Source image directory name')
    print('param[4] - Model directory full path')
    print('param[5] - Habana recipe file name')
    print('param[6] - Batch size')

def arg_parser():
    parser = argparse.ArgumentParser(description='Parsing Arguments')
    parser.add_argument('-cr', '--compile_recipe',  help='Compile recipe', action='store_true', default=False)
    parser.add_argument('-m', '--model_directory',  help='Model Name', type=str, default='./')
    parser.add_argument('-rm', '--referance_model',  help='path to referance model given by mlperf', type=str, default='../mlperf_models/resnet34-ssd1200.onnx')
    parser.add_argument('-cal', '--calibration_list',  help='Path to the calibration list from mlperf', type=str, required=True)
    parser.add_argument('-ip', '--image_path',  help='Path to the Calibration images directory', type=str, required=True)
    parser.add_argument('-b', '--batch_size',  help='Batch size for compilation', type=int, default=1)
    parser.add_argument('-pr', '--preprocess_data',  help='Run Preprocessing', action='store_true', default=False)
    parser.add_argument('-hr', '--habana_recipe_path',  help='Path to Habana recipe', type=str, default='ssd_resnet_recipe')
    parser.add_argument('-ic', '--image_count',  help='Number of images to be preprocessed', type=int)

    args = parser.parse_args()
        
    return args
    
def main():
    args=arg_parser()
    if args.compile_recipe:
        mx_model_path = bind_mxnet_model()
        onnx_params_path= args.referance_model
        model_creator=import_weight_and_compile_ssd_resnet(1200,coco_cal_images_list=args.calibration_list, coco_base=args.image_path,recipe_name=args.habana_recipe_path)
        if not os.path.exists(onnx_params_path):
            os.system('sh download_ssd_r34_model.sh '+onnx_params_path)   
        output_path = mx_model_path + '-0000.params'
        model_creator.export_pytorch_params(onnx_params_path=onnx_params_path,output_path=output_path)
        model_creator.compile(mx_model_path)
        return
    if args.preprocess_data:
        run_func = preprocess_coco_images
        batch_size = args.batch_size
        image_count = args.image_count
        run_func(image_count,args.image_path, args.model_directory,args.habana_recipe_path, batch_size)

    if not args.compile_recipe and not args.preprocess_data:
        print('Please choose at least one --preprocess_data or --complie_rcipe')       

if __name__ == "__main__":
    main()
