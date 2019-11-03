import os
import numpy as np


def read_debug(cls_dbg_name, multibox_dbg_name, dbg_path):
    cls_dbg_name_npy = os.path.splitext(os.path.basename(cls_dbg_name))[0];
    multibox_dbg_name_npy = os.path.splitext(os.path.basename(multibox_dbg_name))[0];
    cls_dbg_name_npy = cls_dbg_name_npy + '.npy'
    multibox_dbg_name_npy = multibox_dbg_name_npy + '.npy'
    cls_full_name = os.path.join(dbg_path,cls_dbg_name)
    multibox_full_name = os.path.join(dbg_path,multibox_dbg_name)
    cls = np.fromfile(cls_full_name,np.int16,15130*81)
    cls = cls.reshape((81,15130))
    multibox = np.fromfile(multibox_full_name,np.float32,4*15130)
    multibox = multibox.reshape((4,15130))
    np.save(os.path.join(dbg_path,cls_dbg_name_npy),cls)
    np.save(os.path.join(dbg_path,multibox_dbg_name_npy),multibox)
    return cls,multibox



ls,multi = read_debug('cls_pred_concat_output_91.bin' ,'multibox_loc_pred_output_91.bin','/home/labuser/projects/demos/Habana_benchmark/benchmark_app/app/Debug/dbg')