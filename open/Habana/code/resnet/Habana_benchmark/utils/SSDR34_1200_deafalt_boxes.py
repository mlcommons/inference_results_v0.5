    
import numpy as np
from math import sqrt, ceil
import itertools

class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                       scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size_w,self.fig_size_h = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh
        
        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps_w = [st[0] for st in steps]
        self.steps_h = [st[1] for st in steps]
        self.scales = scales
        fkw = self.fig_size_w//np.array(self.steps_w)
        fkh = self.fig_size_h//np.array(self.steps_h)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sfeat_w,sfeat_h=sfeat
            sk1 = scales[idx][0]/self.fig_size_w
            sk2 = scales[idx+1][1]/self.fig_size_h
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat_w), range(sfeat_h)):
                    cx, cy = (j+0.5)/fkh[idx], (i+0.5)/fkw[idx]
                    self.default_boxes.append((cx, cy, w, h)) 
        self.dboxes = np.array(self.default_boxes)
        self.dboxes = np.clip(self.dboxes,0,1)
        boxes = open(r"boxes.h", "w")
        s = 'float base_boxes[' + str(15130*4) + '] = {\n'
        print(s,file=boxes)
        #boxes.write(s)
        for i in range(self.dboxes.shape[0]):
            print('%7.5f,' %self.dboxes[i,0],'%7.5f,' %self.dboxes[i,1], '%7.5f,' %self.dboxes[i,2],  '%7.5f,' %self.dboxes[i,3],file=boxes)
        boxes.write('};')
        boxes.close()
    @property
    def scale_xy(self):
        return self.scale_xy_
    
    @property    
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self):
        return self.dboxes

def main():
    figsize = [1200, 1200]
    feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size] #steps = [(24, 24), (48, 48), (92, 92), (171, 171), (400, 400), (400, 400)]
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]] # scales =[(84, 84), (180, 180), (396, 396), (612, 612), (828, 828), (1044, 1044), (1260, 1260)]
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]] 
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    dboxes_xywh = dboxes()
    #import pdb; pdb.set_trace()
    np.save('dboxes_xywh',dboxes_xywh)
if __name__ == '__main__':
    main()