import mxnet as mx
from mxnet import nd, gluon
from gluoncv.model_zoo.center_net.target_generator import CenterNetTargetGenerator
from gluoncv.data.transforms.presets.center_net import transform_test
import cv2

class PikachuDataset(gluon.data.Dataset):
    def __init__(self, record_file, classes, im_shape=(512,512), label_shape=(128,128), **kwargs):
        super(PikachuDataset, self).__init__(**kwargs)
        
        self.record_file = record_file
        self.classes = classes
        self.im_shape = im_shape
        self.label_shape = label_shape
        self.gen = CenterNetTargetGenerator(len(classes), label_shape[0], label_shape[1])
        
    def __len__(self):
        return len(self.record_file)
    
    def __getitem__(self, idx):
        image, label = self.record_file[idx]
        image = image.asnumpy()
        orig_h, orig_w = image.shape[:2]
        
        # Resize input image and label to 512x512
        image = cv2.resize(image, self.im_shape)
        label[:,0] = (label[:,0]*self.im_shape[0])/orig_w
        label[:,2] = (label[:,2]*self.im_shape[0])/orig_w
        label[:,1] = (label[:,1]*self.im_shape[1])/orig_h
        label[:,3] = (label[:,3]*self.im_shape[1])/orig_h
        
        # Resize label to downsampled size
        label[:,0] = (label[:,0]*self.label_shape[0])/self.im_shape[0]
        label[:,2] = (label[:,2]*self.label_shape[0])/self.im_shape[0]
        label[:,1] = (label[:,1]*self.label_shape[1])/self.im_shape[1]
        label[:,3] = (label[:,3]*self.label_shape[1])/self.im_shape[1]
        
        heatmap, wh_target, wh_mask, center_reg, center_reg_mask = self.gen(label[:,:4], label[:,4])
        x, _ = transform_test(nd.array(image), short=min(self.im_shape[0], self.im_shape[1]))
        x = nd.squeeze(x)
        
        return x, heatmap, wh_target, wh_mask, center_reg, center_reg_mask