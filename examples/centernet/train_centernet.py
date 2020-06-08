import mxnet as mx
from mxnet import gluon, autograd, nd
import gluoncv as gcv
from gluon_utils.model_zoo.centernet import get_center_net_transfer
from gluon_utils.losses import CenterNetLoss
from gluon_utils import Accumulator
from tqdm import tqdm, trange

from gluoncv.utils import download
from data.pikachu_dataset import PikachuDataset

classes = ['pikachu']
BATCH_SIZE = 16
EPOCHS = 10

# Get a Centernet model with reset classes

ctx = mx.gpu(0)
net = get_center_net_transfer('center_net_resnet18_v1b_coco', classes, pretrained=True, ctx=ctx)

# Setup Pikachu dataset

url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.rec'
idx_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.idx'
download(url, path='data/pikachu_train.rec', overwrite=False)
download(idx_url, path='data/pikachu_train.idx', overwrite=False)

data = gcv.data.RecordFileDetection('data/pikachu_train.rec')
train_dataset = PikachuDataset(data, classes)
train_data = gluon.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, last_batch='rollover')


# Training Loop

centernet_loss = CenterNetLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})

for e in range(EPOCHS):
    metric = Accumulator(2)
    pbar = tqdm(train_data, desc="Epoch %s"%str(e+1))
    
    for img, heatmap, wh_target, wh_mask, center_reg, center_reg_mask in pbar:
        # Cast to GPU context
        img = img.as_in_context(ctx)
        heatmap = heatmap.as_in_context(ctx)
        wh_target = wh_target.as_in_context(ctx)
        wh_mask = wh_mask.as_in_context(ctx)
        center_reg = center_reg.as_in_context(ctx)
        center_reg_mask = center_reg_mask.as_in_context(ctx)
        
        # Forward and back propogation
        with autograd.record():
            h_pred, wh_pred, reg_pred = net(img)
            loss = centernet_loss(h_pred, heatmap, wh_pred, wh_target, wh_mask, reg_pred, center_reg, center_reg_mask)
        loss.backward()
        trainer.step(BATCH_SIZE)
        
        # Update Loss
        metric.add(loss.sum().asscalar(), BATCH_SIZE)
        pbar.set_postfix(loss=metric[0]/metric[1])
        
    net.save_parameters("data/checkpoints/epoch_{}.params".format(e+1))
