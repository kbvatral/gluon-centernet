import mxnet as mx
import gluoncv as gcv
from mxnet.gluon import nn

def reset_classes(net, classes):
    with net.name_scope():
        # Reinitialize classes and heads
        net.classes = classes
        new_heads = nn.HybridSequential('heads')

        # Reset the output of the heatmap head with new classes
        new_heatmap = nn.HybridSequential('heatmap')
        new_heatmap.add(net.heads[0][0])
        new_heatmap.add(net.heads[0][1])
        ctx = net.heads[0][2].collect_params().list_ctx()
        new_heatmap_out = nn.Conv2D(len(classes), kernel_size=1, strides=1,
                                    padding=0, use_bias=True, weight_initializer=mx.init.Xavier())
        new_heatmap_out.collect_params().initialize(ctx=ctx)
        new_heatmap.add(new_heatmap_out)

        # Reset self's heads to newly initialized
        new_heads.add(new_heatmap)
        new_heads.add(net.heads[1])
        new_heads.add(net.heads[2])
        net.heads = new_heads

    return net

def get_center_net_transfer(base_model_name, classes, **kwargs):
    net = gcv.model_zoo.get_model(base_model_name, **kwargs)
    net = reset_classes(net, classes)

    return net
