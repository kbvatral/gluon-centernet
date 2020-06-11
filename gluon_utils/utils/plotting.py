import numpy as np
import matplotlib.pyplot as plt
import cv2
import mxnet

def imshow(image, swapRB=False, rollChannels="infer", grayCmap="infer", axis="on", **kwargs):
    # pylint: disable=no-member
    plt.axis(axis)
    if isinstance(image, mxnet.ndarray.ndarray.NDArray):
        image = image.asnumpy()
    if swapRB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if rollChannels.lower()=="infer":
        if len(image.shape) == 3 and image.shape[0] < 4:
            np.moveaxis(image, 0, -1)
    elif rollChannels.lower()=="true":
        if len(image.shape) == 3:
            np.moveaxis(image, 0, -1)

    gray_cmap = True
    if grayCmap.lower() == "false":
        gray_cmap = False
    if image.shape[-1] == 1:
        image = np.squeeze(image)
        if grayCmap.lower()=="infer":
            gray_cmap = False
    if len(image.shape) == 2 and gray_cmap:
        plt.imshow(image, cmap="gray", **kwargs)
        return
    
    plt.imshow(image, **kwargs)