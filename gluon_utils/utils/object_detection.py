import numpy as np

def clip_bbox(bbox, width, height):
    """Clip a bounding box to within an image boundary

    Args:
        bbox (np.ndarray): Bounding box stored in tlbr format
        width (int): The maximum potential x value of the bounding box
        height (int): The maximum potential y value of the boudning box

    Returns:
        np.ndarray: Bounding box cliped to between [0,width] and [0,height]
    """
    ret = np.asarray(bbox).copy()
    
    if ret[0] < 0:
        ret[0] = 0
    if ret[1] < 0:
        ret[1] = 0
    if ret[2] > width:
        ret[2] = width
    if ret[3] > height:
        ret[3] = height

    return ret

def resize_bbox(bbox, orig_scale, target_scale):
    """Resize a bounding box from an image of size `orig_scale` to an image of size `target_scale`

    Args:
        bbox (np.NDArray): Bounding box stored in tlbr format
        orig_scale (tuple): Width and height for of the image that the bounding box comes from
        target_scale (tuple): Width and height to scale target

    Returns:
        np.NDArray: Bounding box scaled to the new dimensions
    """
    resize = bbox.copy()

    resize[0] = (resize[0]*target_scale[0])/orig_scale[0]
    resize[2] = (resize[2]*target_scale[0])/orig_scale[0]
    resize[1] = (resize[1]*target_scale[1])/orig_scale[1]
    resize[3] = (resize[3]*target_scale[1])/orig_scale[1]

    return resize