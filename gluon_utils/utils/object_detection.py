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