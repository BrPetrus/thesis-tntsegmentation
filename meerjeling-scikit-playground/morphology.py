import skimage
import numpy as np

def fillhole_morphology(arr):
    if type(arr) == list:
        arr = np.array(arr)
    
    # Create the marker image
    t_max = arr.max()
    marker = arr.copy()

    if len(marker.shape) == 1:
        marker[1:-2] = t_max
    if len(marker.shape) == 2:
        print(f"DEBUG: 2D shape with max={t_max}")
        marker[1:-2, 1:-2] = t_max
    if len(marker.shape) == 3:
        marker[:, 1:-2, 1:-2] = t_max

    # Run reconstruction
    res = skimage.morphology.reconstruction(seed=marker, mask=arr, method='erosion')
    return res, marker

def create_umbra(arr):
    if type(arr) == list:
        arr = np.array(arr)
    
    assert len(arr.shape) == 1

    lvls = arr.max()
    umbra = np.zeros(shape=(lvls+1, arr.shape[0]))
    #umbra[-1, :] = 1
    # for t in range(lvls):
    #     umbra[lvls-t, :] = arr <= t
    # return umbra[::-1, :]

    for t in range(lvls+1):
        umbra[t, :] = arr >= t
    return umbra[:, :]