import numpy as np
import scipy
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import Any, Optional, List, Tuple

import scipy.linalg
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def percentile_stretch(arr: NDArray[np.unsignedinteger | np.floating], low: float = 0.05, high: float = 0.95) -> NDArray[np.float64]:
    arr = arr.copy()
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        arr = arr.astype(np.float64)
    elif not np.issubdtype(arr.dtype, np.floating):
        raise ValueError("Unsupported dtype")
    ql, qh = np.quantile(arr, low), np.quantile(arr, high)
    arr[arr < ql] = ql
    arr[arr > qh] = qh
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

# def hessian_matrix_anisotropic():

# def meijering_filter_alternative(arr: NDArray[np.floating], sigmas: List[float], alpha: Optional[float] = None) -> NDArray[np.float64]:
#     if not np.issubdtype(arr.dtype, np.floating):
#         raise ValueError("Expected a floating dtype")

#     # Calculate the hessian matrix
#     hessian = hessian_matrix(arr, sigma=sigmas, mode='constant', use_gaussian_derivatives=False)
#     hxx, hxy, hxz, hyy, hyz, hzz = hessian

#     new_hessian = np.zeros_like(hessian)
#     # # Modify the hessian
#     # for z in range(arr.shape[0]):
#     #     for y in range(arr.shape[1]):
#     #         for x in range(arr.shape[0]):
#     #             new_hessian[]
#     # new_hessian[0] = hxx + alpha/2. * (hyy+hzz)
#     # new_hessian[1] = (1 - alpha/2.) * hxy
#     # new_hessian[2] = (1 - alpha/2.) * hxz
#     # new_hessian[3] = hyy + alpha/2. * (hxx + hzz)
#     # new_hessian[4] = (1 - alpha/2.) * hyz
#     # new_hessian[5] = hzz + alpha/2. * (hxx + hyy)

#     # Calculate the eigenvals
#     eigens = hessian_matrix_eigvals(hessian)



#     max_eigenval_at_pixel = np.take_along_axis(
#         eigens,
#         np.abs(norm_eigenvals).argmax(0)[None],
#         0).squeeze(0)
#     minimum_eigenval_global = norm_eigenvals.min()

#     max_eigenval_at_pixel[ max_eigenval_at_pixel >= 0] = 0
#     max_eigenval_at_pixel /= minimum_eigenval_global

#     return max_eigenval_at_pixel


def meijering_filter(arr: NDArray[np.floating], sigmas: List[float], alpha: Optional[float] = None) -> NDArray[np.float64]:    
    # This is a modified version of the scikit-image meijering filter.
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError("Expected a floating dtype")

    # Calculate the hessian matrix
    hessian = hessian_matrix(arr, sigma=sigmas, mode='constant', use_gaussian_derivatives=False)

    # Calculate the eigenvals
    eigens = hessian_matrix_eigvals(hessian)

    if alpha is None:
        alpha = 1 / (arr.ndim + 1)
    mult_factors = scipy.linalg.circulant([1, *[alpha] * (arr.ndim - 1)]).astype(arr.dtype)

    result = np.zeros_like(arr)
    # l_i = e_i + sum_{j!=i} 
    norm_eigenvals = np.tensordot(mult_factors, eigens, 1)
    max_eigenval_at_pixel = np.take_along_axis(
        norm_eigenvals,
        np.abs(norm_eigenvals).argmax(0)[None],
        0).squeeze(0)
    minimum_eigenval_global = norm_eigenvals.min()

    max_eigenval_at_pixel[ max_eigenval_at_pixel >= 0] = 0
    max_eigenval_at_pixel /= minimum_eigenval_global

    return max_eigenval_at_pixel

def main(path: str, sigma_1, sigma_2, sigma_3):
    import tifffile
    import matplotlib.pyplot as plt
    import skimage.morphology as skmorph
    raw_data = tifffile.imread(path).astype(np.float64)
    arr = percentile_stretch(raw_data)
    # arr = -1 * percentile_stretch(raw_data) + 1.0
    # arr = skmorph.erosion(arr, skmorph.disk(3))
    print(f"Array shape {arr.shape}")
    res = meijering_filter(arr, [float(sigma_1), float(sigma_2), float(sigma_3)], -1/3)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(arr[2], 'gray')
    ax[1].imshow(res[2], 'gray')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
