import numpy as np
from multiprocessing import Pool
from functools import partial
from matplotlib import pyplot as plt


def mask_random_pixels(imgs, mask_ratio=.2):
    mask = np.random.binomial(1, 1 - mask_ratio, imgs.shape[:-1])
    mask = mask.reshape([*mask.shape, 1])
    mask = np.concatenate([mask] * 3, axis=-1)
    return mask


def generate_masks(imgs, n_masks, mask_size):
    if len(imgs.shape) == 3:
        imgs = imgs[np.newaxis, :]
    mask = np.ones_like(imgs)
    masked_patches = np.empty((len(imgs), mask_size[0], mask_size[1], 3*n_masks))
    indices = np.random.randint([0, 0],
                                [mask.shape[1] - mask_size[0] - 1, mask.shape[2] - mask_size[1] - 1],
                                [len(imgs), n_masks, 2])
    for idx, start_points in enumerate(indices):
        for idx1, start in enumerate(start_points):
            mask[idx, start[0]:start[0] + mask_size[0], start[1]:start[1] + mask_size[1]] = 0
            masked_patches[idx, :, :, idx1:idx1+3] = imgs[idx, start[0]:start[0] + mask_size[0], start[1]:start[1] + mask_size[1]]
    return mask, masked_patches, indices


def mask_random_areas(imgs, n_masks=10, mask_size=(3, 3), parallelise=False, n_pools=4):
    if not parallelise:
        return generate_masks(imgs, n_masks=n_masks, mask_size=mask_size)

    func = partial(generate_masks, n_masks=n_masks, mask_size=mask_size)
    processing_pools = Pool(n_pools)
    offset = len(imgs)//n_pools
    img_subsets = [imgs[idx:idx+offset] for idx in range(0, len(imgs), offset)]
    return np.concatenate(processing_pools.map(func, img_subsets), axis=0)


if __name__ == '__main__':
    img = plt.imread('target_faces/test/El-noruego-Erling-Haaland.jpg')
    img = np.array([img, img, img, img])
    mask1 = mask_random_pixels(img)
    mask2 = mask_random_areas(img, mask_size=(100, 100))
    img1 = mask1 * img
    img2 = mask2 * img
    plt.subplot(311)
    plt.imshow(img[0])
    plt.subplot(312)
    plt.imshow(img1[0])
    plt.subplot(313)
    plt.imshow(img2[0])
    plt.show()
