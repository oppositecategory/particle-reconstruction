import numpy as np
import matplotlib.pyplot as plt 
import cv2

import torch
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta


import mrcfile

from PIL import Image 

def translate_image(image, tx, ty):
    # Continous translation using bilinear interpolation
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32) 
    translated = cv2.warpAffine(image, M, image.shape, flags=cv2.INTER_LINEAR)  
    return translated   


def generate_data_old(N, std=20, std_xy=2,size=256):
    img = Image.open("image.png").convert("L").resize((256,256))
    img = np.array(img)

    n = size
    thetas = np.random.uniform(0,2*np.pi,N)
    translations = np.random.multivariate_normal(mean=[0,0], 
                                             cov=std_xy *np.identity(2),
                                             size=N)
    #scales = np.random.beta(a=2,b=5,size=N)*5
    noise_matrices = np.random.normal(scale=std,size=(N,n,n))

    center =  (n//2,n//2)
    rotated_images = np.array(
        [
            cv2.warpAffine(img,
                           cv2.getRotationMatrix2D(center,angle * 180 / np.pi,scale=1),dsize=(n,n)) 
            for angle in thetas
    ])

    translated_images = np.array([
        translate_image(img, t[0],t[1]) for img,t in zip(rotated_images,translations)
    ])
    #scaled_images = np.array([s*img for img,s in zip(translated_images,scales)])
    #data = scaled_images + noise_matrices
    data = translated_images + noise_matrices

    data = data.astype(np.float32)

    with mrcfile.new('data1.mrc',overwrite=True) as mrc:
        mrc.set_data(data)

def generate_data(K,std=0.01,std_xy=0.2,n=256):
    img = Image.open("image.png").convert("L").resize((n,n))
    img = torch.tensor(np.array(img),dtype=torch.float32)

    thetas = (2 * torch.pi) * torch.rand(size=(K,))
    normal_dist = MultivariateNormal(loc=torch.zeros(2), 
                                     covariance_matrix=std_xy * torch.eye(2))
    beta_dist = Beta(2, 5)
    translations = normal_dist.sample((K,))

    transformations = torch.zeros((K, 2, 3))
    transformations[:, 0, 0] = torch.cos(thetas)
    transformations[:, 0, 1] = -torch.sin(thetas)
    transformations[:, 1, 0] = torch.sin(thetas)
    transformations[:, 1, 1] = torch.cos(thetas)
    transformations[:, :, 2] = translations

    grid = F.affine_grid(transformations, size=(K, 1, n, n), align_corners=False)
    data = F.grid_sample(img.expand(K, 1, n, n), grid, align_corners=False).squeeze(1)
    data = data + torch.normal(0,std,size=(K,n,n))

    data = data.numpy().astype(np.float32)
    with mrcfile.new('data.mrc',overwrite=True) as mrc:
        mrc.set_data(data)

if __name__ == "__main__":
    generate_data(800)