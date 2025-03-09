import torch
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta

import numpy as np

import mrcfile

from PIL import Image 

def generate_data(K,std=0.2,std_xy=5,n=256):
    img = np.array(Image.open("image.png").convert("L").resize((n,n)))
    img = torch.tensor(np.array(img),dtype=torch.float32)
    img = (img - img.min())/(img.max() - img.min())

    # img = np.zeros((8,8))
    # img[2:4,2:3] = 1
    # img[5:7,2:3] = 1
    # img[4:6,5:7] = 1
    # img = torch.tensor(img,dtype=torch.float32)

    thetas = (2 * torch.pi) * torch.rand(size=(K,))
    normal_dist = MultivariateNormal(loc=torch.zeros(2), 
                                     covariance_matrix=std_xy * torch.eye(2))
    beta_dist = Beta(2, 5)
    translations = normal_dist.sample((K,))
    scales = 5*beta_dist.sample((K,))

    transformations = torch.zeros((K, 2, 3))
    transformations[:, 0, 0] = torch.cos(thetas)
    transformations[:, 0, 1] = -torch.sin(thetas)
    transformations[:, 1, 0] = torch.sin(thetas)
    transformations[:, 1, 1] = torch.cos(thetas)
    transformations[:, :, 2] = 2*translations/n

    grid = F.affine_grid(transformations, size=(K, 1, n, n), align_corners=False)
    data = F.grid_sample(img.expand(K, 1, n, n), grid, align_corners=False).squeeze(1)
    data = data * scales.view(K,1,1) + torch.normal(mean=0,std=std,size=(K,n,n))

    data = data.numpy()
    with mrcfile.new('data2.mrc',overwrite=True) as mrc:
        mrc.set_data(data)

if __name__ == "__main__":
    generate_data(800)