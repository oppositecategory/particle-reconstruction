import torch
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta

import numpy as np

import mrcfile

from PIL import Image 
import argparse

def generate_data(K,std,std_xy,n,data_name, filename):
    img = np.array(Image.open(filename).convert("L").resize((n,n)))
    img = torch.tensor(np.array(img),dtype=torch.float32)
    img = (img - img.min())/(img.max() - img.min())

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
    with mrcfile.new(data_name,overwrite=True) as mrc:
        mrc.set_data(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EM algorithm script with init and data inputs.")
    parser.add_argument('--amount', type=int, default=100, help='Size of data')
    parser.add_argument('--image', type=str, default='image.png', help='The name of the original image')
    parser.add_argument('--std', type=int, default=0.01, help='The std for the noise')
    parser.add_argument('--std_xy', type=int, default=5, help='The std for the translations distribution')
    parser.add_argument('--size', type=int, default=256, help='The size of the images')
    parser.add_argument('--data', type=str, default='data.mrc', help='The name of the mrcfile to be created')
    args = parser.parse_args()

    generate_data(args.amount, args.std,args.std_xy, args.size, args.data, args.image)