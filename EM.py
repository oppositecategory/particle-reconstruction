import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torchvision.transforms.functional import affine 
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta

import numpy as np
import mrcfile

from PIL import Image 
import functools

from tqdm import tqdm
import matplotlib.pyplot as plt 
    
@torch.no_grad()
def compute_posterior_normalization(device,A,X, std,std_xy,K=10000):
    n = A.shape[0]
    batch_size = K // 10
    num_batch = K // batch_size

    normal_dist = MultivariateNormal(loc=torch.zeros(2).to(device), 
                                     covariance_matrix=std_xy * torch.eye(2).to(device))
    beta_dist = Beta(2, 5)
    
    C = torch.tensor(0.0,device=device,dtype=torch.float32)
    
    for _ in range(num_batch):
        # Calculate the integral in batches to prevent CUDA OOM 
        thetas = (2 * torch.pi) * torch.rand(size=(batch_size,)).to(device)
        translations = normal_dist.sample((batch_size,)).to(device)
        scales = beta_dist.sample((batch_size,))
    
        affine_matrices = torch.zeros((batch_size, 2, 3)).to(device)
        affine_matrices[:, 0, 0] = torch.cos(thetas)
        affine_matrices[:, 0, 1] = -torch.sin(thetas)
        affine_matrices[:, 1, 0] = torch.sin(thetas)
        affine_matrices[:, 1, 1] = torch.cos(thetas)
        affine_matrices[:, :, 2] = translations

        grid = F.affine_grid(affine_matrices, size=(batch_size, 1, n, n), align_corners=False)
        #A_psi = F.grid_sample(A.expand(K, 1, n, n), grid, align_corners=False).squeeze(1) * scales.view(-1, 1, 1)
        A_psi = F.grid_sample(A.expand(batch_size, 1, n, n), grid, align_corners=False).squeeze(1)

        X_logits = -torch.norm(X - A_psi, dim=(1, 2))**2 / (2 * std**2) - (n**2) * torch.log((2 * torch.pi) ** 0.5 * std)
        psi_logits= -(torch.norm(translations, dim=1) / std_xy) ** 2 * 0.5 - torch.log(4 * torch.pi * std_xy ** 2)

        X_density = torch.exp(X_logits - X_logits.max())
        psi_density = torch.exp(psi_logits - psi_logits.max())
        C += torch.sum(X_density*psi_density)
    return C


@torch.no_grad()
def expectation_maximization(device,A, data, std, std_xy, K=10000):
    """
    TODO:
    - Go through the equations to make sure they are indeed correct 
    - Experiment with a simple distributed script to make data be 8x bigger
    """
    n = A.shape[0]    
    batch_size = K // 10
    num_batch = K // batch_size

    normal_dist = MultivariateNormal(loc=torch.zeros(2).to(device), 
                                     covariance_matrix=std_xy * torch.eye(2).to(device))
    beta_dist = Beta(2, 5)

    A_t = torch.zeros_like(A).to(device)

    for i in tqdm(range(data.shape[0])):
        X = data[i]

        C = compute_posterior_normalization(device,A,X,std,std_xy)
        tqdm.write(f"C value: {C.item()}")
        for _ in range(num_batch):
            thetas = (2 * torch.pi) * torch.rand(size=(batch_size,)).to(device)
            translations = normal_dist.sample((batch_size,)).to(device)
            scales = beta_dist.sample((batch_size,))

            transformations = torch.zeros((batch_size, 2, 3)).to(device)
            transformations[:, 0, 0] = torch.cos(thetas)
            transformations[:, 0, 1] = -torch.sin(thetas)
            transformations[:, 1, 0] = torch.sin(thetas)
            transformations[:, 1, 1] = torch.cos(thetas)
            transformations[:, :, 2] = translations

            grid = F.affine_grid(transformations, size=(batch_size, 1, n, n), align_corners=False)
            #A_psi = F.grid_sample(A.expand(K, 1, n, n), grid, align_corners=False).squeeze(1) * scales.view(-1, 1, 1)
            A_psi = F.grid_sample(A.expand(batch_size, 1, n, n), grid, align_corners=False).squeeze(1)

            inv_rotations = transformations[...,:2].permute(0,2,1)
            inv_t = -torch.bmm(inv_rotations,transformations[...,2:])
            inv_trans = torch.cat([inv_rotations,inv_t],dim=-1)
            inv_grid = F.affine_grid(inv_trans,size=(batch_size,1,n,n), align_corners=False)

            #X_psi = F.grid_sample(X.expand(K, 1, n, n), inv_grid, align_corners=False).squeeze(1) * scales.view(-1, 1, 1)
            X_psi = F.grid_sample(X.expand(batch_size, 1, n, n), inv_grid, align_corners=False).squeeze(1) 

            X_log_density = -torch.norm(X - A_psi, dim=(1, 2))**2 / (2 * std**2) - (n**2) * torch.log((2 * torch.pi) ** 0.5 * std)
            psi_log_density = -(torch.norm(translations, dim=1) / std_xy) ** 2 * 0.5 - torch.log((2*torch.pi * std_xy) ** 2)
            
            # Numerical stability trick to prevent underflow
            X_density = torch.exp(X_log_density - X_log_density.max())
            psi_density = torch.exp(psi_log_density - psi_log_density.max())
            
            w = (X_density * psi_density)/C
            A_t += (w.view(-1, 1, 1) * X_psi).sum(dim=0)
            del X_density, psi_density, A_psi, X_psi
    return A_t/data.shape[0]


    
if __name__ == "__main__":
    #python -m torch.distributed.launch EM.py
    device = 'cuda'
    img = Image.open("image.png").convert("L").resize((256,256))
    img = np.array(img)
    n = img.shape[0]
    std = torch.tensor(0.01).to(device)
    std_xy = torch.tensor(0.2).to(device)

    with mrcfile.open("data.mrc",permissive=True) as mrc:
        X = mrc.data

    A = torch.tensor(np.mean(X[:100],axis=0)).to(device)
    #A = np.array(Image.open('experiment/4_iteration.png').convert('L'))
    #A = torch.tensor(A).to(device).type(torch.float32)
    X = torch.tensor(X[:200]).to(device)

    for i in range(5):
        print(f"EM iteration {i}/5")
        A = expectation_maximization(device, A, X,std, std_xy)
        result = A.cpu().numpy().astype(np.uint8)
        result = Image.fromarray(result,'L')
        result.save(f'experiment/{i}_iteration.png')