import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torchvision.transforms.functional import affine 
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta

import numpy as np
import mrcfile

from PIL import Image 
import functools

from tqdm import tqdm
import matplotlib.pyplot as plt 
    

@torch.no_grad()
def expectation_maximization(device,A, data, std, std_xy, K=10000,N=100):
    n = A.shape[0]    
    batch_size = K // 10
    num_batch = K // batch_size

    normal_dist = MultivariateNormal(loc=torch.zeros(2).to(device), 
                                     covariance_matrix=std_xy * torch.eye(2).to(device))
    uniform_dist = Uniform(0,2*torch.pi)
    beta_dist = Beta(2, 5)

    A_t = torch.zeros_like(A,dtype=torch.float32).to(device)
    data = data
    std = std
    for i in tqdm(range(N)):
        X = data[i]

        integral = torch.zeros_like(A_t,dtype=torch.float32)
        for _ in range(num_batch):
            thetas = uniform_dist.sample((batch_size,)).to(device)
            translations = normal_dist.sample((batch_size,)).to(device)
            scales = beta_dist.sample((batch_size,))

            rotations_logits = uniform_dist.log_prob(thetas).to(device)
            transaltions_logits = normal_dist.log_prob(translations).to(device)

            importance_sampling = rotations_logits.exp() * transaltions_logits.exp()

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

            X_log_density = -torch.norm(X- A_psi,dim=(1,2)) ** 2 / (2*std ** 2)
            X_log_density = X_log_density - torch.logsumexp(X_log_density,0) # Normalize logits

            psi_log_density = -(torch.norm(translations, dim=1) / std_xy) ** 2 * 0.5 - 2*torch.log(2 * torch.pi * std_xy)
  
            w = torch.exp(X_log_density + psi_log_density)
            w /= w.sum()
            #w /= importance_sampling
            tqdm.write(f"W unique: {w.unique()}")
            integral += ((w.view(batch_size, 1, 1) * X_psi)).sum(dim=0)
            #del X_density, psi_density, A_psi, X_psi
        A_t += integral/K
    return A_t


    
if __name__ == "__main__":
    #python -m torch.distributed.launch EM.py
    device = 'cuda'
    std = torch.tensor(1).to(device)    
    std_xy = torch.tensor(1).to(device)

    with mrcfile.open("data1.mrc",permissive=True) as mrc:
        X = mrc.data

    init = np.array(Image.open("init.png").convert('L'))
    n = init.shape[0]
    A = torch.tensor(init,dtype=torch.float32).to(device)
    X = torch.tensor(X[:100]).type(torch.uint8).type(torch.float32).to(device)

    A = (A - A.min())/(A.max() - A.min())
    X = (X - X.min())/(X.max() - X.min())
    for i in range(5):
        print(f"EM iteration {i}/5")
        A = expectation_maximization(device, A, X,std, std_xy)
        A = (A - A.min())/ (A.max() - A.min())
        A_np = A.cpu().numpy()*255
        result = Image.fromarray(A_np.astype(np.uint8),'L')
        result.save(f'experiment/{i}_iteration.png')