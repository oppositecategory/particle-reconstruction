import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta

import numpy as np
import mrcfile

from PIL import Image 

from tqdm import tqdm
    
@torch.no_grad()
def expectation_maximization(device,A, data, std, std_xy, K=10000,N=100):
    """
    TODO: - Fix blurryness in final result.
    """
    n = A.shape[0]    
    batch_size = K // 10
    num_batch = K // batch_size

    normal_dist = MultivariateNormal(loc=torch.zeros(2).to(device), 
                                     covariance_matrix=std_xy * torch.eye(2).to(device))
    uniform_dist = Uniform(0,2*torch.pi)
    beta_dist = Beta(torch.tensor(2.0).to(device), torch.tensor(5.0).to(device))

    A_t = torch.zeros_like(A,dtype=torch.float32).to(device)
    std_t = torch.tensor(0,dtype=torch.float32).to(device)
    stdxy_t = torch.tensor(0,dtype=torch.float32).to(device)
    Z = torch.tensor(0, dtype=torch.float32).to(device)
    for i in tqdm(range(N)):
        X = data[i]

        integral = torch.zeros_like(A_t,dtype=torch.float32)
        std_batch = torch.tensor(0,dtype=torch.float32).to(device) 
        stdxy_batch = torch.tensor(0,dtype=torch.float32).to(device)
        Z_batch = torch.tensor(0, dtype=torch.float32).to(device)
        for _ in range(num_batch):
            thetas = uniform_dist.sample((batch_size,)).to(device)
            translations = normal_dist.sample((batch_size,)).to(device)
            scales = beta_dist.sample((batch_size,)).to(device)

            thetas_logits = uniform_dist.log_prob(thetas).to(device)
            translations_logits = normal_dist.log_prob(translations).to(device)
            scales_logits = beta_dist.log_prob(scales).to(device)

            scales = 5.0*scales

            transformations = torch.zeros((batch_size, 2, 3)).to(device)
            transformations[:, 0, 0] = scales * torch.cos(thetas)
            transformations[:, 0, 1] = -torch.sin(thetas)
            transformations[:, 1, 0] = torch.sin(thetas)
            transformations[:, 1, 1] = torch.cos(thetas)
            transformations[:, :, 2] = 2*translations / n # PyTorch assumes translations are in [-1,1]

            grid = F.affine_grid(transformations, size=(batch_size, 1, n, n), align_corners=False)
            A_psi = F.grid_sample(A.expand(batch_size, 1, n, n), grid, align_corners=False).squeeze(1)*scales.view(batch_size, 1, 1)

            inv_rotations = transformations[...,:2].permute(0,2,1)
            inv_t = -torch.bmm(inv_rotations,transformations[...,2:])
            inv_trans = torch.cat([inv_rotations,inv_t],dim=-1)
            inv_grid = F.affine_grid(inv_trans,size=(batch_size,1,n,n), align_corners=False)

            X_psi = F.grid_sample(X.expand(batch_size, 1, n, n), inv_grid, align_corners=False).squeeze(1) / scales.view(batch_size, 1, 1)

            X_log_density = -torch.norm(X - A_psi,dim=(1,2)) ** 2 / (2*std ** 2)
            X_log_density = X_log_density - torch.logsumexp(X_log_density,0) # Stable normalization

            psi_log_density = thetas_logits + translations_logits + scales_logits
  
            w = torch.exp(X_log_density + psi_log_density)
            w /= w.sum()

            Z_batch += (w * scales**2).sum()
            std_batch += ( w.view(batch_size,1,1) * torch.norm(X - A_psi,dim=(1,2)) ** 2 ).sum()
            stdxy_batch += (w.view(batch_size,1,1) * (torch.norm(translations,dim=1)**2)).sum()
            integral += (w.view(batch_size, 1, 1) * X_psi * (scales**2).view(batch_size,1,1)).sum(dim=0)
        A_t += integral/K
        std_t +=  std_batch / K
        stdxy_t +=  stdxy_batch/K   
        Z += Z_batch/K
    std_t /= ((n**2) * data.shape[0])
    std_t = std_t ** 0.5
    stdxy_t /= 2*data.shape[0]
    stdxy_t = stdxy_t ** 0.5
    print(f"Updated noise variance estimation: {std_t}")
    print(f"Updated translations variance estimation: {stdxy_t}")
    return A_t/Z, std_t, stdxy_t


    
if __name__ == "__main__":
    device = 'cuda'
    std = torch.tensor(1).to(device)    
    std_xy = torch.tensor(1).to(device)

    with mrcfile.open("data1.mrc",permissive=True) as mrc:
        X = mrc.data

    init = np.array(Image.open("init.png").convert('L'))
    n = init.shape[0]
    A = torch.tensor(init,dtype=torch.float32).to(device)
    X = torch.tensor(X).type(torch.float32).to(device)

    A = (A - A.min())/(A.max() - A.min())
    X = (X - X.min())/(X.max() - X.min())
    for i in range(5):
        print(f"EM iteration {i}/5")
        A, std, std_xy = expectation_maximization(device, A, X,std, std_xy)
        A = (A - A.min())/ (A.max() - A.min())
        A_np = A.cpu().numpy()*255
        result = Image.fromarray(A_np.astype(np.uint8),'L')
        result.save(f'experiment/{i}_iteration.png')