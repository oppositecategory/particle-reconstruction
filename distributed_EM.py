import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta

import numpy as np
import mrcfile
from tqdm import tqdm

from PIL import Image 
import functools



class GPUContextManager:
    def __init__(self, rank):
        self.rank = rank 
        self.prev_device = torch.cuda.current_device()

    def __enter__(self):
        torch.cuda.set_device(self.rank)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.set_device(self.prev_device)
        torch.set_default_tensor_type(torch.FloatTensor)

    
def detect_GPU_context(func):
    @functools.wraps(func)
    def wrapper(device, *args, **kwargs):
        with GPUContextManager(device):
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = {k: v.to(device) if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
            return func(device, *args, **kwargs)
    return wrapper

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
    - Fix CUDA out of memory bug 
    - Test and run without scaling then update prior to work with scaling
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
            psi_log_density = -(torch.norm(translations, dim=1) / std_xy) ** 2 * 0.5 - torch.log(4 * torch.pi * std_xy ** 2)

            X_density = torch.exp(X_log_density - X_log_density.max())
            psi_density = torch.exp(psi_log_density - psi_log_density.max())
            
            w = (X_density * psi_density)/C
            A_t += (w.view(-1, 1, 1) * X_psi).sum(dim=0)
            del X_density, psi_density, A_psi, X_psi
    return A_t

def distributed_EM(rank,world_size,A, data,std,std_xy):
    """
    Distributed calculationg of maximizing expectation.

    Args:
        rank: index of GPU to be used
        world_size: number of available GPUs
        A: current estimation of underlying object
        data: the whole image dataset
        std: variance of the noise in the images
        std_xy: the variance of the translations prior
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        A = A.to(device)
        std = std.to(device)
        std_xy = std_xy.to(device)

    dist.barrier()

    dist.broadcast(A, src=0)
    dist.broadcast(std, src=0)
    dist.broadcast(std_xy, src=0)

    local_data = data.chunk(world_size)[rank].to(device)
    local_result = expectation_maximization(device,A, local_data, std,std_xy)

    #global_result = torch.zeros_like(local_result,device=device).to(device)
    dist.all_reduce(local_result, op = dist.ReduceOp.SUM)
    global_result = local_result / data.shape[0]


    dist.barrier()
    dist.destroy_process_group()

    return global_result if rank == 0 else None

if __name__ == "__main__":
    #python -m torch.distributed.launch distributed_EM.py
    world_size = torch.cuda.device_count()
    img = Image.open("image.png").convert("L").resize((256,256))
    img = np.array(img)
    n = img.shape[0]
    std = torch.tensor(0.01).cuda()
    std_xy = torch.tensor(0.2).cuda()

    with mrcfile.open("data.mrc",permissive=True) as mrc:
        X = mrc.data

    A = torch.tensor(np.mean(X[:100],axis=0))
    X = torch.tensor(X[:800])


    mp.spawn(distributed_EM, args=(world_size, A, X, std,std_xy), nprocs=world_size)