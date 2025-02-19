import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist


from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta

import numpy as np
import mrcfile
from tqdm import tqdm

from PIL import Image 



@torch.no_grad()
def expectation_maximization(device,A, data, std, std_xy, K=10000,N=100):
    """
    TODO:
    - Add scaling aswell including it into the prior
    - Check importance sampling
    - Check into ways to reduce sampling in later iterations
    """
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

            transformations = torch.zeros((batch_size, 2, 3)).to(device)
            transformations[:, 0, 0] = torch.cos(thetas)
            transformations[:, 0, 1] = -torch.sin(thetas)
            transformations[:, 1, 0] = torch.sin(thetas)
            transformations[:, 1, 1] = torch.cos(thetas)
            transformations[:, :, 2] = 2*translations / n

            grid = F.affine_grid(transformations, size=(batch_size, 1, n, n), align_corners=False)
            #A_psi = F.grid_sample(A.expand(K, 1, n, n), grid, align_corners=False).squeeze(1) * scales.view(-1, 1, 1)
            A_psi = F.grid_sample(A.expand(batch_size, 1, n, n), grid, align_corners=False).squeeze(1)

            inv_rotations = transformations[...,:2].permute(0,2,1)
            inv_t = -torch.bmm(inv_rotations,transformations[...,2:])
            inv_trans = torch.cat([inv_rotations,inv_t],dim=-1)
            inv_grid = F.affine_grid(inv_trans,size=(batch_size,1,n,n), align_corners=False)

            #X_psi = F.grid_sample(X.expand(K, 1, n, n), inv_grid, align_corners=False).squeeze(1) * scales.view(-1, 1, 1)
            X_psi = F.grid_sample(X.expand(batch_size, 1, n, n), inv_grid, align_corners=False).squeeze(1) 

            X_log_density = -torch.norm(X - A_psi,dim=(1,2)) ** 2 / (2*std ** 2)
            X_log_density = X_log_density - torch.logsumexp(X_log_density,0) # Normalize logits

            psi_log_density = -(torch.norm(translations, dim=1) / std_xy) ** 2 * 0.5 - 2*torch.log(2 * torch.pi * std_xy)
  
            w = torch.exp(X_log_density + psi_log_density)
            w /= w.sum()
            #w /= importance_sampling
            #tqdm.write(f"W unique: {w.unique()}")
            integral += ((w.view(batch_size, 1, 1) * X_psi)).sum(dim=0)
            #del X_density, psi_density, A_psi, X_psi
        A_t += integral/K
    return A_t


def distributed_EM(rank,world_size, A, data, std, std_xy,i):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank ==0:
        A = A.to(device)
    else:
        A = torch.zeros_like(A, device=rank)

    std = std.to(device)
    std_xy = std_xy.to(device)
    
    dist.broadcast(A, src=0)
    dist.barrier()

    local_data = data.chunk(world_size)[rank].to(device)
    local_result = expectation_maximization(device,A, local_data, std,std_xy)

    dist.all_reduce(local_result, op=dist.ReduceOp.SUM)
    dist.barrier()

    if rank == 0: 
        A = local_result / data.shape[0]
        A = (A - A.min())/ (A.max() - A.min())
        A_np = A.cpu().numpy()*255
        result = Image.fromarray(A_np.astype(np.uint8),'L')
        result.save(f'experiment/{i}_iteration.png')
    dist.destroy_process_group()



if __name__ == "__main__":
    #python -m torch.distributed.launch EM.py
    world_size = torch.cuda.device_count()

    std = torch.tensor(2)
    std_xy = torch.tensor(5)

    with mrcfile.open("data3.mrc",permissive=True) as mrc:
        data = mrc.data

    init = np.array(Image.open('init.png').convert('L'))
    #init = np.array(Image.open('experiment/3_iteration.png').convert('L'))
    #init = np.mean(data,axis=0)
    n = init.shape[0]
    A = torch.tensor(init,dtype=torch.float32).cuda()
    data = torch.tensor(data).type(torch.float32)

    A = (A - A.min())/(A.max() - A.min())
    data = (data - data.min())/(data.max() - data.min())

    mp.spawn(distributed_EM, 
             args=(world_size,A, data,std, std_xy,4), 
             nprocs=world_size, 
             join=True)

    # for i in range(3):
    #     mp.spawn(distributed_EM, 
    #          args=(world_size,A, data,std, std_xy,4), 
    #          nprocs=world_size, 
    #          join=True)
    #     A = np.array(Image.open(f'experiment/{i}_iteration.png').convert('L'))




