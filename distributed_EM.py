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

import pickle
from PIL import Image 

from scipy.ndimage import gaussian_filter


@torch.no_grad()
def expectation_maximization(device,A, data, std, std_xy, K=10000,N=100):
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
            scales_logits = beta_dist.log_prob(scales/5.0).to(device) - torch.log(torch.tensor(5.0))

            scales = 5.0*scales

            transformations = torch.zeros((batch_size, 2, 3)).to(device)
            transformations[:, 0, 0] = torch.cos(thetas)
            transformations[:, 0, 1] = -torch.sin(thetas)
            transformations[:, 1, 0] = torch.sin(thetas)
            transformations[:, 1, 1] = torch.cos(thetas)
            transformations[:, :, 2] = 2*translations / n # PyTorch assumes translations are in [-1,1]

            grid = F.affine_grid(transformations, size=(batch_size, 1, n, n), align_corners=False)
            A_psi = F.grid_sample(A.expand(batch_size, 1, n, n), grid, align_corners=False).squeeze(1) * (scales.view(-1,1,1))

            inv_rotations = transformations[...,:2].permute(0,2,1)
            inv_t = -torch.bmm(inv_rotations,transformations[...,2:])
            inv_trans = torch.cat([inv_rotations,inv_t],dim=-1)
            inv_grid = F.affine_grid(inv_trans,size=(batch_size,1,n,n), align_corners=False) 

            X_psi = F.grid_sample(X.expand(batch_size, 1, n, n) , inv_grid, align_corners=False).squeeze(1) / scales.view(-1,1,1)

            X_log_density = -(torch.norm(X - A_psi,dim=(1,2))) ** 2 / (2*std ** 2)
            X_log_density = X_log_density - torch.logsumexp(X_log_density,0)

            psi_log_density = thetas_logits + translations_logits + scales_logits

            w = torch.exp(X_log_density + psi_log_density)
            w /= w.sum()

            Z_batch += (w * (scales**2)).sum()
            std_batch += ( w.view(batch_size,1,1) * scales**2 * (torch.norm(X_psi - A,dim=(1,2))) ** 2 ).sum()
            stdxy_batch += (w.view(batch_size,1,1) * (torch.norm(translations,dim=1)**2)).sum()
            integral += (w.view(batch_size, 1, 1) * X_psi * (scales**2).view(batch_size,1,1)).sum(dim=0)
        A_t += integral/K
        std_t +=  std_batch / K
        stdxy_t +=  stdxy_batch/K   
        Z += Z_batch/K
    std_t /= (n**2)
    std_t /= data.shape[0]
    std_t = std_t ** 0.5
    stdxy_t /= 2*data.shape[0]
    stdxy_t = stdxy_t ** 0.5
    return A_t, std_t, stdxy_t,Z

def distributed_EM(rank,world_size, A, data, std, std_xy, i):
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
    A_local, std_local, stdxy_local,Z_local = expectation_maximization(device,A, local_data, std,std_xy, N = local_data.shape[0])

    dist.all_reduce(A_local, op=dist.ReduceOp.SUM)
    dist.all_reduce(std_local,op=dist.ReduceOp.SUM)
    dist.all_reduce(stdxy_local,op=dist.ReduceOp.SUM)
    dist.all_reduce(Z_local,op=dist.ReduceOp.SUM)
    dist.barrier()

    if rank == 0: 
        std = std_local 
        stdxy = stdxy_local
        n = A.shape[0]
        A = A_local / Z_local
        std /= (n**2)
        std /= data.shape[0]
        std = std ** 0.5
        stdxy /= 2*data.shape[0]
        stdxy = stdxy ** 0.5

        statistics = np.array([std.cpu().numpy(),std_xy.cpu().numpy()])
        with open(r"experiment/variance_estimatinon.obj","wb") as file:
            pickle.dump(statistics,file)

        A_np = A.cpu().numpy()
        with mrcfile.new(f'experiment/{i}_iteration.mrc',overwrite=True) as mrc:
            mrc.set_data(A_np)
    dist.destroy_process_group()



if __name__ == "__main__":
    #python -m torch.distributed.launch distributed_EM.py
    world_size = torch.cuda.device_count()

    std = torch.tensor(0.001)
    std_xy = torch.tensor(20)

    with mrcfile.open("data2.mrc",permissive=True) as mrc:
        data = mrc.data

    # img = np.array(Image.open("image.png").convert("L").resize((256,256)))
    # img = img = (img - img.min())/(img.max() - img.min())
    # init = gaussian_filter(img, sigma=5)      

    with mrcfile.open("experiment/0_iteration.mrc",permissive=True) as mrc:
        init = mrc.data 

    n = init.shape[0]
    A = torch.tensor(init,dtype=torch.float32).cuda()
    data = torch.tensor(data).type(torch.float32)

    for i in range(1,2):
        mp.spawn(distributed_EM, 
             args=(world_size,A, data,std, std_xy,i), 
             nprocs=world_size, 
             join=True)
        A = np.array(Image.open(f'experiment/{i}_iteration.mrc').convert('L'))
        A = torch.tensor(A, dtype=torch.float32).cuda()

        with open(r"experiment/variance_estimatinon.obj","rb") as file:
            deviations = pickle.load(file)
        
        std = torch.tensor(deviations[0],dtype=torch.float32).cuda()
        std_xy = torch.tensor(deviations[1],dtype=torch.float32).cuda()
        print("Updated noise std estimation:", deviations[0])
        print("Updated translation std estimation:", deviations[1])




