import multiprocessing
import numpy as np
import os
import torch
from mpi4py import MPI
from curiosity_rl.utils.mpi_tools import broadcast, mpi_avg, num_procs, proc_id


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    #print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    #print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)


def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):
        self._sum = torch.Tensor(
            dtype=torch.float64,
            shape=shape)
        self._sumsq = torch.Tensor(
            dtype=torch.float64,
            shape=shape)
        self._count = torch.Tensor(
            dtype=torch.float64,
            shape=())
        self.shape = shape

        self.mean = (self._sum / self._count).float()
        self.std = torch.sqrt(torch.maximum(
            (self._sumsq / self._count).float() - torch.square(self.mean), 1e-2))

    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n*2+1, 'float64')
        addvec = np.concatenate(
            [x.sum(axis=0).ravel(), 
             np.square(x).sum(axis=0).ravel(), 
             np.array([len(x)], 
             dtype='float64')]
        )
        MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        self.incfiltparams(
            totalvec[0:n].reshape(self.shape), 
            totalvec[n:2*n].reshape(self.shape), 
            totalvec[2*n]
        )

    def incfiltparams(self, newsum, newsumsq, newcount):
        self._sum += torch.Tensor(newsum)
        self._sumsq += torch.Tensor(newsumsq)
        self._count += torch.Tensor(newcount)
