import numpy as np
from mpi4py import MPI
import sys 
import time


def ddot(x, y, comm):
    local_ddot = np.inner(x, y)
    global_ddot = comm.allreduce(local_ddot, MPI.SUM)
    return global_ddot


def CG(A, b, x, eps, comm):
    extra = (0 if comm.Get_rank() < A.shape[1] % comm.Get_size() else 1)
    row = A.shape[0] + extra
    r = np.zeros(row)
    r[:b.size] = b
    matvec(A, x, r, comm, -1)
    p = np.copy(r)
    r0 = ddot(r, r, comm) ** 0.5
    Ap = np.zeros(row)
    for i in range(1, A.shape[1]):
        if (i % 15 == 0):
            matvec(A, x, r, comm, -1)
            r += b
        alpha = ddot(r, r, comm)
        beta = 1 / alpha
        matvec(A, p, Ap, comm)
        alpha /= ddot(Ap, p, comm)
        r -= alpha * Ap
        x += alpha * p

        resid = ddot(r, r, comm)
        if ((resid ** 0.5) / r0 < eps):
            if (comm.Get_rank() == 0):
                print(f"Падение невязки: {(resid ** 0.5)/ r0}")
                print(f"{i} итераций")
            return
        beta *= resid
        p += (beta - 1) * p + r

    print("FULL ITERATIONS")
    print(f"Невязка: {resid}")


def matvec(A, x_recv, res, comm, alpha=1):
    col = A.shape[1]
    rank = comm.Get_rank()
    size = comm.Get_size()
    offset = (col // size) * rank + (rank if rank < col % size else col % size)
    dest = (rank + size - 1) % size
    source = (rank + 1) % size
    x_send = np.copy(x_recv)
    
    res[:] = 0
    for k in range(size):
        col_block = col // size + (1 if ((k + rank) % size < col % size) else 0)
        req_r = comm.Irecv(x_recv, source, 21)
        req_s = comm.Isend(x_send, dest, 21) 
        res[:A.shape[0]] += alpha * (A[:, offset : offset + col_block] @ x_send[:col_block])
        offset = (offset + col_block) % col
        req_r.wait()
        req_s.wait()
        np.copyto(x_send, x_recv)
        comm.Barrier()


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = sys.argv
    n = int(args[1])

    m = n // size + (1 if rank < n % size else 0)
    extra = (0 if rank < n % size else 1) # для выравнивания памяти
    start = rank * (n // size) + min(n % size, rank)
    A = np.genfromtxt("A.txt", skip_header = start, skip_footer = n - start - m).reshape(m, -1)
    b = np.genfromtxt("b.txt", skip_header = start, skip_footer = n - start - m)
    x_0 = np.random.rand(m + extra)
    res = np.random.rand(m)
    CG(A, b, x_0, 1e-8, comm)

