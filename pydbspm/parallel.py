import numpy as np


def dftd3_MPI_grid(cell, shape, nidx, posS, posT, nS, nT, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        ijk = np.indices(shape, dtype=float).T
        x, y, z = np.dot(ijk / shape, cell).T
        shape = x[nidx].shape
        gx, gy, gz = x[nidx].flatten(), y[nidx].flatten(), z[nidx].flatten()
        res_size = gx.size
        # Prepare parallelization
        div, res = divmod(res_size, size)
        ## Number of points for each core
        count = np.array([div + 1 if p < res else div for p in range(size)])
        ## Starting index for each core
        displ = np.array([sum(count[:p]) for p in range(size)])
    else:
        cell = None
        posS = None
        posT = None
        numbersS = None
        numbersT = None
        gx = None
        gy = None
        gz = None
        res_size = None
        count = np.zeros(size, dtype=int)
        displ = None

    # Scatter gridpoints for each core
    comm.Bcast(count, root=0)
    par_x, par_y, par_z = np.zeros([3, count[rank]])
    comm.Scatterv([gx, count, displ, MPI.DOUBLE], par_x, root=0)
    comm.Scatterv([gy, count, displ, MPI.DOUBLE], par_y, root=0)
    comm.Scatterv([gz, count, displ, MPI.DOUBLE], par_z, root=0)
    comm.Barrier()

    # Broadcast necessary data
    cell = comm.bcast(cell, root=0)
    posS = comm.bcast(posS, root=0)
    posT = comm.bcast(posT, root=0)
    numbersS = comm.bcast(numbersS, root=0)
    numbersT = comm.bcast(numbersT, root=0)
    res_size = comm.bcast(res_size, root=0)
    comm.Barrier()

    if rank == 0:
        print("\n> Performing DFT-D3 calculation.")
    res_array = get_dftd3_api_grid(
        cell, posS, posT, numbersS, numbersT, par_x, par_y, par_z
    )
    comm.Barrier()
    if rank == 0:
        print("< done.\n")
        gatherE = np.zeros(res_size)
    else:
        gatherE = None

    comm.Gatherv(res_array, [gatherE, count, displ, MPI.DOUBLE], root=0)
    if rank == 0:
        print("< data gathered.")
        gatherE = gatherE.reshape(calc_grid.shape)
