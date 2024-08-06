from argparse import Namespace
from typing import Literal, Annotated, Union, Tuple, Any
import numpy as np
from numpy.linalg import norm
from numpy.ma import masked_array
from numpy.typing import ArrayLike
from numba import jit
from scipy.special import gammainc
from ase.atoms import Atoms
from ase.geometry import get_layers
from ase.units import Bohr

threeD = Annotated[ArrayLike, Literal["N", "N", "N"]]
threeAxis = Annotated[ArrayLike, Literal[3]]
threeN = Annotated[ArrayLike, Literal[3, "N"]]


class Grid:

    def __init__(
        self,
        shape,
        cell=None,
        span=None,
        origin=None,
        zref=None,
        z0=None,
        zf=None,
        atoms=None,
        layer_idx=(0, 0, 1),
        layer_tol=0.1,
        params=None,
        positions=None,
        numbers=None,
        verbose=False,
    ):
        if (cell is None) and (span is None):
            raise ValueError("One of cell or span must be defined.")
        elif (span is not None) and (cell is None):
            self.cell = np.array(span)
        else:
            self.cell = np.array(cell)
        self.origin = origin
        self.span = span
        self.shape = shape

        if (zref is None) and (atoms is not None):
            il, _ = get_layers(atoms, layer_idx, tolerance=layer_tol)
            self.zref = np.round(atoms.positions[il == il.max(), 2].mean(), 4)

        if (origin is None) ^ (span is None):
            raise ValueError("Both origin and span must be defined.")
        elif origin is not None:
            if verbose:
                print("Setting origin and span from input.")
            self.origin = origin
            self.span = span
            if zref:
                self.zref = zref
                self.z0 = self.origin[2] - self.zref
                self.zf = self.z0 + self.span[2]
        elif ((zref is None) ^ (z0 is None)) or ((z0 is None) ^ (zf is None)):
            raise ValueError("All of zref, z0, and zf values must be defined.")
        elif (zref is not None) and (z0 is not None) and (zf is not None):
            if verbose:
                print("Setting origin and span from zref, z0, and zf.")
            self.zref = zref
            self.z0 = z0
            self.zf = zf
            self.origin = np.array([0.0, 0.0, self.zref + self.z0])
            self.span = np.array([cell[0], cell[1], self.zf - self.z0])
        else:
            if verbose:
                print("Setting origin and span from cell.")
            self.origin = np.array([0.0, 0.0, 0.0])
            self.span = self.cell
            self.zref = 0.0

        self.x, self.y, self.z, self.dr, self.dR, self.ortho, self.lvec = get_grid(
            self.span, self.shape, mesh=True, origin=self.origin
        )

        self.z -= self.zref

        if atoms:
            self.atoms = atoms
            self.positions = self.atoms.positions
            self.numbers = self.atoms.numbers
        elif (positions is not None) and (numbers is not None):
            assert positions.shape[0] == len(numbers)
            self.positions = positions
            self.numbers = numbers

        self.labels = []

    def set_density(self, label: str, data: threeD, safe=True):
        if safe:
            if label in self.__dict__.keys():
                print(
                    f"Density labeled {label} already exists. Choose a different label or set `safe=False`"
                )
                pass
        assert data.ndim == 3, "Data array is not 3D."
        self.__setattr__(label, data)
        self.labels.append(label)

    def save(self, filename: str):
        save_dict = dict((k, self.__getattribute__(k)) for k in self.labels)
        save_grid(filename, self, **save_dict)

    def save_density(self, label: str, filename: str):
        save_grid(filename, self, **{label: self.__getattribute__(label)})

    def set_zref(self, zref: float):
        zref_ = self.zref
        self.z = (self.z + zref_) - zref
        self.zref = zref

    # def replicate(self, *n, axis=(0, 1)):
    #     assert len(n) == 3, "Replication requires an argument for each axis."
    #     return Grid(self.shape * n, self.cell * n)

    # def __mul__(self, *n):
    #     return self.replicate(*n)

    def __repr__(self):
        return f"Grid({self.cell}, {self.shape}, {self.dr})"

    def __str__(self):
        return self.__repr__()


def save_grid(filename, grid: Grid, **kwargs: threeD):
    if kwargs:
        for v in kwargs.values():
            assert np.equal(v.shape, grid.shape).all(), "Shape mismatch."
    else:
        kwargs = {}
    np.savez(
        open(filename, "wb"),
        **kwargs,
        shape=grid.shape,
        span=grid.span,
        origin=grid.origin,
        ortho=grid.ortho,
        cell=grid.cell,
        numbers=grid.numbers,
        positions=grid.positions,
    )


def get_grid(cell, shape, mesh=None, origin=None):
    if not check_orthogonal(cell):
        print(f"Cell is not orthogonal: {cell}")
        raise ValueError("z-axis is not orthogonal to xy-axes.")
    ortho = 1 if angle(cell[0], cell[1]) == np.pi / 2 else 0
    lvec = norm(cell, axis=1)
    dr = lvec / shape
    dR = cell / shape
    if ortho and not mesh:
        x = np.arange(shape[0]) * dr[0]
        y = np.arange(shape[1]) * dr[1]
    else:
        if not ortho and mesh == False:
            print(
                "Mesh option set to False but cell is not orthogonal, reverting to mesh-coordinates."
            )
        ij = np.indices(shape[:2], dtype=float).T
        x, y = np.dot(ij / shape[:2], cell[:2, :2]).T
    z = np.arange(shape[2]) * dr[2]
    if origin is not None:
        x += origin[0]
        y += origin[1]
        z += origin[2]
    return x, y, z, dr, dR, ortho, lvec


def get_grid_from_params(
    params: Namespace,
    grid: Grid,
    nidx=False,
    nbox=False,
    rpivot=0.0,
    zpad=0.0,
    numbers=None,
    positions=None,
):
#-> Grid | Tuple[Grid, Any] | Tuple[Grid, Any, Any]:
    zref = params.ZREF
    if params.BBOX is None:
        params.BBOX = np.array(
            [[0, 0, params.Z0], [grid.lvec[0], grid.lvec[1], params.ZF]]
        )
    zrange = params.BBOX[1, 2] - params.BBOX[0, 2]
    bbox = params.BBOX.copy()
    if zrange >= rpivot:
        bbox[1, 2] += zpad
    else:
        bbox[1, 2] = bbox[0, 2] + rpivot + zpad
    bbox[:, 2] += zref
    span = np.zeros(grid.cell.shape)
    span[0, 0] = bbox[1, 0] - bbox[0, 0]
    span[1, 1] = bbox[1, 1] - bbox[0, 1]
    span[2, 2] = bbox[1, 2] - bbox[0, 2]
    nspan = np.rint(span / grid.dr).astype(int)
    nbox_ = np.rint(bbox / grid.dr).astype(int)
    shape = nspan.diagonal()
    g = Grid(
        shape,
        cell=grid.cell,
        span=nspan * grid.dr,
        origin=np.rint(bbox[0] / grid.dr) * grid.dr,
        zref=zref,
        numbers=numbers if numbers is not None else None,
        positions=positions if positions is not None else None,
    )
    nidx_ = tuple([slice(i, j) for i, j in zip(nbox_[0], nbox_[1])])
    if nidx:
        if nbox:
            return g, nidx_, nbox_
        return g, nidx_
    elif nbox:
        return g, nbox_
    return g


def check_orthogonal(cell: ArrayLike) -> bool:
    if angle(cell[0], cell[2]) != np.pi / 2 or angle(cell[2], cell[1]) != np.pi / 2:
        return False
    else:
        return True


def angle(
    x: Union[float, ArrayLike], y: Union[float, ArrayLike]
) -> Union[float, ArrayLike]:
    return np.arccos(np.dot(x, y) / (norm(x) * norm(y)))


def read_rhoS(params: Namespace) -> Tuple[threeD, Grid]:
    if params.SAMPLERHO.exists():
        print(f"- Loading rho_sample (rhoS) from {params.SAMPLERHO}.")
        sample = np.load(params.SAMPLERHO)
    else:
        sample_dir = params.SAMPLE / "sample.npz"
        print(f"- Loading sample from {sample_dir}.")
        sample = np.load(sample_dir)
    rhoS = sample["rhoS"]
    gridS = Grid(
        rhoS.shape,
        cell=sample["cell"],
        positions=sample["positions"],
        numbers=sample["numbers"],
    )
    return rhoS, gridS


@jit(nopython=True)
def sph2grid(th, az, r, dr):
    nx = r * np.sin(th) * np.cos(az) / dr[0]
    ny = r * np.sin(th) * np.sin(az) / dr[1]
    nz = r * np.cos(th) / dr[2]
    return [nx, ny, nz]


@jit(nopython=True)
def sph2grid_opt(pi_th, az, r, dr):
    sinth = np.sin(pi_th)
    nx = r * sinth * np.cos(az) / dr[0]
    ny = r * sinth * np.sin(az) / dr[1]
    nz = r * -np.cos(pi_th) / dr[2]
    return [nx, ny, nz]


@jit(nopython=True)
def sph2cart(th, az, r):
    Xsph = r * np.sin(th) * np.cos(az)
    Ysph = r * np.sin(th) * np.sin(az)
    Zsph = r * np.cos(th)
    return Xsph, Ysph, Zsph


def neutralize_charge(
    rho: threeD,
    dr: threeAxis,
    positions: threeN,
    zval: threeAxis,
    rcut=Bohr,
    filter_negative=True,
    neutral=True,
):

    if filter_negative:
        rho = filter_neg(rho)
    # Create zero array to put the neutralizing charges
    nrho = np.zeros(rho.shape)
    # Loop around positions and charge
    for pos, c in zip(positions, zval):
        # Add and substract rcut to pos
        imin = (np.sign(pos - rcut) * np.ceil(abs(pos - rcut) / dr)).astype(int)
        imax = (np.sign(pos + rcut) * np.ceil(abs(pos + rcut) / dr)).astype(int) + 1
        # Size of grid inside rcut
        nroll = imax - imin

        # Get rcut grid positions
        X, Y, Z = np.meshgrid(
            *[dr[i] * np.arange(imin[i], imax[i]) for i in range(3)], indexing="ij"
        )
        # Get distances (radii) from atom to all points of rcut grid
        ## rcut grid positions centered on atom
        D = norm([X - pos[0], Y - pos[1], Z - pos[2]], axis=0)
        # Mask all values outside the rcut sphere
        mD = masked_array(D, mask=[D > rcut])
        # Get all the unique radii values inside the rcut sphere
        ## ir is the index that maps each point of mD
        ## to a value in the radii array r
        r, ir = np.unique(mD, return_inverse=True)
        # Set parameters for the generalized gaussian cumulative distribution
        ggd_params = dict(loc=0.5 * rcut, shape=4, scale=0.2)
        # Get the charge factor multiplied by the sph harm with l=0
        ## Factor 0.23... is unknown and seems to be in atomic units
        c = (c / (0.23633650699 * Bohr**3)) * 0.5 * np.sqrt(1 / np.pi)
        # Call the normalized GGD function with every unique r
        ## Directly multiply by the charge factor and sph harmonic
        res = c * norm_cumulative_ggd(r, **ggd_params)
        # Displace (roll) the neutral rho array to match the rcut grid at (0,0,0)
        nrho = np.roll(nrho, -imin, (0, 1, 2))
        # Slicers to put neutral charge in place
        ix, iy, iz = slice(nroll[0]), slice(nroll[1]), slice(nroll[2])
        # Add neutral rho mapping index ir (which is flat, reshape to nroll)
        ## Fill value for all grid points outside rcut
        nrho[ix, iy, iz] = res[ir.reshape(nroll)].filled(fill_value=0)
        # Unroll nrho to original position
        nrho = np.roll(nrho, imin, (0, 1, 2))

    if neutral:
        # Rescale nrho to the exact number of electrons in rho
        nrho *= rho.sum() / nrho.sum()
        # Finally, substract nrho from original rho
        nrho = rho - nrho
    return nrho


def filter_neg(rho, prec=0):
    """Takes a density, replaces all negative values with 0,
    and renormalizes the density accordingly.
    If prec < 0, also removes values below 10^prec.
    Required for exponentiation to non-integer powers in
    density overlap calculation."""
    tot = rho.sum()
    prec = int(prec)
    if prec < 0:
        print("Setting 0 for values < 1x10^{}".format(int(prec)))
        rho[rho < 10**prec] = 0
    else:
        rho[rho < 0] = 0
    rho *= tot / rho.sum()
    return rho


def norm_cumulative_ggd(r, loc, shape, scale):
    s = 1 / gammainc(1 / shape, (loc / scale) ** shape)
    res = 0.5 - s * np.sign(r - loc) / 2 * gammainc(
        1 / shape, ((r - loc) / scale) ** shape
    )
    return res