import numpy as np
from scipy.ndimage import gaussian_filter
from pandas import DataFrame
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase import Atoms
import matplotlib.pyplot as plt
import copy

from pydbspm.input_parser import default_params as params
from pydbspm.grid import Grid
from pydbspm.tools import get_fs


class SPMGrid:

    def __init__(self,
                 data,
                 label,
                 grid=None,
                 zref=None,
                 atoms=None,
                 grid_from_atoms=False,
                 rpivot=params['RPIVOT'],
                 tip=False,
                 tip_axis=0,
                 **kwargs):
        self.label = label
        if type(data) is str:
            file_ext = data.split('.')[-1]
            if file_ext == 'npz':
                data = np.load(data)
                self.data = data[label]
                span = data['span']
                origin = data['origin']
                cell = data['cell']
                atoms = Atoms(numbers=data['numbers'],
                              positions=data['positions'],
                              cell=cell)
                if 'zref' in data.keys():
                    zref = data['zref']
                else:
                    zref = None
                grid = Grid(self.data.shape,
                            span=span,
                            origin=origin,
                            cell=cell,
                            atoms=atoms,
                            zref=zref)
            elif file_ext == 'cube':
                if atoms or grid:
                    'Reading cube file, embedded data overwrites supplied atoms and grid objects.'
                from ase.io.cube import read_cube
                d = read_cube(data)
                data = d['data']
                atoms = d['atoms']
                span = d['span']
                origin = d['origin']
                grid = Grid(data.shape,
                            span=span,
                            origin=origin,
                            cell=atoms.cell[:3],
                            atoms=atoms,
                            zref=zref)
            # elif file_ext == 'xsf':
            #     if atoms or grid:
            #         'Reading xsf file, embedded data overwrites supplied atoms and grid objects.'
            #     from ase.io.xsf import read_xsf
            #     out = read_xsf(data, read_data=True)
            #     data = out[0]
            #     atoms = out[-1]
            #     grid = get_grid_from_atoms(atoms, data.shape)
        try:
            #if type(data) == np.lib.npyio.NpzFile:
            self.data = data[label]
        except:
            self.data = data
        if not grid:
            if grid_from_atoms:
                grid = Grid(self.data.shape,
                            cell=atoms.cell[:],
                            atoms=atoms,
                            zref=zref)
            else:
                grid = data

        self.zref = grid.zref
        self.x = grid.x
        self.y = grid.y
        self.z = grid.z
        self.x_ = np.linalg.norm([self.x[:, 0], self.y[:, 0]], axis=0)
        self.y_ = np.linalg.norm([self.x[0, :], self.y[0, :]], axis=0)
        self.grid = [self.x_, self.y_, self.z]
        self.shape = self.data.shape
        self.dr = grid.dr
        if label == 'tip' or tip:
            self.set_tip(data['tip'] if tip else self.data,
                         rpivot=rpivot,
                         axis=tip_axis)
            if label == 'tip':
                self.shape = self.data.shape
        elif self.shape != (len(self.x_), len(self.y_), len(self.z)):
            raise Exception(f'Data shape {self.shape} is different to grid \
coordinates size {(*self.x.shape, len(self.z))}')
        if atoms: self.atoms = atoms
        self.kwargs = kwargs if kwargs else None
        if kwargs: self.__dict__.update(kwargs)

    def loc(self, *lims, axis=2, inplace=False, index=False):
        N = self.data.ndim
        axes = np.core.numeric.normalize_axis_tuple(axis, N)
        idx = [slice(None)] * N
        if len(axes) != len(lims):
            raise Exception(f'Number of axes to locate ({len(axes)}) is \
different from number of limits specified ({len(lims)}).')
        for a, lim in zip(axes, lims):
            n0 = np.abs(np.arange(0, self.shape[0]) * self.dr[0] -
                        lim[0]).argmin()
            nf = np.abs(np.arange(0, self.shape[1]) * self.dr[1] -
                        lim[1]).argmin()
            idx[a] = slice(n0, nf + 1)
        idx = tuple(idx)
        if inplace:
            self.data = self.data[idx]
            self.x = self.x[idx[:2]]
            self.y = self.y[idx[:2]]
            self.z = self.z[idx[2]]
        elif index:
            return idx, self.data[idx]
        else:
            return self.data[idx]

    def slice(self, *n, axis=2, info=False):
        N = self.data.ndim
        axes = np.core.numeric.normalize_axis_tuple(axis, N)
        idx = [slice(None)] * N
        loc = []
        if len(axes) != len(n):
            raise Exception(f'Number of axes to slice ({len(axes)}) is \
different from number of locations specified ({len(n)}).')
        for a, lim in zip(axes, n):
            n = np.abs(self.grid[a] - lim).argmin()
            idx[a] = slice(n, n + 1)
            loc += [self.grid[a][n]]
        if info:
            print(', '.join([f'Axis {a}: {x:.4f}' for a, x in zip(axes, loc)]))
        idx = tuple(idx)
        return self.data[idx].squeeze()

    def plot(self,
             z,
             xlim=None,
             ylim=None,
             ax=None,
             levels=128,
             info=False,
             units='',
             conv=1,
             scalebar=None,
             scale_lw=2,
             color='black',
             text=False,
             textprec=0,
             text_sci=False,
             fontsize=18,
             textbox=False,
             boxstyle='round',
             boxcolor='w',
             boxalpha=0.5,
             boxpad=0.1,
             sigma=None,
             show_atoms=False,
             atoms_size=100,
             center_zero=False,
             **kwargs):
        if not ax:
            ax = plt.gca()
            ax.set_aspect('equal')
        nz = np.abs(self.z - z).argmin()
        if (z - self.z.max()) > self.dr[2] / 2:
            print(f'Provided z ({z}) is more than dz/2 higher than the maximum\
 z in this data set ({self.z.max():.4f}). Plotting the maximum z-distance...')
        lims = []
        axis = []
        if xlim:
            lims += [xlim]
            axis += [0]
        if ylim:
            lims += [ylim]
            axis += [1]
        if len(lims) > 0:
            idx, data = self.loc(*lims, axis=axis, index=True)
            data = data[:, :, nz] * conv
        else:
            idx = (slice(None), slice(None))
            data = self.data[:, :, nz] * conv
        if sigma:
            data = gaussian_filter(data, sigma)
        if center_zero:
            vmax = np.abs(np.max([data.max(), data.min()]))
            vmin = -vmax
            ax.contourf(self.x[idx[:2]],
                        self.y[idx[:2]],
                        data,
                        vmin=vmin,
                        vmax=vmax,
                        levels=levels,
                        **kwargs)
        else:
            ax.contourf(self.x[idx[:2]],
                        self.y[idx[:2]],
                        data,
                        levels=levels,
                        **kwargs)
        if info:
            print(f'z={self.z[nz]:.4f}, min={data.min():.4e}{units}, max=\
{data.max():.4e}{units}, range={data.max()-data.min():.4e}{units}')
        if scalebar:
            if not xlim: xlim = (self.x[0], self.x[-1])
            if not ylim: ylim = (self.y[0], self.y[-1])
            ax.hlines(y=(ylim[1] - 1),
                      xmin=xlim[0] + 1,
                      xmax=xlim[0] + 1 + scalebar,
                      color=color,
                      linewidth=scale_lw)
        if text:
            if text_sci:
                textstr = f'{data.min():.{textprec}E}...\
{data.max():.{textprec}E} {units}'

            else:
                textstr = f'{data.min():.{textprec}f}...\
{data.max():.{textprec}f} {units}'

            if textbox:
                boxparams = dict(boxstyle=boxstyle,
                                 facecolor=boxcolor,
                                 alpha=boxalpha,
                                 pad=boxpad)
            ax.text(0.05,
                    0.05,
                    textstr,
                    transform=ax.transAxes,
                    color=color,
                    fontsize=fontsize,
                    bbox=boxparams if textbox else None)
        if show_atoms:
            idx = ((xlim[0] < self.atoms.positions[:, 0]) &
                   (self.atoms.positions[:, 0] < xlim[1])) if xlim else True
            idy = ((ylim[0] < self.atoms.positions[:, 1]) &
                   (self.atoms.positions[:, 1] < ylim[1])) if ylim else True
            s = np.logical_and(idx, idy) if xlim or ylim else slice(None)
            ax.scatter(self.atoms.positions[s, 0],
                       self.atoms.positions[s, 1],
                       s=covalent_radii[self.atoms.numbers[s]] * atoms_size,
                       color=jmol_colors[self.atoms.numbers[s]],
                       facecolors='none')
        return ax

    def to_sites(self, sites, labels=None):
        if type(sites) is dict:
            nx, ny = np.rint(np.array(list(sites.values()) /
                                      self.dr[:2])).astype(int).T
            if not labels:
                labels = list(sites.keys())
        else:
            nx, ny = np.rint(np.array(sites / self.dr[:2])).astype(int).T
        return DataFrame(self.data[nx, ny, :].T,
                         index=np.round(self.z, 4),
                         columns=labels)

    def get_fs(self, amp, k0=1800, f0=30300, conv=16.0217656, verbose=True):
        n = np.rint(amp * 2 / self.dr[2]).astype(int)
        amp = n * self.dr[2]
        if verbose:
            print(f'Selected semi-amplitude as multiple of dz: {n/2:.1f}*dz \
= {amp/2:.4f} ({amp:.4f} peak-to-peak)')
        new = copy.deepcopy(self)
        new.data = get_fs(self.data,
                          dz=self.dr[2],
                          n=n,
                          k0=k0,
                          f0=f0,
                          conv=conv)
        new.z = self.z[:new.data.shape[2]]
        new.grid[2] = new.z
        new.shape = new.data.shape
        new.amp = amp
        new.k0 = k0
        new.f0 = f0
        new.label = 'fs'
        return new

    def set_tip(self, tip, rpivot=params['RPIVOT'], axis=0):
        if len(tip.shape) != 4:
            raise ValueError(
                'Tip position array must have the shape (3,nx,ny,nz),\
    or (nx,ny,nz,3) with axis=-1')
        if axis not in [-1, 0]:
            raise ValueError('Axis tells the function in which axis the three \
different cartesians are strored, it can only be 0 (first) or -1 (last axis)')
        tip_shape = tip.shape[1:] if axis == 0 else tip.shape[:-1]
        if self.shape != tip_shape:
            raise ValueError(f'Data shape {self.shape} is different than tip \
position array shape {tip_shape}')
        if axis == 0:
            tip = np.moveaxis(tip, 0, -1)
        tip[..., 2] += rpivot
        self.tip = tip

    def interpolate_data(self, data):
        import tricubic
        interp = tricubic.tricubic(list(data.data), list(data.shape))
        x_min, x_max = data.x_.min(), data.x_.max() + data.dr[0]
        y_min, y_max = data.y_.min(), data.y_.max() + data.dr[1]
        z_min, z_max = data.z.min(), data.z.max()
        idx = [(self.x_ >= x_min) & (self.x_ <= x_max),
               (self.y_ >= y_min) & (self.y_ <= y_max),
               (self.z >= z_min) & (self.z <= z_max)]
        tip_pos = self.tip
        tip_pos = tip_pos[idx[0], ...]
        tip_pos = tip_pos[:, idx[1], ...]
        tip_pos = tip_pos[..., idx[2], :]
        X = np.repeat(self.x[..., np.newaxis], len(self.z), axis=2) - x_min
        Y = np.repeat(self.y[..., np.newaxis], len(self.z), axis=2) - y_min
        Z = np.ones(self.shape) * self.z - z_min
        nX = (X + tip_pos[..., 0]) / data.dr[0]
        nY = (Y + tip_pos[..., 1]) / data.dr[1]
        nZ = (Z + tip_pos[..., 2]) / data.dr[2]
        res = np.zeros(tip_pos.shape[:-1])
        it = np.nditer([nX, nY, nZ], flags=['multi_index'])
        for x, y, z in it:
            res[it.multi_index] = interp.ip(list([x, y, z]))
        new = self.copy()
        if new.shape != res.shape:
            new.shape = res.shape
            new.x = self.x[idx[0]]
            new.y = self.y[idx[1]]
            new.z = self.z[idx[2]]
            new.grid = [new.x, new.y, new.z]
        new.data = res
        return new

    def interpolate_line(self, xi, xf, points=100):
        import tricubic
        interp = tricubic.tricubic(list(self.data), list(self.data.shape))
        x_min, x_max = self.x_.min(), self.x_.max() + self.dr[0]
        y_min, y_max = self.y_.min(), self.y_.max() + self.dr[1]
        z_min, z_max = self.z.min(), self.z.max()
        X, Y, Z = np.linspace(xi, xf, points).T
        nX = (X - x_min) / self.dr[0]
        nY = (Y - y_min) / self.dr[1]
        nZ = (Z - z_min) / self.dr[2]
        it = np.nditer([nX, nY, nZ, None])
        for x, y, z, r in it:
            r[...] = interp.ip(list([x, y, z]))
        return it.operands[-1]

    def get_z(self, z, n=False):
        """
        Get the closest z-coordinate from the given value.

        Parameters:
            z (float): The target z-coordinate value.
            n (bool): If True, return the index of the closest z-coordinate instead of the value.

        Returns:
            float or int: The closest z-coordinate value or index, depending on the value of `n`.
        """
        nz = np.abs(self.z - z).argmin()
        if n:
            return nz
        return self.z[nz]

    def get_gradient(self, axis=2):
        new = copy.deepcopy(self)
        new.data = np.gradient(self.data, -self.dr[axis], axis=axis)
        new.label = f'grad({self.label})'
        return new

    def func(self, func, save=False, args=[], kwargs={}):
        new = func(self.data, *args, **kwargs)
        if save:
            self.__setattr__(func.__name__, new)
        else:
            return new

    def copy(self):
        new = copy.deepcopy(self)
        return new

    def __neg__(self):
        new = copy.deepcopy(self)
        new.data = -self.data
        return new

    def __add__(self, other):
        new = copy.deepcopy(self)
        try:
            idx0 = [i in other.z for i in self.z]
            idx1 = [i in self.z for i in other.z]
            data = self.data[:, :, idx0] + other.data[:, :, idx1]
            z = self.z[idx0]
        except AttributeError:
            data = self.data + other
            z = self.z
        new.data = data
        new.z = z
        return new

    def __sub__(self, other):
        new = copy.deepcopy(self)
        return new.__add__(-other)

    def __mul__(self, other):
        new = copy.deepcopy(self)
        new.data = self.data * other
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        new = copy.deepcopy(self)
        new.data = self.data / other
        return new

    def __str__(self):
        string = f'STMGrid(Label: {self.label}, z-min: {self.z.min():.4f}, \
z-max: {self.z.max():.4f}, dz: {self.dr[2]:.4f}, zref: {self.zref:.4f}, min. \
value: {self.data.min():.2e}, max. value: {self.data.max():.2e}'

        if self.kwargs:
            string += ', '
            for k, u in self.kwargs.items():
                try:
                    string += k + f': {u:2e}, '
                except ValueError:
                    string += k + ': ' + u.__str__() + ', '
            string = string[:-2]
        string = string + ')'
        return string

    def __repr__(self):
        return self.__str__()
