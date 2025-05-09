import numpy as np
from scipy.optimize import minimize
from numba import jit
from dftd3.interface import DispersionModel, ZeroDampingParam
from ase.units import Bohr, Ha
from pydbspm.grid import sph2grid, sph2grid_opt, sph2cart


def get_density_overlap(rho0, rho1, dr):
    """Get the Pauli exclusion energy with the charge density overlap of sample and tip.
    Takes:
        rho0:       Sample charge density.                      [(n,m,p) flt array]
        rho1:       Tip charge density on the origin.           [(n,m,p) flt array]
        dr:         Grid differentials (dx,dy,dz).              [(1,3) flt array]
    Returns:
        E:          Short-range energy in each tip position.    [(n,m,p) flt array]
    """
    dV = np.prod(dr)
    convFFT = np.fft.fftn(rho0) * np.fft.fftn(rho1[::-1, ::-1, ::-1])
    E = np.real(np.fft.ifftn(convFFT * dV))
    E = np.roll(E, 1, axis=(0, 1, 2))
    return E


def get_dftd3_api_grid(cell, posS, posT, numbersS, numbersT, gx, gy, gz):
    gx /= Bohr
    gy /= Bohr
    gz /= Bohr
    posS = np.array(posS / Bohr)
    posT = np.array(posT / Bohr)
    positions = np.append(posS, posT, axis=0)
    ntip = len(numbersT)
    atomtype = np.array([*numbersS] + [*numbersT], dtype=int)
    damp = ZeroDampingParam(s6=1.0, s8=0.722, rs6=1.217)
    d3 = DispersionModel(
        positions=positions,
        numbers=atomtype,
        lattice=cell / Bohr,
        periodic=np.array([True, True, False]),
    )
    it = np.nditer([gx, gy, gz, None])
    for xi, yi, zi, E in it:
        positions[-ntip:] = posT + [xi, yi, zi]
        d3.update(positions=positions)
        E[...] = d3.get_dispersion(param=damp, grad=False)["energy"]
    return it.operands[3] * Ha


@jit(nopython=True)
def spring_energy(th, kappa):
    spr = 0.5 * kappa * ((np.pi - th) ** 2)
    return spr


def Emin(pol, func, origin, kappa, rpivot, dr):
    th = pol[0]
    az = pol[1]
    spr = spring_energy(th, kappa)
    sol = func(list(origin + sph2grid(th, az, rpivot, dr))) + spr
    return sol

def Emin_norm(pol, func, origin, kappa, rpivot, dr, dR):
    th = pol[0]
    az = pol[1]
    spr = spring_energy(th, kappa)
    sol = func(list(origin + np.linalg.inv(dR.T).dot(sph2cart(th, az, rpivot)))) + spr
    return sol


def relax_tip_z(i, j, interpE, kappa, npivot, zi, dr, method="Powell"):
    origin = np.array([i, j, 0])
    guess = [np.pi, 0]
    rpivot = npivot * dr[2]
    z_it = np.nditer([zi[::-1], None], op_dtypes=[int, (float, (3))])
    for k, r in z_it:
        origin[2] = k + npivot
        emin = minimize(
            Emin,
            guess,
            method=method,
            args=(interpE.ip, origin, kappa, rpivot, dr),
        )
        guess = emin.x
        r[...] = [emin.fun, *guess]
    arr = z_it.operands[1][::-1]
    return arr

def relax_noortho_tip_z(i, j, interpE, kappa, npivot, zi, dr, dR, method="Powell"):
    origin = np.array([i, j, 0])
    guess = [np.pi, 0]
    rpivot = npivot * dr[2]
    z_it = np.nditer([zi[::-1], None], op_dtypes=[int, (float, (3))])
    for k, r in z_it:
        origin[2] = k + npivot
        emin = minimize(
            Emin_norm,
            guess,
            method=method,
            #                                bounds=[(np.pi/2,np.pi), (0,2*np.pi)],
            args=(interpE.ip, origin, kappa, rpivot, dr, dR),
        )
        guess = emin.x
        r[...] = [emin.fun, *guess]
    arr = z_it.operands[1][::-1]
    return arr


@jit(nopython=True)
def spring_energy_half_kappa(pi_th, half_kappa):
    spr = half_kappa * (pi_th**2)
    return spr


def Emin_opt(pol, func, origin, half_kappa, rpivot, dr):
    pi_th = pol[0]
    az = pol[1]
    spr = spring_energy_half_kappa(pi_th, half_kappa)
    sol = func(list(origin + sph2grid_opt(pi_th, az, rpivot, dr))) + spr
    return sol


def relax_tip_z_opt(i, j, interpE, kappa, npivot, zi, dr, method="Powell"):
    origin = np.array([i, j, 0])
    guess = [0.0, 0.0]
    rpivot = npivot * dr[2]
    half_kappa = 0.5 * kappa
    z_it = np.nditer([zi[::-1], None], op_dtypes=[int, (float, (3))])
    for k, r in z_it:
        origin[2] = k + npivot
        emin = minimize(
            Emin_opt,
            guess,
            method=method,
            args=(interpE.ip, origin, half_kappa, rpivot, dr),
        )
        guess = emin.x
        r[...] = [emin.fun, *guess]
    arr = z_it.operands[1][::-1]
    arr[1] = np.pi - arr[1]
    return arr

def relax_noortho_tip_z(i, j, interpE, kappa, npivot, zi, dr, dR, method="Powell"):
    origin = np.array([i, j, 0])
    guess = [np.pi, 0]
    rpivot = npivot * dr[2]
    z_it = np.nditer([zi[::-1], None], op_dtypes=[int, (float, (3))])
    for k, r in z_it:
        origin[2] = k + npivot
        emin = minimize(
            Emin_norm,
            guess,
            method=method,
            #                                bounds=[(np.pi/2,np.pi), (0,2*np.pi)],
            args=(interpE.ip, origin, kappa, rpivot, dr, dR),
        )
        guess = emin.x
        r[...] = [emin.fun, *guess]
    arr = z_it.operands[1][::-1]
    return arr
