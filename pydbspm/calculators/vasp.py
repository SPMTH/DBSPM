from re import compile, M
from pathlib import Path
from argparse import Namespace
import numpy as np
from numpy import array, empty, int32, linalg
from scipy.ndimage.filters import uniform_filter1d
from ase.calculators.vasp import Vasp, VaspChargeDensity
from ase.io.vasp import read_vasp
from ase.io import read
from ase import Atoms
from vaspwfc import vaspwfc
from time import gmtime, strftime
from scipy.ndimage import gaussian_filter

from pydbspm.grid import Grid, neutralize_charge, filter_neg, threeAxis, threeN


def get_vasp_params(INPUT=None) -> dict:
    vasp_dict = dict(
        encut=400,
        istart=0,
        icharg=2,
        sigma=0.1,
        prec="Accurate",
        algo="fast",
        ismear=0,
        ivdw=11,
        idipol=3,
        lreal="Auto",
        lcharg=True,
        lwave=True,
    )

    if INPUT:  # read vasp params if available
        vasp = vasp_params(INPUT)
        # check for conflicting flags
        for l in ["ivdw", "idipol", "lcharg", "lwave"]:
            if l in vasp.keys() and vasp[l] != vasp_dict[l]:
                print(
                    f"! Warning: {l} in {INPUT} ({vasp[l]}) is different from the recommended value ({vasp_dict[l]}). Hope you know what you are doing!!!"
                )
        # update vasp_scf dictionary with options.input params
        vasp_dict.update(vasp)

    return vasp_dict


def calculate_sample(params):

    if not (params.DFTCHG or params.DFTPOT):
        Exception(
            f"Both DFTCHG and DFTPOT flags are set to False in {params.INPUT}. Nothing to calculate. Fix your input file or the steps you requested in the command-line."
        )

    # Set SCF VASP calculation.
    vasp_scf = get_vasp_params(params.INPUT)
    vasp_init = Vasp(command=params.CMD, pp=params.PSEUDOS, **vasp_scf)
    wd = Path(params.SAMPLE)
    print(f" - DFT command: {params.CMD}")
    print(f" - Sample directory: {params.SAMPLE}")

    # Import coordinates to atoms objects and set needed variables.
    sample = read_vasp(params.SAMPLE / "POSCAR")
    print("\nSample coordinates (Ang)")
    print("----------------------------")
    for i, s in enumerate(sample.get_chemical_symbols()):
        print("{}\t{:.4f}\t{:.4f}\t{:.4f}".format(s, *sample.positions[i]))
    print("\nCell lattice vectors (Ang)")
    print("----------------------------")
    print(sample.cell)

    if params.DFTCHG:
        # vasp_init.set(directory=str(wd))
        # sample.set_calculator(vasp_init)
        # print("\n > Running DFT code")
        # E = sample.get_potential_energy()
        # assert E is not None, "Error: SCF calculation failed."

        vasp_init.set(directory=str(wd), atoms=sample, txt='vasp.out')
        vasp_init.write_input(sample, properties=["energy"])

        print("\n>> Running DFT code")
        with vasp_init._txt_outstream() as out:
            E = vasp_init._run(out=out)

        assert E == 0, "Error: SCF calculation failed."

        print("<< done.\n")

    if params.DFTPOT:
        print(">> Setting local potential calculation")
        potdir = wd / "LDIPOL"
        potdir.mkdir(parents=True, exist_ok=True)
        wvcar = potdir / "WAVECAR"
        chcar = potdir / "CHGCAR"
        if not (wvcar.is_file() or wvcar.is_symlink()):
            wvcar.symlink_to("../WAVECAR")
        if not wvcar.exists():
            raise Exception("WAVECAR was not present from SCF calculation.")
        if not (chcar.is_file() or chcar.is_symlink()):
            chcar.symlink_to("../CHGCAR")
        if not chcar.exists():
            raise Exception("CHGCAR was not present from SCF calculation.")

        vasp_scf = get_vasp_params(params.INPUT)

        # Set DIPOL VASP calculation.
        vasp_pot = Vasp(command=params.CMD,
                        pp=params.PSEUDOS,
                        directory=str(wd),
                        **vasp_scf)

        if params.DFTPOT and not params.DFTCHG:
            # read existing sample (charge density) calculation
            print(" - Reading sample charge density from: ", wd)
            vasp_pot.read()
        # set flags for postprocessing
        vasp_pot.set(
            command=params.CMD,
            pp=params.PSEUDOS,
            directory=potdir,
            icharg=1,
            istart=1,
            lcharg=False,
            lwave=False,
            ldipol=True,
            lvtot=True,
            lvhar=True,
        )
        print("<< done.\n")

        # print(" > Running DFT code")
        # sample.set_calculator(vasp_pot)
        # E_ = sample.get_potential_energy()

        # assert E_ is not None, "Error: Local potential calculation failed."

        vasp_pot.set(atoms=sample, txt='vasp.out')
        vasp_pot.write_input(sample, properties=["energy"])

        print(">> Running DFT code")
        with vasp_pot._txt_outstream() as out:
            E_ = vasp_pot._run(out=out)

        assert E_ == 0, "Error: SCF calculation failed."

        print("<< done.\n")

    return E if params.DFTCHG else E_


def calculate_tip(params: Namespace):
    # Set SCF VASP calculation.
    vasp_scf = get_vasp_params(params.INPUT)
    vasp_init = Vasp(command=params.CMD, pp=params.PSEUDOS, **vasp_scf)

    wd = Path(params.TIP)
    print(f" - DFT command: {params.CMD}")
    print(f" - Tip directory: {params.TIP}")

    # Import coordinates to atoms objects and set needed variables.
    if (params.TIP / "POSCAR").is_file():
        tip = read_vasp(params.TIP / "POSCAR")
    else:
        sample = read_vasp(params.SAMPLE / "POSCAR")
        tip = Atoms(
            "CO",
            [[0.0, 0.0, 1.153425], [0.0, 0.0, 0.0]],
            cell=sample.cell,
            pbc=True,
            calculator=vasp_init,
        )
    print("\nTip coordinates (Ang)")
    print("----------------------------")
    for i, s in enumerate(tip.get_chemical_symbols()):
        print("{}\t{:.4f}\t{:.4f}\t{:.4f}".format(s, *tip.positions[i]))
    print("\nCell lattice vectors (Ang)")
    print("----------------------------")
    print(tip.cell)

    vasp_init.set(directory=str(wd), txt='vasp.out')
    # tip.set_calculator(vasp_init)
    # print("\n > Running DFT code")
    # E = tip.get_potential_energy()

    # assert E is not None, "Error: SCF calculation failed."
    # print(" < done.\n")
    vasp_init.write_input(tip, properties=["energy"])

    print("\n>> Running DFT code")
    with vasp_init._txt_outstream() as out:
        E = vasp_init._run(out=out)

    assert E == 0, "Error: SCF calculation failed."

    print("<< done.\n")

    return E


def read_sample(params: Namespace, rank=0):
    s = params.SAMPLE
    sample_path = s / "CHGCAR"
    sampot_path = s / "LDIPOL/LOCPOT"

    if rank == 0:
        sample = read_vasp(sample_path)
        cell = sample.cell[:].copy()
        shape = get_shape(sample_path, len(sample))

        grid = Grid(shape,
                    cell,
                    numbers=sample.numbers,
                    positions=sample.positions)
        
        print(" - Reading sample charge density from: ", sample_path)
        sample_data = VaspChargeDensity(sample_path)
        rhoS = sample_data.chg[0]
        rhoS = filter_neg(rhoS, prec=0)
        print(" - Reading sample local potential from: ", sampot_path)
    if (not params.COMM and rank == 0) or (params.COMM and rank == 1):
        sample_pot = VaspChargeDensity(sampot_path)
        potS = sample_pot.chg[0]
        # Multiply POT*V because ASE does chg/V
        potS *= sample_pot.atoms[0].get_volume()
        if params.POTFILTER:
            potS = uniform_filter1d(potS, 2, axis=2)

    if params.COMM:
        if rank == 1:
            params.COMM.Send(potS, dest=0)
            rhoS = potS = grid = None
        elif rank == 0:
            potS = empty(shape, dtype=float)
            params.COMM.Recv(potS, source=1)

    return rhoS, potS, grid


def read_tip(params: Namespace):
    t = params.TIP
    tip_path = t / "CHGCAR"
    tip_outcar = t / "OUTCAR"
    tip = read_vasp(tip_path)
    cell = tip.cell[:].copy()
    shape = get_shape(tip_path, len(tip))

    grid = Grid(shape, cell, numbers=tip.numbers, positions=tip.positions)

    rhoT = None

    print(" - Reading sample charge density from: ", tip_path)
    tip_data = VaspChargeDensity(tip_path)
    rhoT = tip_data.chg[0]
    rhoT = filter_neg(rhoT, prec=0)

    print(" > Neutralazing tip charge density...")
    if not params.ZVAL:
        print(" >> Using the OUTCAR file to get the atomic charges.")
        zval = get_zval(tip_outcar)
    else:
        print(" >> Using the atomic charges (ZVAL) in the input file.")
        assert len(params.ZVAL) == len(
            tip), "ZVAL must have the same number of atoms as the tip."
        zval = params.ZVAL
    print(" - Tip atoms:  ", tip.symbols)
    print(" - ZVAL: ", zval)
    print(" << done.\n")

    nrhoT = neutralize_charge(
        rhoT,
        grid.dr,
        tip.positions,
        zval=zval,
        filter_negative=True,
        neutral=True,
    )
    print(" - New tip total charge: {}".format(nrhoT.sum() * grid.dr))
    print(" < done.\n")
    return rhoT, nrhoT, grid


def check_grid_consistency(gridS, gridP, gridT):
    if (gridT - gridS).any():
        raise ValueError(
            "Tip and sample grids are different: {} != {}.".format(gridT, gridS)
        )
    if (gridT - gridP).any():
        raise ValueError(
            "Tip and sample potential grids are different: {} != {}.".format(
                gridT, gridS
            )
        )
    return True


def vasp_params(input):
    with open(input) as f:
        pattern = compile(r"^\$\s*(\w+)\s*=\s*([^#|^\n]+)", M)
        vasp = pattern.findall(f.read())
        for i, t in enumerate(vasp):
            t = list(t)
            t[0] = t[0].lower()
            t[1] = eval(t[1])
            vasp[i] = t
        vasp = dict(vasp)
        if "magmom" in vasp.keys():
            vasp["magmom"] = array(vasp["magmom"])
    return vasp


def get_shape(chg_file: str, lenght: int) -> threeAxis:
    """Get the grid shape from a CHGCAR-type file (chg_file).
    Requires number of atoms in calculation (lenght)."""
    shape = empty(3, dtype=int32)
    with open(chg_file) as f:
        for line in f:
            if line.startswith("Direct"):
                for _ in range(lenght + 1):
                    f.readline()
                line = f.readline()
                shape[...] = line.split()
                assert len(shape) == 3, "Incorrect grip shape."
                break
        Exception("Grid shape not found in CHGCAR.")
    return shape


def get_zval(outcar: str):
    """Get the valence electrons from the OUTCAR file."""
    chem_sym = []
    with open(outcar) as f:
        for line in f:
            if line.startswith("TITEL", 3):
                chem_sym += line.split()[3]
            if line.startswith("ZVAL", 3):
                zval = array(line.split()[2:], dtype=float)
                print(f"Valence electrons for {chem_sym} in OUTCAR: {zval}")
                assert len(chem_sym) == len(zval), "ZVAL incorrect."
                break
        Exception("ZVAL not found in OUTCAR.")
    return zval


""" def constant_height(ldos, z, cell, zref=0, repeat=(1,1)):
    z0 = zref + z
    nz = ldos.shape[2]
    ldos_ = ldos.reshape((-1, nz))

    I = np.empty(ldos_.shape[0])

    zp = z0 / cell[2, 2] * nz
    zp = int(zp) % nz

    for i, a in enumerate(ldos_):
        I[i] = a[zp]

    s0 = I.shape = ldos.shape[:2]
    I = np.tile(I, repeat)
    s = I.shape

    ij = np.indices(s, dtype=float).reshape((2, -1)).T
    x, y = np.dot(ij / s0, cell[:2, :2]).T.reshape((2,) + s)
    return x, y, I """

    
""" def constant_height(self, channel, z, zref=0, repeat=(1,1)):
    x, y, I = constant_height(
        self.__getattribute__(channel),
        z,
        self.atoms.cell,
        zref=zref,
        repeat=repeat
        )
    return x, y, I """

def read_wfc(atoms,directory):
        wfc = vaspwfc(directory/"WAVECAR")
        shape = wfc.wfc_r(1,1,1).shape
        dr = linalg.norm(atoms.cell, axis=1)/shape
        return wfc, shape, dr

def get_bias(bias):
    if bias < 0:
        emin = bias
        emax = 0.0
    else:
        emin = 0
        emax = bias
    return emin, emax

def get_calc(directory='.'):
        '''We are going to try to read all the parameters we need
        from the vasprun.xml file. This is not always posible for
        the k-point weights and the Fermi level. If we fail in
        reading any of those from the xml, we are going to fallback to the ASE vasp calculator that will try to read IBZKPT and
        the OUTCAR to get each parameter.

        P.s. ASE has a bug when trying to get the kpoint weights, it
        does not read the IBZKPT in the right directory with the
        built-in get_k_point_weights method. We need to use the
        function directly.'''
        atoms = read(str(directory)+'/vasprun.xml')
        calc = atoms.calc
        c = Vasp(directory=directory)
        try:
            weights = calc.get_k_point_weights()
        except:
            try:
                weights = c.read_k_point_weights("IBZKPT")
            except FileNotFoundError:
                print(f'{directory}/IBZKPT not found. Assuming single k-point with weight=[1]. Make sure this behavior is expected!!!')
                weights = [1]
        nkpts = len(weights)
        # The xml ASE calculator does not return an exception if it
        # does not find the fermi level.
        Ef = calc.get_fermi_level()
        if not Ef:
            Ef = c.read_fermi()
        nspins = calc.get_number_of_spins()
        #self.nbands = self.calc.get_number_of_bands()
        eigs = np.array([[calc.get_eigenvalues(k, s)
                    for k in range(nkpts)]
                    for s in range(nspins)]) - Ef
        
        return atoms, nkpts, weights, Ef, eigs

def set_stm(bias, eigs, symmetries=[]):
    symmetries = symmetries
    emin, emax = get_bias(bias)
    skn = ((eigs < emax) & (emin < eigs)).nonzero()
    eigs_ = eigs[skn]
    return skn, eigs_


def get_stm(bias, directory, s=True, p=True, ws=0.5, wp=0.5, split=True, sigma=None, verbose=False):

    
    atoms, nkpts, weights, Ef, eigs = get_calc(directory)
    skn, eigs_ = set_stm(bias,eigs)
    wfc,shape,dr = read_wfc(atoms,directory)

    grid = Grid(shape,
                atoms.cell,
                numbers=atoms.numbers,
                positions=atoms.positions)

    ldos = np.zeros(shape)
    if split:
        ldos_p = np.zeros(shape)
    if verbose:
        print(strftime("Reading wavefunctions and making operations... (%H:%M:%S)", gmtime()))
        print(f'Number of kpoints: {nkpts}')
        print(f'Fermi level: {Ef} eV')
        print(f'Number of bands inside E_int: {len(eigs_)}')
    for s_, k_, n_ in zip(*skn):
        psi = wfc.wfc_r(s_+1,k_+1,n_+1)
        if s:
            ldos += ws * weights[k_] * (psi * np.conj(psi)).real
        if p:
            psi_dx = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0))\
                        / (2 * dr[0])
            psi_dy = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1))\
                        / (2 * dr[1])
            p_ = wp * weights[k_] * (psi_dx * np.conj(psi_dx) + psi_dy * np.conj(psi_dy)).real
            if sigma:
                p_ = gaussian_filter(p_, sigma=sigma, mode='wrap')
            if split:
                ldos_p += p_
            else:
                ldos += p_
    if verbose: print(strftime("... done. (%H:%M:%S)", gmtime()))

    if split:
        return grid, atoms, ldos, ldos_p
    else:
        return grid, atoms, ldos, ldos
