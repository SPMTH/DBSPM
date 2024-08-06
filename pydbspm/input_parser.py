#!/usr/bin/python
import re
from numpy import array, ndarray, stack
from argparse import Namespace
from pathlib import Path

path_params = {
    "TIP": Path("./tip"),
    "SAMPLE": Path("./sample"),
    "GRID": Path("grid.npz"),
    "DFT": Path("./DFT/Fz"),
}
string_params = {
    "PSEUDOS": "potpaw_PBE",
    "CMD": "mpirun vasp_gam",
    "CALC": "vasp",
    "SAMPLERHO": None,
    "SAMPLEPOT": None,
    "TIPRHO": None,
    "TIPNRHO": None,
    "MINMETHOD": "Powell",
}
bool_params = {
    "DFTCHG": True,
    "DFTPOT": True,
    "TIPGRID": True,
    "SAMPLEGRID": True,
    "POTFILTER": False,
    "CENTER": False,
    "FIT": False,
    "FITV0": False,
    "REFIT": False,
    "REAL": True,
    "MPI": False,
    "RELAX": True,
    "OPT": False,
}
float_params = {
    "ALPHA": 1.08,
    "V": 42.91,
    "Z0": 2.8,
    "ZF": 4.0,
    "ZREF": None,
    "ZPAD": 1.0,
    "RPIVOT": 3.02,
    "K": 0.2,
}
array_params = {
    "ZVAL": None,
    "BBOX0": None,
    "BBOXF": None,
    "BBOX": None,
    "ARANGE": [1.0, 1.12, 0.01],
    "XLIM": [0.0, 1.0],
    "YLIM": [0.0, 1.0],
    "COMPONENTS": ["sr", "es", "vdw"],
}
# Parameter not read from input file
internal_params = {
    "COMM": None,
}
default_params = {
    **path_params,
    **string_params,
    **bool_params,
    **float_params,
    **array_params,
    **internal_params,
}


def parse_params(input: str) -> Namespace:
    """
    Parse input file and return a Namespace object with the parameters.

    Parameters:
        input (str): Path to the input file.
    """
    with open(input) as f:
        pattern = re.compile(r"^(\w+)\s*=\s*([^#|^\n]+)", re.M)
        params = dict(pattern.findall(f.read()))

    for paths in path_params.keys():
        try:
            params[paths] = Path(params[paths])
        except KeyError:
            params[paths] = default_params[paths]

    for bools in bool_params.keys():
        try:
            if params[bools] == "True":
                params[bools] = True
            else:
                params[bools] = False
        except KeyError:
            params[bools] = default_params[bools]

    for flts in float_params.keys():
        try:
            params[flts] = float(params[flts])
        except KeyError:
            params[flts] = default_params[flts]

    for strngs in string_params.keys():
        try:
            params[strngs] = str(params[strngs])
        except KeyError:
            params[strngs] = default_params[strngs]

    for arrs in array_params.keys():
        try:
            params[arrs] = array(params[arrs].split(), dtype=float)
        except KeyError:
            params[arrs] = default_params[arrs]

    if type(params["BBOX0"]) is ndarray and type(params["BBOXF"]) is ndarray:
        params["BBOX"] = stack([params["BBOX0"], params["BBOXF"]])

    return Namespace(**params)


def parse_sites(input: str) -> "dict[str, ndarray]":
    with open(input) as f:
        site_patt = re.compile(
            r"(^\w+-\w+|^\w+)\s+([+-]?\d+.\d+|[+-]?.\d+|[+-]?\d+.)\s+([+-]?\d+.\d+|[+-]?.\d+|[+-]?\d+.)",
            re.M,
        )
        site_data = site_patt.findall(f.read())
        sites = {}
        for site in site_data:
            sites[site[0]] = array(site[1:], dtype=float)
        if not sites.keys():
            raise ValueError(
                "There were no sites on the input file. Sites must be provided for the calculation."
            )
    return sites
