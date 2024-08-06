import numpy as np
from pydbspm.grid import Grid, get_grid_from_params


class DensityCalculator:
    def __init__(self, params=None):
        self.params = params
        self.sample = None
        self.rhoS = None
        self.potS = None
        self.gridS = None
        self.tip = None
        self.rhoT = None
        self.nrhoT = None
        self.gridT = None
        self.grid = None
        self.nidx = None
        self.stm = None
        self.stms = None
        self.stmp = None
        self.tippos = None

    def set_sample(self, gridS):
        self.gridS = gridS
        self.rhoS = gridS.rhoS
        self.potS = gridS.potS

    def set_tip(self, gridT):
        self.gridT = gridT
        self.rhoT = gridT.rhoT
        self.nrhoT = gridT.nrhoT

    def load_sample(self, sample_dir=None):
        if not sample_dir and self.params:
            sample_dir = self.params.SAMPLE / "sample.npz"
        elif not sample_dir:
            sample_dir = "sample/sample.npz"
        self.sample = np.load(sample_dir)
        self.rhoS = self.sample["rhoS"]
        self.potS = self.sample["potS"]
        self.gridS = Grid(
            self.rhoS.shape,
            cell=self.sample["cell"],
            positions=self.sample["positions"],
            numbers=self.sample["numbers"],
        )

    def load_tip(self, tip_dir=None):
        if not tip_dir and self.params:
            tip_dir = self.params.TIP / "tip.npz"
        elif not tip_dir:
            tip_dir = "tip/tip.npz"
        self.tip = np.load(tip_dir)
        self.rhoT = self.tip["rhoT"]
        self.nrhoT = self.tip["nrhoT"]
        self.gridT = Grid(
            self.rhoT.shape,
            cell=self.tip["cell"],
            positions=self.tip["positions"],
            numbers=self.tip["numbers"],
        )

    def set_calculation_grid(self):
        self.grid, self.nidx = get_grid_from_params(
            self.params,
            self.gridS,
            nidx=True,
            rpivot=self.params.RPIVOT,
            zpad=self.params.ZPAD,
            numbers=self.gridS.numbers,
            positions=self.gridS.positions,
        )
