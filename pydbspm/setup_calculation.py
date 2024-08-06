from argparse import Namespace

# Define available calculation steps.
steps = Namespace(
    sample=False,
    tip=False,
    grid=False,
    sr=False,
    es=False,
    vdw=False,
    relax=False,
    stm=False,
    brstm=False,
)

stepsstr = list(steps.__dict__.keys())

static_potenital = Namespace(sr=True, es=True, vdw=True)

def set_steps_for_static_potential():
    for k, v in static_potenital._get_kwargs():
        setattr(steps, k, v)

mpirequired = ["vdw", "relax"]

def setup_calc(configstr):
    for i in configstr:
        # Set steps from the config string if they are as is in the step list.
        if i in stepsstr:
            setattr(steps, i, True)
        # Set steps from the config string if they are aliases or combinations.
        elif i == "prepdft":
            steps.sample = True
            steps.tip = True
        elif i == "prep":
            steps.grid = True
        elif i == "fullprep":
            steps.sample = True
            steps.tip = True
            steps.grid = True
        elif i == "pauli":
            steps.sr = True
        elif i in ["dens", "density_interactions"]:
            steps.sr = True
            steps.es = True
        elif i in ["disp", "d3"]:
            steps.vdw = True
        elif i in ["static", "all"]:
            set_steps_for_static_potential()
        elif i in ["afm", "fdbm"]:
            set_steps_for_static_potential()
            steps.relax = True
        elif i in ["fromdft"]:
            steps.grid = True
            set_steps_for_static_potential()
            steps.relax = True
        elif i in ["fullauto", "dft+afm"]:
            steps.sample = True
            steps.tip = True
            steps.grid = True
            set_steps_for_static_potential()
            steps.relax = True
        elif i == "stm":
            steps.stm = True
        elif i == "brstm":
            steps.brstm = True
        else:
            print("Unknown option: {}\nIgnoring...".format(i))
    return steps
