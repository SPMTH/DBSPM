from ase.io import read

def set_sample_atoms(sample_path, params):
    wd = sample_path.parent
    sample = read(sample_path)
    if not sample.cell.any():
        sample_size = sample.positions.max(axis=0) - sample.positions.min(axis=0)
        sample_size += [12, 12, 18]
        sample.cell = (sample_size // 3) * 3
        if params.CENTER:
            sample.center()
            sample.positions[:, 2] += params.ZREF - sample.positions[:, 2].mean()
    return wd, sample

