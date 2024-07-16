"""Read NestedSamples from dnest4 output files"""
import os
import numpy as np
from anesthetic.samples import MCMCSamples


def heuristically_determine_columns(n_params):
    """
    heuristically determines column names. If none are given, parameters are named x_i
    """
    return [f'x{i}' for i in range(n_params)]


def read_dnest4(root,
                *args,
                **kwargs):
    """Read dnest4 output files.

    Parameters
    ----------
    levels_file default output name from dnest4
    sample_file default output name from dnest4
    sample_info_file default output name from dnest4
    root : str
        root specify the directory only, no specific roots,
        The files read files are levels_file, sample_file and sample_info.
    """
    levels_file = 'levels.txt'
    sample_file = 'sample.txt'
    sample_info_file = 'sample_info.txt'
    weights_file = 'weights.txt'

    levels = np.loadtxt(os.path.join(root, levels_file), dtype=float, delimiter=' ', comments='#')
    samples = np.genfromtxt(os.path.join(root, sample_file), dtype=float, delimiter=' ', comments='#', skip_header=1)
    sample_info = np.loadtxt(os.path.join(root, sample_info_file), dtype=float, delimiter=' ', comments='#')
    weights = np.loadtxt(os.path.join(root, weights_file), dtype=float, delimiter=' ', comments='#')
    n_params = samples.shape[1]
    columns = heuristically_determine_columns(n_params)

    return MCMCSamples(data=samples,
                   columns=columns,
                   weights=weights,
                   logL=sample_info[:, 1],
                   labels=columns, *args,
                   **kwargs)
