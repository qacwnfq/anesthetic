"""Read NestedSamples from dnest4 output files."""
import os
import numpy as np
from anesthetic.samples import DiffusiveNestedSamples


def _determine_columns(n_params, header, delim=' '):
    """
    Determine column names from DNest4 output.

    If none are given by DNest4, parameters are named x_i.
    ' ' is the default delimiter in DNest4.
    """
    dnest4_column_descriptions = header[1:].lstrip().split(delim)
    if len(dnest4_column_descriptions) != n_params:
        # header can not have contained column names
        columns = [f'x_{i}' for i in range(n_params)]
    else:
        columns = [d.strip() for d in dnest4_column_descriptions]
    return columns


def read_dnest4(root,
                levels_file='levels.txt',
                sample_file='sample.txt',
                sample_info_file='sample_info.txt',
                *args,
                **kwargs):
    """
    Read dnest4 output files.

    Parameters
    ----------
    root : str
        root specify the directory only, no specific roots,
        The files read files are levels_file, sample_file and sample_info.
    levels_file: str
        output name from DNest4
    sample_file: str
        output name from DNest4
    sample_info_file: str
        output name from DNest4
    """
    levels = np.loadtxt(os.path.join(root, levels_file),
                        dtype=float,
                        delimiter=' ',
                        comments='#')
    samples = np.genfromtxt(os.path.join(root, sample_file),
                            dtype=float,
                            delimiter=' ',
                            comments='#')
    sample_info = np.loadtxt(os.path.join(root, sample_info_file),
                             dtype=float,
                             delimiter=' ',
                             comments='#')

    with open(os.path.join(root, sample_file), 'r') as f:
        header = f.readline()

    n_params = samples.shape[1]

    sample_level = sample_info[:, 0].astype(int)
    logL = sample_info[:, 1]
    logL_birth = levels[sample_level, 1]

    kwargs['label'] = kwargs.get('label', os.path.basename(root))
    columns_ = _determine_columns(n_params, header)
    columns = kwargs.pop('columns', columns_)
    labels_ = {c: '$' + c + '$' for c in columns}
    labels = kwargs.pop('labels', labels_)

    return DiffusiveNestedSamples(sample_info=sample_info,
                                  levels=levels,
                                  samples=samples,
                                  logL=logL,
                                  logL_birth=logL_birth,
                                  columns=columns,
                                  labels=labels,
                                  *args,
                                  **kwargs)
