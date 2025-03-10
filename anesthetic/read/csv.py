"""Read and write CSV files for anesthetic."""
from anesthetic.weighted_labelled_pandas import read_csv as wl_read_csv
from anesthetic.samples import MCMCSamples, NestedSamples
from pathlib import Path


def read_csv(filename, *args, **kwargs):
    """Read a CSV file into a :class:`Samples` object."""
    try:
        filename = Path(filename)
        kwargs['label'] = kwargs.get('label', filename.stem)
        filename = filename.with_suffix('.csv')
    except TypeError:
        pass
    wldf = wl_read_csv(filename)
    if 'nlive' in wldf.columns:
        return NestedSamples(wldf, *args, **kwargs)
    else:
        return MCMCSamples(wldf, *args, **kwargs)
