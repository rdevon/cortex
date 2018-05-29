from cortex.datasets import Shapes


class Flame(Shapes):
    """Download and use the Flame dataset.

    N=240, k=2, D=2

    L. Fu and E. Medico,
    FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data.
    BMC bioinformatics, 2007.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/flame.txt"]
