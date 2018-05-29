from cortex.datasets import Shapes

class Pathbased(Shapes):
    """Download and use the Pathbased dataset.

    N=300, k=3, D=2

    H. Chang and D.Y. Yeung,
    Robust path-based spectral clustering.
    Pattern Recognition, 2008.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/pathbased.txt"]
