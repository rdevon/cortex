from cortex.datasets import Shapes

class Spiral(Shapes):
    """Download and use the Spiral dataset.

    N=312, k=3, D=2

    H. Chang and D.Y. Yeung,
    Robust path-based spectral clustering.
    Pattern Recognition, 2008.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/spiral.txt"]
