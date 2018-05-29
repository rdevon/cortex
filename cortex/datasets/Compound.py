from cortex.datasets import Shapes


class Compound(Shapes):
    """Download and use the Compound dataset.

    N=399, k=6, D=2

    C.T. Zahn,
    Graph-theoretical methods for detecting and describing gestalt clusters.
    IEEE Transactions on Computers, 1971.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/Compound.txt"]


