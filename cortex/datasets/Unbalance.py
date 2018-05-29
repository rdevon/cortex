from cortex.datasets import SmallDataset
import os

class Unbalance(SmallDataset):
    """Download and use the Unbalance dataset.

    Synthetic 2-d data with N=6500 vectors and k=8 Gaussian clusters

    There are 3 "dense" clusters of 2000 vectors each and
    5 "sparse" clusters of 100 vectors each.

    M. Rezaei and P. Fränti,
    "Set-matching methods for external cluster validity",
    IEEE Trans. on Knowledge and Data Engineering, 28 (8), 2173-2186,
    August 2016.
    """

    urls = [
        "http://cs.joensuu.fi/sipu/datasets/unbalance.txt",
        "http://cs.joensuu.fi/sipu/datasets/unbalance-gt-pa.zip",
        ]

    sync_files = 4

    def files(self):
        return 'unbalance.txt', 'unbalance-gt.pa'

    def check_exists(self):
        data, labels = self.files()
        return os.path.exists(os.path.join(self.root, data)) and\
            os.path.exists(os.path.join(self.root, labels))

