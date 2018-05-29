from cortex.datasets import SmallDataset
import os

class A_set(SmallDataset):
    """Download and use A-sets dataset.

    Synthetic 2-d data with varying number of vectors (N) and clusters (k).
    There are 150 vectors per cluster.

    Select Arguments
    ----------------
    num : int
       Higher `num` means, higher chance of overlapping between the modes.
       Choose: [1, 2, 3]

    A1: N=3000, k=20
    A2: N=5250, k=35
    A3: N=7500, k=50

    I. Kärkkäinen and P. Fränti,
    "Dynamic local search algorithm for the clustering problem",
    Research Report A-2002-6
    """

    urls = [
        "http://cs.joensuu.fi/sipu/datasets/a1.txt",
        "http://cs.joensuu.fi/sipu/datasets/a2.txt",
        "http://cs.joensuu.fi/sipu/datasets/a3.txt",
        "http://cs.joensuu.fi/sipu/datasets/a-gt-pa.zip"
        ]

    sync_files = 4

    def files(self, num):
        return 'a{}.txt'.format(num), 'a{}-ga.pa'.format(num)

    def check_exists(self):
        return os.path.exists(os.path.join(self.root, 'a1.txt'))
