from cortex.datasets import SmallDataset
import os
class S_set(SmallDataset):
    """Download and use S-sets dataset.

    Synthetic 2-d data with N=5000 vectors and k=15 Gaussian clusters
    with different degree of cluster overlapping.

    Select Arguments
    ----------------
    num : int
       Higher `num` means, higher chance of overlapping between the modes.
       Choose: [1, 2, 3, 4]

    P. Fränti and O. Virmajoki,
    "Iterative shrinking method for clustering problems",
    Pattern Recognition, 39 (5), 761-765, May 2006.
    """

    urls = [
        "http://cs.joensuu.fi/sipu/datasets/s1.txt",
        "http://cs.joensuu.fi/sipu/datasets/s2.txt",
        "http://cs.joensuu.fi/sipu/datasets/s3.txt",
        "http://cs.joensuu.fi/sipu/datasets/s4.txt",
        "http://cs.joensuu.fi/sipu/datasets/s-originals.zip"
        ]

    sync_files = 5

    def files(self, num):
        """Make torch Tensors from 's{num}.txt' and fetch labels."""
        return 's{}.txt'.format(num), 's{}-label.pa'.format(num)

    def check_exists(self):
        return os.path.exists(os.path.join(self.root, 's1.txt'))