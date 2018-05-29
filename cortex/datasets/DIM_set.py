from cortex.datasets import SmallDataset

class DIM_set(SmallDataset):
    """Download and use the (high) DIM-sets dataset.

    High-dimensional data sets N=1024 and k=16 Gaussian clusters.

    Select arguments
    ----------------
    dim : int
       Dimension of the input space. Choose: [32, 64, 128, 256, 512, 1024]

    P. Fränti, O. Virmajoki and V. Hautamäki,
    "Fast agglomerative clustering using a k-nearest neighbor graph",
    IEEE Trans. on Pattern Analysis and Machine Intelligence, 28 (11), 1875-1881,
    November 2006.
    """

    urls = [
        "http://cs.joensuu.fi/sipu/datasets/dim032.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim032.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim064.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim064.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim128.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim128.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim256.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim256.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim512.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim512.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim1024.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim1024.pa",
        ]

    sync_files = 5

    def files(self, dim):
        return 'dim{:03d}.txt'.format(dim), 'dim{:03d}.pa'.format(dim)

    def check_exists(self):
        return os.path.exists(os.path.join(self.root, 'dim032.txt'))

