from cortex.datasets import SmallDataset
import torch
import os

class G2(SmallDataset):
    """Download and use G2 dataset.

    Select arguments
    ----------------
    dim : int
       Dimension of the input space
    sd : int
       Standard deviation of the Gaussian used to generate the 2 modes

    See possible values below.

    -----------------------------------------------------------------------
    G2 datasets creation
    -----------------------------------------------------------------------

    The datasets include two Gaussian normal distributions:

    Dataset name:    G2-dim-sd
    Centroid 1:      [500,500, ...]
    Centroid 2:      [600,600, ...]
    Dimensions:      dim = 1,2,4,8,16, ... 1024
    St.Dev:          sd  = 10,20,30,40 ... 100

    They have been created using the following C-language code:

    Calculate random value in (0,1]:

    U = (double)(rand()+1)/(double)(RAND_MAX+1);
    V = (double)(rand()+1)/(double)(RAND_MAX+1);

    Box-Muller method to create two independent standard
    one-dimensional Gaussian samples:

    X = sqrt(-2*log(U))*cos(2*3.14159*V);  /* pi = 3.14159 */
    Y = sqrt(-2*log(U))*sin(2*3.14159*V);

    Adjust mean and deviation:

    X_final = 500 + s * X;    /* mean + deviation * X */
    Y_final = 600 + s * Y;

    The points are stored in the files so that:
    - First 1024 points are from the cluster 1
    - Rest  1024 points are from the cluster 2

    -----------------------------------------------------------------------

    P. Fränti R. Mariescu-Istodor and C. Zhong, "XNN graph"
    IAPR Joint Int. Workshop on Structural, Syntactic, and Statistical Pattern Recognition Merida,
    Mexico, LNCS 10029, 207-217, November 2016.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/g2-txt.zip"]

    def prepare(self, dim, sd):
        """Make torch Tensors from g2-`dim`-`sd` and infer labels."""
        filename = 'g2-{}-{}.txt'.format(dim, sd)
        data = []
        target = []
        with open(os.path.join(self.root, filename)) as in_f:
            for i, line in enumerate(in_f):
                a, b = list(map(int, line.split())), 0 if i < 1024 else 1
                data.append(a)
                target.append(b)
        data = torch.Tensor(data)
        target = torch.Tensor(target)

        if self.stardardize:
            data = (data - 550) / 50

        return data, target

    def check_exists(self):
        return os.path.exists(os.path.join(self.root, 'g2-1-10.txt'))
