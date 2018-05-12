'''Toy dataset clustering

'''

from sklearn import metrics
import torch

from ali import apply_penalty, score
from featnet import score as featnet_score, get_results, visualize
from gan import generator_loss, apply_gradient_penalty
from modules.fully_connected import FullyConnectedNet
from utils import to_one_hot


# Must have data, models, losses, results, and viz. **kargs should match the keys in DEFAULT_CONFIG.routines below.
def noise_discriminator_routine(data, models, losses, results, viz, noise_measure='JSD'):
    X, Y_P = data.get_batch('1.images', 'y')
    encoder = models.encoder
    Y_Q = encoder(X, nonlinearity=torch.nn.Softmax(dim=1))

    E_pos, E_neg, _, _ = featnet_score(models, Y_P, Y_Q, noise_measure, key='noise_discriminator')
    losses.noise_discriminator = E_neg - E_pos


def mine_discriminator_routine(data, models, losses, results, viz, mine_measure='JSD'):
    X_P, X_Q, T = data.get_batch('1.images', '2.images', '1.targets')
    encoder = models.encoder

    noise = data.noise['y'][0]
    with torch.set_grad_enabled(True):
        N = noise.sample()
        N.requires_grad_()
        loss = N.sum()

    loss.backward()
    print(N.grad)
    assert False, N.requires_grad

    Y_Q = encoder(X_P, nonlinearity=torch.nn.Softmax(dim=1))
    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Y_Q, Y_Q, mine_measure, key='mine')

    losses.mine = E_neg - E_pos


def encoder_routine(data, models, losses, results, viz, mine_measure=None, noise_measure=None,
                    generator_loss_type='non-saturating'):
    X_P, X_Q, T, Y_P = data.get_batch('1.images', '2.images', '1.targets', 'y')
    dim_l = data.get_dims('labels')
    encoder = models.encoder
    Y_Q = encoder(X_P, nonlinearity=torch.nn.Softmax(dim=1))

    D = Y_Q.argmax(1)
    C = D.data.cpu().numpy()
    ARI = metrics.adjusted_rand_score(T, C)
    AMI = metrics.adjusted_mutual_info_score(T, C)
    homogeneity = metrics.homogeneity_score(T, C)
    completeness = metrics.completeness_score(T, C)
    v_measure = metrics.v_measure_score(T, C)
    FMI = metrics.fowlkes_mallows_score(T, C)
    results.update(Cluster_scores=dict(ARI=ARI, AMI=AMI, homogeneity=homogeneity, completeness=completeness,
                                       v_measure=v_measure, FMI=FMI))
    class_numbers = {}
    target_numbers = to_one_hot(T, dim_l).sum(0)
    found_numbers = to_one_hot(D, dim_l).sum(0)
    for l in range(dim_l):
        class_numbers['GT_{}'.format(l)] = target_numbers[l]
        class_numbers['EN_{}'.format(l)] = found_numbers[l]
    results.update(Class_numbers=class_numbers)

    # MINE
    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Y_Q, Y_Q, mine_measure, key='mine')

    losses.encoder = E_neg - E_pos
    get_results(P_samples, Q_samples, E_pos, E_neg, mine_measure, results=results, name='mine')
    visualize(Y_Q, P_samples, Q_samples, X_P, D, Y_Q=X_P, viz=viz)

    # Featnet
    E_pos_n, E_neg_n, P_samples_n, Q_samples_n = featnet_score(models, Y_P, Y_Q, noise_measure,
                                                               key='noise_discriminator')
    get_results(P_samples_n, Q_samples_n, E_pos_n, E_neg_n, noise_measure, results=results, name='noise')
    losses.encoder += generator_loss(Q_samples_n, noise_measure, loss_type=generator_loss_type)


def penalty_routine(data, models, losses, results, viz, mine_penalty_amount=1.0, penalty_amount=5.0):
    X_P, X_Q, Y_P = data.get_batch('1.images', '2.images', 'y')
    encoder = models.encoder
    Y_Q = encoder(X_P, nonlinearity=torch.nn.Softmax(dim=1))

    penalty = apply_gradient_penalty(data, models, inputs=(Y_P, Y_Q), model='noise_discriminator',
                                     penalty_amount=penalty_amount)
    if penalty:
        losses.noise_discriminator = penalty

    penalty = apply_penalty(models, losses, results, X_P, Y_Q, mine_penalty_amount, key='mine')
    if penalty:
        losses.mine = penalty


# CORTEX ===============================================================================================================
# Must include `BUILD` and `TRAIN_ROUTINES`

def BUILD(data, models, dim_embedding=2, noise_type='dirichlet', noise_parameters=dict(concentration=0.2),
          encoder_args=dict(dim_h=[20, 20], batch_norm=True),
          noise_discriminator_args=dict(dim_h=[200, 100, 10], batch_norm=False),
          mine_discriminator_args=dict(dim_h=[20, 20], batch_norm=False)):
    dim_in = data.get_dims('x')
    data.add_noise('y', dist=noise_type, size=dim_embedding, **noise_parameters)

    encoder = FullyConnectedNet(dim_in, dim_out=dim_embedding, **encoder_args)
    mine_bot = FullyConnectedNet(dim_in, dim_out=dim_embedding, **mine_discriminator_args)
    mine_top = FullyConnectedNet(dim_embedding, dim_out=dim_embedding, **mine_discriminator_args)
    mine_fin = FullyConnectedNet(2 * dim_embedding, dim_out=1, **mine_discriminator_args)

    noise_discriminator = FullyConnectedNet(dim_embedding, dim_out=1, **noise_discriminator_args)

    models.update(mine=(mine_bot, mine_top, mine_fin), noise_discriminator=noise_discriminator, encoder=encoder)


# Dictionary reference to train routines. Keys are up to you
TRAIN_ROUTINES = dict(mine=mine_discriminator_routine, discriminator=noise_discriminator_routine,
                      encoder=encoder_routine, penalty=penalty_routine)

# Dictionary reference to test routines. If not set, will be copied from train. If value is None, will not be used in test.
TEST_ROUTINES = dict(penalty=None)

# Default configuration for this model
DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=640), duplicate=2),
    optimizer=dict(),
    train=dict()
)