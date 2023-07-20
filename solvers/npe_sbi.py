from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchmark_utils.typing import Distribution, Tensor

with safe_import_context() as import_ctx:
    from functools import partial
    from nflows import transforms, distributions, flows
    from sbi.inference import SNPE

    from benchmark_utils.common import dump


class Solver(BaseSolver):
    r"""Neural posterior estimation (NPE) solver implemented with the
    :mod:`sbi` package.

    The solver trains a parametric conditional distribution :math:`q_\phi(\theta | x)`
    to approximate the posterior distribution :math:`p(\theta | x)` of parameters given
    observations.

    References:
        | Fast :math:`\espilon`-free Inference of Simulation Models with Bayesian Conditional Density Estimation (Papamakarios et al., 2016)
        | https://arxiv.org/abs/1605.06376

        | Automatic posterior transformation for likelihood-free inference (Greenberg et al., 2019)
        | https://arxiv.org/abs/1905.07488
    """  # noqa:E501

    name = "npe_sbi"
    stopping_criterion = SufficientProgressCriterion(
        patience=10,
    )
    parameters = {
        "flow": ["maf", "nsf"],
        "transforms": [1, 3, 5],
    }
    requirements = [
        "pip:sbi",
    ]

    @staticmethod
    def get_next(n_iter: int) -> int:
        return n_iter + 10

    def set_objective(self, theta: Tensor, x: Tensor, prior: Distribution):
        self.theta, self.x, self.prior = theta, x, prior

    def run(self, n_iter: int):
        def build(theta, x):
            features, context = theta.shape[-1], x.shape[-1]

            if self.flow == "maf":
                MAT = partial(
                    transforms.MaskedAffineAutoregressiveTransform,
                    features=features,
                    context_features=context,
                    hidden_features=64,
                    num_blocks=2,
                    use_residual_blocks=False,
                )
            elif self.flow == "nsf":
                MAT = partial(
                    transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform,  # noqa:E501
                    features=features,
                    context_features=context,
                    hidden_features=64,
                    num_blocks=2,
                    use_residual_blocks=False,
                    tails="linear",
                    tail_bound=5.0,
                )

            ts = []

            for _ in range(self.transforms):
                ts.extend([MAT(), transforms.ReversePermutation(features)])

            transform = transforms.CompositeTransform(ts)
            base = distributions.StandardNormal(shape=[features])

            return flows.Flow(transform=transform, distribution=base)

        npe = SNPE(self.prior, density_estimator=build)
        npe.append_simulations(self.theta, self.x)

        with dump():
            self.npe = npe.train(
                validation_fraction=1 / len(self.theta),
                max_num_epochs=n_iter + 1,
                stop_after_epochs=n_iter + 1,
                training_batch_size=128,
                learning_rate=1e-3,
            )

    def get_result(self):
        return (
            lambda theta, x: self.npe.log_prob(theta, x),
            lambda x, n: self.npe.sample(n, x[None]).squeeze(0).detach(),
        )
