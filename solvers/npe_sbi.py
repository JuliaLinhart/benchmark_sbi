from benchopt import BaseSolver, safe_import_context
from benchmark_utils.typing import Distribution, Tensor

with safe_import_context() as import_ctx:
    from sbi.inference import SNPE
    from sbi.utils.get_nn_models import posterior_nn

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
    """

    name = "npe_sbi"
    parameters = {
        "flow": ["maf", "nsf"],
        "transforms": [1, 3, 5],
    }

    def get_next(self, n_iter: int) -> int:
        return int(max(n_iter + 10, n_iter * 1.5))

    def set_objective(self, theta: Tensor, x: Tensor, prior: Distribution):
        self.theta, self.x, self.prior = theta, x, prior

    def run(self, n_iter: int):
        estimator = posterior_nn(
            self.flow,
            num_transforms=self.transforms,
            use_random_permutations=False,
        )

        npe = SNPE(self.prior, density_estimator=estimator)
        npe.append_simulations(self.theta, self.x)

        with dump():
            self.npe = npe.train(
                validation_fraction=0.1,
                max_num_epochs=n_iter + 1,
                training_batch_size=128,
                learning_rate=1e-3,
            )

    def get_result(self):
        return (
            lambda theta, x: self.npe.log_prob(theta, x),
            lambda x, n: self.npe.sample(n, x[None]).squeeze(0).detach(),
        )
