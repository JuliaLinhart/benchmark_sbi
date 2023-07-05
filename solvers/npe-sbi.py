from benchopt import BaseSolver, safe_import_context
from benchmark_utils.typing import Distribution, Tensor

with safe_import_context() as import_ctx:
    from sbi.inference import SNPE
    from sbi.utils.get_nn_models import posterior_nn

    from benchmark_utils.common import dump


class Solver(BaseSolver):
    r"""Neural Posterior Estimator.

    Solver implemented using the sbi package.

    Trains a parametric conditional distribution to approximate the posterior distribution of parameters.
    The optimization problem is to minimize the Kullback-Leibler divergence between the distribution of the observations and the distribution of the flow.

    References:
        | Fast :math:`\espilon`-free Inference of Simulation Models with Bayesian Conditional Density Estimation (Papamakarios & Murray, 2016)
        | https://arxiv.org/abs/1605.06376

        | Automatic posterior transformation for likelihood-free inference (Greenberg, Nonnenmacher & Macke,. 2019)
        | https://arxiv.org/abs/1905.07488
    """
    name = "NPE-SBI"
    # stopping_strategy = "default"
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
