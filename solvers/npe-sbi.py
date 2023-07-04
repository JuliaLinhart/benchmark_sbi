from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sbi.inference import SNPE
    from torch import Tensor
    from torch.distributions import Distribution

    from benchmark_utils import dump


class Solver(BaseSolver):
    name = "NPE-SBI"
    # stopping_strategy = "callback"
    parameters = {
        # "transforms": [1, 3, 5],
    }

    def set_objective(self, theta: Tensor, x: Tensor, prior: Distribution):
        self.theta, self.x, self.prior = theta, x, prior

    def run(self, n_iter: int):
        npe = SNPE(self.prior)
        npe.append_simulations(self.theta, self.x)

        with dump():
            self.flow = npe.train(
                validation_fraction=0.1,
                max_num_epochs=n_iter + 1,
                training_batch_size=128,
                learning_rate=1e-3,
            )

    def get_result(self):
        return (
            lambda theta, x: self.flow.log_prob(theta, x),
            lambda x, n: self.flow.sample(n, x),
        )
