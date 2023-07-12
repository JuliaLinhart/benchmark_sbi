from benchopt import BaseDataset, safe_import_context
from benchmark_utils.typing import Distribution, Tensor
from typing import Dict

with safe_import_context() as import_ctx:
    import lampe
    import torch

    from benchmark_utils.common import fork
    from torch.distributions import MultivariateNormal, Independent, Uniform
    from tqdm import tqdm


class Dataset(BaseDataset):
    """Simple likelihood complex posterior (SLCP) dataset."""

    name = "slcp"
    parameters = {
        "train_size": [4096, 16384],
        "test_size": [256],
        "ref_size": [16],
        "n_per_ref": [1024],
        "seed": [42],
    }

    install_cmd = "conda"
    requirements = [
        "pytorch",
        "pip:lampe",
    ]

    def prior(self):
        r"""p(theta)"""

        low = torch.full((5,), -3.0)
        high = torch.full((5,), 3.0)

        return Independent(Uniform(low, high), 1)

    def likelihood(self, theta: Tensor, eps: float = 1e-8) -> Distribution:
        r"""p(x | theta)"""

        # Mean
        mu = theta[..., :2]

        # Covariance
        s1 = theta[..., 2] ** 2 + eps
        s2 = theta[..., 3] ** 2 + eps
        rho = theta[..., 4].tanh()

        cov = torch.stack(
            [
                s1**2,
                rho * s1 * s2,
                rho * s1 * s2,
                s2**2,
            ],
            dim=-1,
        ).reshape(theta.shape[:-1] + (2, 2))

        # Repeat
        mu = mu.unsqueeze(-2).repeat_interleave(4, -2)
        cov = cov.unsqueeze(-3).repeat_interleave(4, -3)

        # Normal
        normal = MultivariateNormal(mu, cov)

        return Independent(normal, 1)

    def simulator(self, theta: Tensor) -> Tensor:
        r"""x ~ p(x | theta)"""

        return self.likelihood(theta).sample()

    def get_data(self) -> Dict:
        with fork():
            torch.manual_seed(self.seed)

            prior = self.prior()
            simulator = self.simulator

            theta_train = prior.sample((self.train_size,))
            x_train = simulator(theta_train).flatten(1)

            theta_test = prior.sample((self.test_size,))
            x_test = simulator(theta_test).flatten(1)

            if self.ref_size > 0:
                theta_ref = prior.sample((self.ref_size,))
                x_ref = simulator(theta_ref)

                theta_ref = []

                for i in tqdm(range(self.ref_size), desc="Posterior sampling"):
                    theta_0 = prior.sample((self.n_per_ref,))
                    x = x_ref[i]

                    def log_joint(theta: Tensor) -> Tensor:
                        log_likelihood = self.likelihood(theta).log_prob(x)
                        log_prior = prior.log_prob(theta)

                        return log_likelihood + log_prior

                    sampler = lampe.inference.MetropolisHastings(
                        theta_0, log_f=log_joint
                    )
                    samples = next(sampler(4096 + 1, burn=4096))
                    theta_ref.append(samples)

                x_ref = x_ref.flatten(1)
            else:
                theta_ref = None
                x_ref = None

            return dict(
                prior=prior,
                theta_train=theta_train,
                x_train=x_train,
                theta_test=theta_test,
                x_test=x_test,
                theta_ref=theta_ref,
                x_ref=x_ref,
            )
