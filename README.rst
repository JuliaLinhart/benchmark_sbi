
My Benchopt Benchmark
=====================
|Build Status| |Python 3.8+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms. This benchmark is dedicated to simulation-based inference (SBI) algorithms. The goal of SBI is to approximate the posterior distribution of a stochastic model.

Formally, a stochastic model takes (a vector of) parameters $\\theta \\in \\Theta$ as input, samples internally a series $z \\in \\mathcal{Z}$ of latent variables and, finally, produces an observation $x \\in \\mathcal{X} \\sim p(x | \\theta, z)$ as output, thereby defining an implicit likelihood $p(x | \\theta)$. This likelihood is typically intractable as it corresponds to the integral of the joint likelihood $p(x, z | \\theta)$ over all possible trajectories through the latent space $\\mathcal{Z}$. Moreover, in Bayesian inference, we are interested in the posterior distribution

$$p(\\theta | x) = \\frac{p(x | \\theta) p(\\theta)}{p(x)} = \\frac{p(x | \\theta) p(\\theta)}{\\int_\\Theta p(x | \\theta') p(\\theta') d\\theta'}$$

for some observation $x$ and a prior distribution $p(\\theta)$, which not only involves the intractable likelihood $p(x | \\theta)$ but also an intractable integral over the parameter space $\\Theta$.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/JuliaLinhart/benchmark_sbi
   $ benchopt run benchmark_sbi

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_sbi -s npe -d slcp --n-repetitions 3

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/JuliaLinhart/benchmark_sbi/workflows/Tests/badge.svg
   :target: https://github.com/JuliaLinhart/benchmark_sbi/actions
.. |Python 3.8+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-380/
