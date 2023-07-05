
My Benchopt Benchmark
=====================
|Build Status| |Python 3.8+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms. This benchmark is dedicated to simulation-based inference (SBI) algorithms. The goal of SBI is to approximate the posterior distribution of a stochastic model:

$$q(\\theta | x) \\approx p(\\theta | x) \\propto p(\\theta)p(x | \\theta)$$

where $\\theta$ denotes the model-parameters and $x$ is an observation. In SBI the likelihood $p(x | \\theta)$ is implicitly modeled by the stochastic simulator. 
By placing a prior $p(\\theta)$ over the simulator-parameters, SBI-algorithms use samples from the joint distribution $p(\\theta, x)$ to approximate the posterior distribution

$$p(\\theta | x) = \\frac{p(x | \\theta) p(\\theta)}{p(x)} = \\frac{p(x | \\theta) p(\\theta)}{\\int_\\Theta p(x | \\theta') p(\\theta') d\\theta'}$$

which not only involves the typically intractable likelihood $p(x | \\theta)$ (black box simulator), but also an intractable integral over the parameter space.

In this benchmark we consider only amortized SBI-algorithms, i.e. that allow for a quick inference procedure for any new observation $x$ after a one-time training phase.

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
