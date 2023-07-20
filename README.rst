Simulation-based Inference Benchmark
====================================
|Build Status| |Python 3.8+|

Benchopt is a package to simplify and make more transparent and reproducible the comparisons of optimization algorithms. This benchmark is dedicated to simulation-based inference (SBI) algorithms. The goal of SBI is to approximate the posterior distribution of a stochastic model (or simulator):

$$q_{\\phi}(\\theta \\mid x) \\approx p(\\theta | x) = \\frac{p(x \\mid \\theta) p(\\theta)}{p(x)}$$

where $\\theta$ denotes the model parameters and $x$ is an observation. In SBI the likelihood $p(x \\mid \\theta)$ is implicitly modeled by the stochastic simulator. Placing a prior $p(\\theta)$ over the simulator parameters, allows us to generate samples from the joint distribution $p(\\theta, x) = p(x \\mid \\theta) p(\\theta)$ which can then be used to approximate the posterior distribution $p(\\theta \\mid x)$, e.g. via the training of a deep generative model $q_{\\phi}(\\theta | x)$.

In this benchmark we only consider amortized SBI algorithms that allow for inference for any new observation $x$, without simulating new data after the initial training phase.

Installation
------------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/JuliaLinhart/benchmark_sbi
   $ benchopt run benchmark_sbi

Alternatively, options can be passed to ``benchopt run`` to restrict the runs to some solvers or datasets:

.. code-block::

	$ benchopt run benchmark_sbi -s npe_lampe -d slcp --n-repetitions 3

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

Contributing
------------

Everyone is welcome to contribute by adding datasets, solvers (algorithms) or metrics.

* Datasets represent different prior-simulator pairs that define a joint distribution $p(\\theta, x) = p(\\theta) p(x \\mid \\theta)$. 

	The data they are expected to return (``Dataset.get_data``) consist in a set of training parameters-observation pairs $(\\theta, x)$, a set of testing parameters-observation pairs $(\\theta, x)$ and and a set of reference posterior-observation pairs $(p(\\theta \\mid x), x)$.

	To add a dataset, add a file in the `datasets` folder.

* Solvers represent different amortized SBI algorithms (NRE, NPE, FMPE, ...) or different implementations (``sbi``, ``lampe``, ...) of such algorithms. 

	They are initialized (``Solver.set_objective``) with the training pairs and the prior $p(\\theta)$. After training (``Solver.run``), they are expected to return (``Solver.get_result``) a pair of functions ``log_prob`` and ``sample`` that evaluate the posterior log-density $\\log q_{\\phi}(\\theta \\mid x)$ and generate parameters $\\theta \\sim q_{\\phi}(\\theta \\mid x)$, respectively.

	To add a solver, add a file in the `solvers` folder.

* Metrics evaluate the quality of the estimated posterior obtained from the solver. 

	The main objective is the expected negative log-likelihood $\\mathbb{E}_{p(\\theta, x)} [ - \\log q_{\\phi}(\\theta \\mid x) ]$ over the test set. Other metrics such as the C2ST and EMD scores are computed (``Objective.compute``) using the reference posteriors (if available).

	To add a metric, implement it in the `benmark_utils.metrics.py` file.

.. |Build Status| image:: https://github.com/JuliaLinhart/benchmark_sbi/workflows/Tests/badge.svg
   :target: https://github.com/JuliaLinhart/benchmark_sbi/actions
.. |Python 3.8+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-380/
