import pytest


def check_test_solver_install(solver_class):
    if solver_class.name.lower() == 'npe_sbi':
        pytest.skip('`npe_sbi` is not easy to install automatically.')