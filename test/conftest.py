import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fix_seed():
    np.random.seed(42)
