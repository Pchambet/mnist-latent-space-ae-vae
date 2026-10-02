import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_and_seed():
    torch.set_num_threads(2)
    torch.manual_seed(0)
