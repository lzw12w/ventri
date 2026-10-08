import pytest

from . import cfg_plugins


@pytest.fixture(autouse=True)
def _reset_plugins():
    cfg_plugins.INSTANCES.clear()
    cfg_plugins.PORTS.clear()
    cfg_plugins.FLAKY["fail"] = True
    yield
