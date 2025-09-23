import types

import pytest

from app.utils.requestvars import g, request_global


# Reset the context variable before each test
@pytest.fixture(autouse=True)
def reset_request_global():
    # Clear the context variable for isolation between tests
    request_global.set(types.SimpleNamespace(blah=1))
    yield


def test_g_default_value():
    assert g().blah == 1


def test_g_set_value():
    assert g().blah == 1
    new_value = types.SimpleNamespace(new_val=42)
    request_global.set(new_value)
    assert g().new_val == 42
