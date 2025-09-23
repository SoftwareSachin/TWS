# Reference http://glenfant.github.io/flask-g-object-for-fastapi.html
import contextvars
import types

request_global = contextvars.ContextVar("request_global", default=None)


# This is the only public API
def g():
    value = request_global.get()
    if value is None:
        value = types.SimpleNamespace(blah=1)
        request_global.set(value)
    return value
