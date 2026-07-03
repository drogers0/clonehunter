def foo():
    # This comment appears before the docstring (comment is not a statement).
    """This IS the docstring; the preceding comment is not a statement."""
    return 42


def bar():
    """Normal docstring, no preceding comment."""
    return 0


def baz():
    # Multiple comments
    # before docstring
    """Still the docstring."""
    x = 1
    return x
