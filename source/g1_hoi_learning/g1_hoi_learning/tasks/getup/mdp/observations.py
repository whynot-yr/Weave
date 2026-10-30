from .state import get_state


def proprio_history(env):
    # Explicit history prevents terminal observation reads from shifting it twice.
    return get_state(env).observation()
