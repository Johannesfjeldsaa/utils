import inspect
from typing import get_type_hints, Union, get_origin, get_args

def type_check_decorator(func):
    """
    Decorator [1] for type checking of user input to functions at runtime.
    Known limitations:
        - Does not support parameterized typechecking as it is based on isinstance [3]. That is does not support
          hints like list[int] or numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], etc. Therefor
          aliases like numpy.typing.ArrayLike are not supported. The workaround is to be more specific with the type hints
          and use things like Union[int, list, numpy.ndarray].

    Parameters
    ----------
    func : Callable
        A function with implemented type hints [2].
    """

    def wrapper(*args, **kwargs):
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        user_passed_args = bound_args.arguments.items()

        for name, value in user_passed_args:
            if name in type_hints:
                expected_type = type_hints[name]
                origin = get_origin(expected_type)
                args_ = get_args(expected_type)

                if origin is Union:
                    # Remove NoneType from args for Optional
                    valid_types = tuple(arg for arg in args_ if arg is not type(None))
                    if value is not None and not isinstance(value, valid_types):
                        raise TypeError(
                            f"Argument '{name}' must be of type {expected_type}, but got {type(value)}"
                        )
                else:
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Argument '{name}' must be of type {expected_type}, but got {type(value)}"
                        )
        return func(*args, **kwargs)
    return wrapper