
import functools
import inspect
from typing import Dict, Type, TypeVar


class Injector:
    def __init__(self, parent: 'Injector' = None):
        self._dependencies: Dict[type, object] = {}

        self._parent: Injector = parent

        self._previous_injector = None

    def bind(self, dependency_type: Type, to: object, overwrite: bool = False):
        """
        Register a dependency by type.
        """
        if not overwrite and dependency_type in self._dependencies:
            raise ValueError(f"Dependency {dependency_type} already registered")
        self._dependencies[dependency_type] = to

    T = TypeVar("T")

    def resolve(self, type: T) -> T:
        """
        Resolve a dependency by name.
        """
        return self._dependencies.get(type) \
            or (self._parent and self._parent.resolve(type))

    def has(self, type: type) -> bool:
        """
        Check if a dependency is registered.
        """
        return type in self._dependencies \
            or (self._parent and self._parent.has(type))

    def __enter__(self):
        self._previous_injector = get_injector()
        set_injector(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_injector(self._previous_injector)
        self._previous_injector = None


_injector = Injector()


def set_injector(injector: Injector):
    """
    Set the global injector instance.
    """
    global _injector

    if injector is None:
        raise ValueError("Injector cannot be None")

    _injector = injector


def get_injector() -> Injector:
    """
    Get the global injector instance.
    """
    return _injector


def inject(func):
    """
    Decorator to inject dependencies.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        injector = get_injector()

        for name, (ix, type) in func._injectable_args.items():
            if name not in kwargs and injector.has(type):
                dependency = injector.resolve(type)
                if ix < len(args):
                    args = args[:ix] + (dependency,) + args[ix:]
                else:
                    kwargs[name] = dependency

        return func(*args, **kwargs)

    sig = inspect.signature(func)

    typed_args = filter(
        lambda p: p[1].annotation not in (inspect.Parameter.empty, str,
                                          int, float, bool),
        enumerate(sig.parameters.values()))
    func._injectable_args = dict(
        map(lambda p: (p[1].name, (p[0], p[1].annotation)), typed_args))

    if inspect.iscoroutinefunction(func):
        return inspect.markcoroutinefunction(wrapper)
    return wrapper
