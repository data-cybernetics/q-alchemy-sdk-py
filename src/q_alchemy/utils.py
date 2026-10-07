from numbers import Integral

from scipy.sparse import coo_matrix


def nonnegative_integer(value, name: str) -> int:
    """Validate integer counts/indices without rounding or accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def is_power_of_two(state_vector: coo_matrix) -> bool:
    length = state_vector.shape[1]
    return length > 0 and (length & (length - 1)) == 0
