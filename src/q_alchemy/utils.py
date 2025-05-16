from scipy.sparse import coo_matrix


def is_power_of_two(state_vector: coo_matrix) -> bool:
    length = state_vector.shape[1]
    return length > 0 and (length & (length - 1)) == 0
