from scipy.sparse import coo_matrix


def is_power_of_two(state_vector: coo_matrix) -> bool:
    length = state_vector.shape[1]
    return length > 0 and (length & (length - 1)) == 0


def reject_unknown_options(kind: str, given, known) -> None:
    """Raise TypeError naming every option in given that is not in known, with a
    suggestion for a near miss.

    Options used to be filtered to the known ones, so a misspelt option was
    dropped and the call ran with its default, with no sign why.
    """
    import difflib

    known = set(known)
    unknown = sorted(set(given) - known)
    if unknown:
        hints = [
            f"'{name}' (did you mean '{match[0]}'?)" if (match := difflib.get_close_matches(name, known, n=1))
            else f"'{name}'"
            for name in unknown
        ]
        raise TypeError(f"Unknown {kind} option(s): {', '.join(hints)}.")
