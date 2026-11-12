"""Startup warm-up shared by direct and remote (policy-server) inference.

Before execution starts, a few discarded queries run on one captured startup
observation so the first real query is not cold (CUDA/cuDNN initialisation,
allocator growth, server connection). Each backend supplies its own query.
"""

DEFAULT_WARMUP_STEPS = 3


def warm_up(query, steps=DEFAULT_WARMUP_STEPS, *, check=None):
    """Run ``steps`` discarded queries; ``query(index)`` returns its timing dict.

    The timing dict carries ``query_total`` in seconds. ``check(timing)``, if
    given, validates each query (for example the async overlap budget).
    Returns the timings.
    """
    if steps < 0:
        raise ValueError('warm-up steps must be nonnegative')
    timings = []
    for index in range(steps):
        timing = query(index)
        if check is not None:
            check(timing)
        timings.append(timing)
    if timings:
        print(f"Policy warm-up: {steps} queries, last {timings[-1]['query_total'] * 1000:.0f} ms")
    return timings
