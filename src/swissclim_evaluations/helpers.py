import itertools

"""Helper utilities for chunking over init and lead times."""


def time_chunks(
    init_times, lead_times, init_time_chunk_size=None, lead_time_chunk_size=None
):
    # Accept non-contiguous init_times; just slice by chunk size without assuming uniform spacing
    try:
        init_times = init_times.astype("datetime64[ns]")
    except Exception:
        pass
    total_init = len(init_times)
    step_i = init_time_chunk_size or total_init
    init_time_chunks = [
        init_times[i : i + step_i] for i in range(0, total_init, step_i)
    ]

    if isinstance(lead_times, slice):
        lead_time_chunks = [lead_times]
    else:
        try:
            lead_times = lead_times.astype("timedelta64[ns]")
        except Exception:
            pass
        total_lead = len(lead_times)
        step_l = lead_time_chunk_size or total_lead
        lead_time_chunks = [
            lead_times[i : i + step_l] for i in range(0, total_lead, step_l)
        ]

    return itertools.product(init_time_chunks, lead_time_chunks)


## Note: patch_time_dimensions was intentionally removed to keep library API unchanged.
