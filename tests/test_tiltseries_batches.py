from copick_helix.tiltseries import _reconstruction_jobs


def segments(tmp_path, name, count):
    return [{"stem": str(tmp_path / f"{name}_{i}"), "segment": i} for i in range(count)]


def test_single_tomogram_distributes_every_segment_exactly_once(tmp_path):
    segs = segments(tmp_path, "run", 522)
    source = object()
    jobs = _reconstruction_jobs({"run": segs}, source, 190, 30, 4)
    assert [len(j[1]) for j in jobs] == [131, 131, 131, 129]
    assert [s for job in jobs for s in job[1]] == segs
    assert all(j[0] == "run" and j[2:] == (source, 190, 30) for j in jobs)


def test_cached_segments_are_excluded_before_balancing(tmp_path):
    segs = segments(tmp_path, "run", 100)
    for s in segs[:90]:
        (tmp_path / (s["stem"].split("/")[-1] + ".npy")).touch()
    jobs = _reconstruction_jobs({"run": segs}, None, 190, 30, 4)
    assert len(jobs) == 1
    assert jobs[0][1] == segs[90:]
    assert _reconstruction_jobs({"run": segs[:90]}, None, 190, 30, 4) == []


def test_batches_never_mix_tomograms_or_split_small_serial_jobs(tmp_path):
    by = {"a": segments(tmp_path, "a", 10), "b": segments(tmp_path, "b", 20)}
    for workers in (1, 8):
        jobs = _reconstruction_jobs(by, None, 190, 30, workers)
        assert [(j[0], j[1]) for j in jobs] == list(by.items())
