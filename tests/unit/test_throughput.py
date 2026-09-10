from troiani_platform.jobs.skypilot import spec_from_task_yaml
from troiani_platform.training.throughput import (
    CATALOG,
    DEFAULT_SEQ,
    estimate_fit,
    estimate_matrix,
    format_estimate_table,
    get_arch,
    parse_size,
)


def test_default_seq_is_2048():
    assert DEFAULT_SEQ == 2048


def test_350m_batch1_fits_40_and_80():
    arch = get_arch("350M")
    assert estimate_fit(arch, batch=1, seq=DEFAULT_SEQ, gpu_gb=40).fits
    assert estimate_fit(arch, batch=1, seq=DEFAULT_SEQ, gpu_gb=80).fits


def test_2b_large_batch_fails_40gb():
    arch = get_arch("2B")
    row = estimate_fit(arch, batch=16, seq=DEFAULT_SEQ, gpu_gb=40)
    assert row.fits is False
    assert row.verdict == "NO-FIT"
    assert row.total_gib > row.usable_gib


def test_2b_large_batch_fits_80gb_headroom():
    arch = get_arch("2B")
    row = estimate_fit(arch, batch=16, seq=DEFAULT_SEQ, gpu_gb=80)
    assert row.total_gib > 40
    assert row.fits is True


def test_catalog_param_classes():
    counted = {name: arch.n_params for name, arch in CATALOG.items()}
    assert 300_000_000 < counted["350M"] < 400_000_000
    assert 450_000_000 < counted["500M"] < 560_000_000
    assert 700_000_000 < counted["777M"] < 850_000_000
    assert 900_000_000 < counted["1B"] < 1_100_000_000
    assert 1_400_000_000 < counted["1.5B"] < 1_600_000_000
    assert 1_800_000_000 < counted["2B"] < 2_100_000_000
    for arch in CATALOG.values():
        assert arch.hidden % arch.n_heads == 0


def test_parse_size_aliases():
    assert parse_size("1b").name == "1B"
    assert parse_size("1500M").name == "1.5B"
    assert parse_size("0.35B").name == "350M"


def test_matrix_marks_2b_batch16_40_nofit_and_350m_fit():
    rows = estimate_matrix(
        sizes=("350M", "2B"), batches=(1, 16), gpu_sizes=(40, 80), seq=DEFAULT_SEQ
    )
    by = {(r.name, r.batch, int(r.gpu_gb)): r for r in rows}
    assert by[("350M", 1, 40)].fits
    assert by[("350M", 1, 80)].fits
    assert not by[("2B", 16, 40)].fits
    table = format_estimate_table(rows)
    assert "NO-FIT" in table
    assert "350M" in table


def test_bench_yaml_is_argv_not_shell():
    spec = spec_from_task_yaml(
        {
            "name": "bench-throughput",
            "gpus": 1,
            "min_gpu_memory_gb": 40,
            "command": [
                "python",
                "-m",
                "troiani_platform.training.throughput",
                "--params",
                "350M",
                "--batch",
                "1",
                "--seq",
                "2048",
            ],
        }
    )
    assert spec["command"][0] == "python"
    assert spec["command"][2] == "troiani_platform.training.throughput"
    assert "sh" not in spec["command"]
    assert spec["min_gpu_memory_gb"] == 40
