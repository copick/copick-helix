from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from copick_helix import cli


@pytest.mark.parametrize("extra,expected", [([], 12.0), (["--term-axis-shift", "24"], 24.0)])
def test_term_axis_search_is_configurable_without_changing_default(monkeypatch, tmp_path, extra, expected):
    config = tmp_path / "config.json"
    config.write_text("{}")
    family = SimpleNamespace(lattice_analysis=True, route="terms", term_segment_length=760.0)
    monkeypatch.setattr(cli, "get_family", lambda _: family)
    monkeypatch.setattr(cli, "_straighten_all", lambda *a, **kw: (None, {}))
    received = []
    monkeypatch.setattr(cli, "_polarity_terms", lambda *a, **kw: received.append(a[-1]))
    result = CliRunner().invoke(
        cli.polarity,
        ["-c", str(config), "-i", "actin:test/v1", "-t", "test@10", "--work-dir", str(tmp_path), *extra],
    )
    assert result.exit_code == 0, result.output
    assert received == [expected]
