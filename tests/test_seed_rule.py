"""Regression for the Series identity comparison that silently rejected all seeds."""

import numpy as np
import pandas as pd

from copick_helix.bands import TermConfig, _seed_mask


def test_seed_rule_accepts_eligible_filaments_and_rejects_missing_halves():
    table = pd.DataFrame(
        {
            "z": [3.0, 4.0, 4.0, 2.99, 4.0, np.nan],
            "halves_agree": [True, False, None, True, True, True],
            "stability": [0.8, 1.0, 1.0, 1.0, 0.79, 1.0],
        },
        index=[11, 12, 13, 14, 15, 16],
    )
    mask = _seed_mask(table, TermConfig())
    assert mask.index.equals(table.index)
    assert mask.tolist() == [True, False, False, False, False, False]


def test_nullable_halves_are_not_seeds():
    table = pd.DataFrame(
        {"z": [4.0, 4.0], "halves_agree": pd.array([True, pd.NA], dtype="boolean"), "stability": [1.0, 1.0]},
    )
    assert _seed_mask(table, TermConfig()).tolist() == [True, False]
