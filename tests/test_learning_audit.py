"""Independent diagnostic metrics must catch sign, alignment and constant predictions."""

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_learning_diagnostics import independent_daily

# independent_daily now re-scores the saved predictions through the single eval-v2
# contract, so it requires the archived run's evaluation config explicitly. Keeping
# the argument required avoids a hidden local default for the evaluation caliber.
EVALUATION = {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3}


def test_independent_daily_recovers_known_reversed_and_constant_rankings():
    date = pd.Timestamp("2023-07-18 14:30")
    keys = pd.DataFrame({"date": [date] * 4, "instrument": list("ABCD")})
    targets = keys.assign(target=[-2., -1., 1., 2.])
    for scores, expected in (([-2., -1., 1., 2.], 1.), ([2., 1., -1., -2.], -1.)):
        output, _ = independent_daily(
            keys.assign(score=scores).iloc[::-1], targets, EVALUATION)
        assert output.iloc[0].rank_ic == pytest.approx(expected)
        assert output.iloc[0].ic == pytest.approx(expected)
    output, _ = independent_daily(keys.assign(score=0.), targets, EVALUATION)
    assert np.isnan(output.iloc[0].rank_ic) and np.isnan(output.iloc[0].ic)
    with pytest.raises(ValueError):
        independent_daily(keys.iloc[:3].assign(score=0.), targets, EVALUATION)
    with pytest.raises(ValueError):
        independent_daily(pd.concat([keys, keys]).assign(score=0.), targets, EVALUATION)
