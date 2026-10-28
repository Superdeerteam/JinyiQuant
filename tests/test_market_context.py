"""Behavioural tests for the market-context heads and the same-timestamp market summary.

Offline: no bars, no labels, no checkpoint. Each test pins one property that the acceptance
checks and the replay path rely on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from modified_tlob import market_head
from modified_tlob.market_head import (CONTEXT_DIM, REPRESENTATION_DIM, build_head,
                                      cross_section_floor, decode_state, head_from_payload,
                                      head_payload, market_context, parameter_count, text_state,
                                      timestamp_codes)


def toy_rows():
    """Two timestamps: rows 0-3 are one cross-section, rows 4-6 the next."""
    keys = pd.DataFrame({"date": pd.to_datetime(
        ["2023-11-01 09:31", "2023-11-01 09:31", "2023-11-01 09:31", "2023-11-01 09:31",
         "2023-11-01 09:32", "2023-11-01 09:32", "2023-11-01 09:32"]),
        "instrument": ["000001", "000002", "000003", "000004", "000001", "000002", "000003"]})
    generator = np.random.default_rng(7)
    return keys, generator.normal(size=(len(keys), 4)).astype(np.float64)


def test_market_context_summarises_mean_std_and_coverage():
    keys, hidden = toy_rows()
    order, codes, groups, counts = timestamp_codes(keys)
    context = market_context(hidden[order], codes, counts=counts, n_groups=groups)
    assert context.shape == (groups, 2 * hidden.shape[1] + 1)
    first = hidden[order][codes == 0]
    np.testing.assert_allclose(context[0, :4], first.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(context[0, 4:8], first.std(axis=0, ddof=0), rtol=0, atol=1e-12)
    assert context[0, 8] == 1.0 and context[1, 8] == 1.0
    assert counts.tolist() == [4, 3]


def test_invalid_rows_leave_the_summary_but_stay_in_the_denominator():
    keys, hidden = toy_rows()
    order, codes, groups, counts = timestamp_codes(keys)
    valid = np.array([True, True, False, True, True, False, True])
    context = market_context(hidden[order], codes, valid=valid[order], counts=counts,
                             n_groups=groups)
    kept = hidden[order][(codes == 0) & valid[order]]
    np.testing.assert_allclose(context[0, :4], kept.mean(axis=0), rtol=0, atol=1e-12)
    assert context[0, 8] == pytest.approx(3 / 4)
    assert context[1, 8] == pytest.approx(2 / 3)
    # A row excluded by its history is invisible to the summary whatever value it carries.
    polluted = hidden[order].copy()
    polluted[~valid[order]] = 1e6
    after = market_context(polluted, codes, valid=valid[order], counts=counts, n_groups=groups)
    np.testing.assert_array_equal(after, context)


def test_market_context_is_order_independent():
    keys, hidden = toy_rows()
    shuffled = np.random.default_rng(0).permutation(len(keys))
    left = timestamp_codes(keys)
    right = timestamp_codes(keys.iloc[shuffled].reset_index(drop=True))
    left_context = market_context(hidden[left[0]], left[1], counts=left[3], n_groups=left[2])
    right_context = market_context(hidden[shuffled][right[0]], right[1], counts=right[3],
                                   n_groups=right[2])
    np.testing.assert_array_equal(left_context, right_context)


@pytest.mark.parametrize("arm", ["B", "C"])
def test_zero_initialised_head_reproduces_the_base_score(arm):
    torch.manual_seed(0)
    head = build_head(arm).eval()
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(17, REPRESENTATION_DIM, generator=generator)
    base = torch.randn(17, generator=generator, dtype=torch.float32) * 3.0
    context = torch.randn(17, CONTEXT_DIM, generator=generator)
    delta = head.delta(hidden, context) if arm == "C" else head.delta(hidden)
    assert float(delta.detach().abs().max()) == 0.0
    assert torch.equal(base + delta, base)
    assert parameter_count(head) > 10_000


def test_residual_of_a_row_ignores_every_other_row():
    """`delta` must not reduce across rows: the replay slice size cannot leak into a score."""
    torch.manual_seed(0)
    head = build_head("C").eval()
    with torch.no_grad():
        head.mix.weight.normal_()
    generator = torch.Generator().manual_seed(2)
    hidden = torch.randn(9, REPRESENTATION_DIM, generator=generator)
    context = torch.randn(9, CONTEXT_DIM, generator=generator)
    whole = head.delta(hidden, context)
    scrambled = hidden.clone()
    scrambled[1:] += 1.0
    other_context = context.clone()
    other_context[1:] -= 2.0
    assert torch.equal(head.delta(scrambled, other_context)[:1], whole[:1])
    assert whole.shape == (9,)


def test_both_arms_have_comparable_parameter_counts():
    counts = {arm: parameter_count(build_head(arm)) for arm in ("B", "C")}
    assert abs(counts["B"] - counts["C"]) / counts["C"] < 0.05, counts


def test_head_payload_round_trip_preserves_the_residual():
    torch.manual_seed(3)
    head = build_head("C").eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_()
    payload = head_payload(head)
    stored = {"arm": payload["arm"], "spec": payload["spec"],
              "state_dict": text_state(payload["state_dict"])}
    restored = head_from_payload(stored, text_state_weights=True)
    generator = torch.Generator().manual_seed(4)
    hidden = torch.randn(5, REPRESENTATION_DIM, generator=generator)
    context = torch.randn(5, CONTEXT_DIM, generator=generator)
    assert torch.equal(restored.delta(hidden, context), head.delta(hidden, context))


def test_head_spec_is_checked_against_the_context_width():
    with pytest.raises(ValueError):
        build_head("C", ctx_dim=REPRESENTATION_DIM)
    assert parameter_count(build_head("C", hidden=REPRESENTATION_DIM)) > 10_000
    with pytest.raises(ValueError):
        build_head("A")


def test_representation_feeds_the_frozen_scoring_chain():
    from modified_tlob.model import MLPLOB

    torch.manual_seed(5)
    network = MLPLOB(num_features=12, seq_size=128).eval()
    assert network.representation_dim == REPRESENTATION_DIM
    x = torch.randn(4, 128, 12)
    with torch.inference_mode():
        score, representation = network.forward_with_representation(x)
        reference = network.final_layers[-1](network.represent(x))[:, 0]
    assert representation.shape == (4, REPRESENTATION_DIM)
    assert torch.equal(representation, network.represent(x))
    assert torch.equal(score, reference)
    assert torch.equal(score, network(x))


def test_text_state_round_trips_float32_exactly():
    weights = {"project.weight": torch.randn(16, REPRESENTATION_DIM)}
    back = decode_state(text_state(weights), True)
    assert torch.equal(back["project.weight"], weights["project.weight"])


def test_a_nonfinite_row_cannot_poison_its_timestamp():
    keys, hidden = toy_rows()
    order, codes, groups, counts = timestamp_codes(keys)
    clean = market_context(hidden[order], codes, counts=counts, n_groups=groups)
    polluted = hidden[order].copy()
    polluted[0, 2] = np.nan
    after = market_context(polluted, codes, counts=counts, n_groups=groups)
    assert np.isfinite(after).all()
    assert after[1, :] == pytest.approx(clean[1, :])
    assert after[0, 8] == pytest.approx(3 / 4)


def test_a_rechunked_batch_falls_below_the_cross_section_floor():
    keys, _ = toy_rows()
    _, codes, groups, _ = timestamp_codes(keys)
    assert cross_section_floor(codes, groups) == pytest.approx(3)  # toy cross-section
    _, fragment_codes, fragment_groups, _ = timestamp_codes(keys.iloc[[0, 4]])
    assert cross_section_floor(fragment_codes, fragment_groups) == 1
    assert cross_section_floor(fragment_codes, fragment_groups) < market_head.MIN_CROSS_SECTION


def test_unknown_split_names_are_refused_instead_of_building_nothing(tmp_path):
    from scripts.market_context import requested_splits

    assert requested_splits("train,replay") == ["train", "replay"]
    assert requested_splits("replay,") == ["replay"]
    for value in ("final", "train,final", ""):
        with pytest.raises(SystemExit) as raised:
            requested_splits(value)
        assert "expected a non-empty subset of ['train', 'replay']" in str(raised.value.code)


def test_a_killed_build_is_reported_as_such_not_as_a_missing_file(tmp_path):
    from scripts.market_context import load_cache

    killed = tmp_path / "train" / "parts"
    killed.mkdir(parents=True)
    with pytest.raises(ValueError, match="no completed build found"):
        load_cache(tmp_path / "train")
