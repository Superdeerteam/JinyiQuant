import pytest

from scripts.annual_baseline import validate_window


def window():
    return {'train_end_exclusive': '2022-01-01', 'evaluation_start': '2022-01-01',
            'evaluation_end_exclusive': '2022-02-01', 'train_dates': list(range(10))}


def test_forward_weights_allowed():
    validate_window(window())


@pytest.mark.parametrize('changes', [
    {'train_end_exclusive': '2022-01-02'},
    {'evaluation_start': '2024-07-01', 'evaluation_end_exclusive': '2024-08-01'},
    {'evaluation_end_exclusive': '2023-01-02'},
    {'train_dates': list(range(9))},
])
def test_backward_weights_and_reserved_periods_rejected(changes):
    with pytest.raises(ValueError):
        validate_window(window() | changes)
