import pytest

from config import parse_float


@pytest.mark.parametrize('value', ['nan', 'NaN', 'inf', '-inf', '1e999'])
def test_nonfinite_settings_rejected(value):
    with pytest.raises(RuntimeError, match='finite'):
        parse_float(value, 'MIN_SPEECH_RATIO')
