from swissclim_evaluations.helpers import build_output_filename


def test_tokens_basic_mapping():
    assert build_output_filename(metric="m", ensemble=None) == "m_ensmean.csv"
    assert build_output_filename(metric="m", ensemble="mean") == "m_ensmean.csv"
    assert build_output_filename(metric="m", ensemble="enspooled") == "m_enspooled.csv"
    assert build_output_filename(metric="m", ensemble="ensprob") == "m_ensprob.csv"
    assert build_output_filename(metric="m", ensemble=2) == "m_ens2.csv"


def test_members_token_passthrough():
    # Already formatted ens<idx> string should pass through directly
    assert build_output_filename(metric="m", ensemble="ens7") == "m_ens7.csv"
