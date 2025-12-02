from swissclim_evaluations.helpers import get_colormap_for_variable


def test_diverging_variable_colormap():
    assert get_colormap_for_variable("u_component_of_wind") == "RdBu_r"
    assert get_colormap_for_variable("vertical_velocity") == "RdBu_r"
    assert get_colormap_for_variable("integrated_vapor_transport") == "RdBu_r"


def test_precipitation_variable_colormap():
    assert get_colormap_for_variable("total_precipitation") == "Blues"
    assert get_colormap_for_variable("snow_depth") == "Blues"


def test_temperature_variable_colormap():
    assert get_colormap_for_variable("2m_temperature") == "magma"
    assert get_colormap_for_variable("sea_surface_temperature") == "magma"


def test_case_insensitive():
    assert get_colormap_for_variable("TEMPERATURE") == "magma"
    assert get_colormap_for_variable("Precipitation") == "Blues"


def test_unknown_variable_returns_default():
    assert get_colormap_for_variable("unknown_var") == "viridis"
    assert get_colormap_for_variable("") == "viridis"


def test_ambiguous_variables_order():
    # "integrated_vapor_transport" contains "vapor" (precipitation) but is in diverging list
    # It should be caught by the diverging check first
    assert get_colormap_for_variable("integrated_vapor_transport") == "RdBu_r"
