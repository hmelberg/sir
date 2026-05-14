from sir.constants import (
    AGE_BIN_EDGES,
    AGE_BIN_LABELS,
    DEFAULT_V_A,
    GroupType,
    DiseaseState,
)


def test_age_bins_have_seven_decadal_groups():
    assert len(AGE_BIN_LABELS) == 7
    assert AGE_BIN_LABELS == ("0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+")
    # edges: 0, 10, 20, 30, 40, 50, 60, +inf
    assert AGE_BIN_EDGES[0] == 0
    assert AGE_BIN_EDGES[-1] == float("inf")
    assert len(AGE_BIN_EDGES) == 8


def test_default_v_a_has_one_value_per_age_bin():
    assert len(DEFAULT_V_A) == 7
    assert DEFAULT_V_A == (0.5, 0.5, 1.0, 2.0, 3.0, 5.0, 30.0)


def test_group_type_enum_has_seven_members():
    assert {gt.name for gt in GroupType} == {
        "HOUSEHOLD", "KIN", "SCHOOL", "WORKPLACE",
        "RECURRING_LEISURE", "ONE_OFF_EVENT", "COMMUNITY",
    }


def test_disease_state_enum_has_sirv():
    assert {ds.name for ds in DiseaseState} == {"S", "V", "I", "R"}
    assert DiseaseState.S.value == 0
    assert DiseaseState.V.value == 1
    assert DiseaseState.I.value == 2
    assert DiseaseState.R.value == 3
