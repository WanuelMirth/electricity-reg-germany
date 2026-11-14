import pandas as pd
import pytest

from src.outliers import add_comparisons, grouped_iqr


def hourly_frame(start, end):
    timestamps = pd.date_range(start, end, freq="h", tz="Europe/Berlin", inclusive="left")
    return pd.DataFrame({"start_local": timestamps, "consumption_mwh": 100.0})


def test_week_comparison_preserves_local_hour_across_spring_shift():
    df = hourly_frame("2023-03-19", "2023-03-28")
    # At 10am last Sunday, distinguish the local-hour match from a 168-hour shift.
    df.loc[df.start_local.eq(pd.Timestamp("2023-03-19 10:00", tz="Europe/Berlin")),
           "consumption_mwh"] = 123.0
    result = add_comparisons(df, "hourly")
    row = result.loc[result.start_local.eq(
        pd.Timestamp("2023-03-26 10:00", tz="Europe/Berlin"))].iloc[0]
    assert row.week_reference_consumption_mwh == 123.0


def test_both_sides_of_ambiguous_autumn_hour_are_excluded_from_week_comparison():
    result = add_comparisons(hourly_frame("2023-10-22", "2023-11-06"), "hourly")
    local = result.start_local.dt.tz_localize(None)
    mask = local.isin(pd.to_datetime(["2023-10-29 02:00", "2023-11-05 02:00"]))
    assert mask.sum() == 3
    assert result.loc[mask, "week_reference_consumption_mwh"].isna().all()
    assert len(result) == len(hourly_frame("2023-10-22", "2023-11-06"))


def test_daily_energy_is_normalized_by_actual_day_length():
    df = pd.DataFrame({
        "start_local": pd.to_datetime(["2023-03-26", "2023-10-29"]).tz_localize(
            "Europe/Berlin"),
        "consumption_mwh": [230.0, 250.0],
    })
    result = add_comparisons(df, "daily")
    assert result.interval_hours.tolist() == [23.0, 25.0]
    assert result.average_load_mw.tolist() == [10.0, 10.0]
    assert result.change_consumption_mwh.isna().all()  # Nonadjacent days.


def test_missing_hour_does_not_create_an_hour_to_hour_change():
    df = hourly_frame("2023-01-01", "2023-01-02").drop(index=2)
    result = add_comparisons(df, "hourly")
    assert pd.isna(result.loc[2, "change_consumption_mwh"])


def test_unsorted_or_duplicate_instants_fail_explicitly():
    df = hourly_frame("2023-01-01", "2023-01-02")
    with pytest.raises(ValueError, match="ordered and unique"):
        add_comparisons(df.iloc[::-1], "hourly")
    with pytest.raises(ValueError, match="ordered and unique"):
        add_comparisons(pd.concat([df.iloc[:1], df]), "hourly")


def test_grouped_screen_retains_values_and_marks_small_groups_unassessed():
    df = pd.DataFrame({"year": [2023] * 9, "month": [1] * 9,
                       "day_type": ["Weekday"] * 9,
                       "average_load_mw": [10, 11, 12, 13, 14, 15, 16, 17, 100]})
    original = df.copy(deep=True)
    result = grouped_iqr(df, "daily")
    assert result.iqr_candidate.sum() == 1
    assert len(result) == len(df)
    pd.testing.assert_frame_equal(df, original)
    assert not grouped_iqr(df.iloc[:5], "daily").iqr_screened.any()
