import pandas as pd

THRESHOLD = 95
SUMMARY_FREQ = 12_000


def first_step_at(training: pd.DataFrame, threshold: float) -> pd.Series:
    above = training[training["cumulative_reward"] >= threshold]
    return above.groupby("RunID")["Step"].min()


def last_logged_step(training: pd.DataFrame) -> pd.Series:
    return training.groupby("RunID")["Step"].max()


def expected_last_step(max_steps: int) -> int:
    return (max_steps // SUMMARY_FREQ) * SUMMARY_FREQ


def compute_outcomes(runs: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    runs = runs.set_index("RunID")
    first = first_step_at(training, THRESHOLD).reindex(runs.index)
    last = last_logged_step(training).reindex(runs.index)
    ended = last >= expected_last_step(runs["max_steps"])

    reached = pd.Series(pd.NA, index=runs.index, dtype="boolean")
    reached[first.notna()] = True
    reached[first.isna() & ended] = False

    return pd.DataFrame({
        "last_logged_step": last,
        "reached_95": reached,
        "steps_to_95": first.astype("Int64"),
    })
