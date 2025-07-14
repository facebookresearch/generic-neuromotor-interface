# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from typing import Literal

import numba as nb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from generic_neuromotor_interface.constants import GestureType

THRESHOLD = 0.35
DEBOUNCE = 0.05
TOLERANCE = (-0.05, 0.25)

# Numba alignment constants
LEFT = 1
RIGHT = 2
BOTH = 3


def map_gestures_to_probabilities(
    probabilities: NDArray,
    times: NDArray
) -> dict[str, NDArray]:
    """
    Assembles a mapping from each gesture type to its corresponding
    probabilities from the model outputs, plus a "time" key containing
    the timestamps associated with each output.

    Parameters
    ----------
    probabilities : NDArray
        Model output array of shape (num_gestures, sequence_length)
    times : NDArray
        Timestamps corresponding to each probability, of shape
        (sequence_length,)

    Returns
    -------
    gesture_probabilities : dict[str, NDArray]
        Dictionary mapping gesture names to their predicted
        probabilities. Keys are gesture names, plus "time"
        key containing the timestamps corresponding to each
        prediction.
    """
    num_probs = probabilities.shape[0]
    num_gestures = len(GestureType)
    if num_probs != num_gestures:
        raise ValueError(
            f"Expected probabilities array with {num_gestures} rows, got {num_probs}"
        )

    gesture_probabilities = {}
    for gesture in GestureType:
        gesture_probabilities[gesture.name] = probabilities[gesture.value]

    gesture_probabilities["time"] = times

    return gesture_probabilities


def debounce_events(
    events: list[tuple[str, float]], debounce: float
) -> list[tuple[str, float]]:
    """
    Apply debouncing to a list of events with specific rules:
    1. If two non-release events occur within debounce time, the later event is removed
    2. For release events (index_release, middle_release), only debounce the same
       kind of release

    Parameters
    ----------
    events : list[tuple[str, float]]
        List of (gesture_name, timestamp) tuples to process
    debounce : float
        Minimum time (in seconds) between consecutive events

    Returns
    -------
    List[Tuple[str, float]]
        Events after debouncing is applied
    """
    if not events:
        return []

    result = [events[0]]  # Always keep the first event

    # Define release event types
    release_events = {"index_release", "middle_release"}

    for name, time in events[1:]:
        prev_name, prev_time = result[-1]
        time_diff = time - prev_time

        # Skip if too close in time and meets debounce criteria
        if time_diff < debounce:
            # Case 1: Both are non-release events - skip the current (later) one
            if name not in release_events and prev_name not in release_events:
                continue

            # Case 2: Both are release events of the same kind - skip the current one
            if name in release_events and name == prev_name:
                continue

            # Otherwise, don't debounce (different types or non-matching release events)

        # Add event if it passes debounce rules
        result.append((name, time))

    return result


def detect_gesture_events(
    gesture_probabilities: dict[str, NDArray],
    threshold: float,
    debounce: float,
) -> pd.DataFrame:
    """
    Identify threshold crossings of gesture probabilities to detect
    discrete gesture events and their associated timestamps.

    Parameters
    ----------
    gesture_probabilities : dict[str, NDArray]
        Dictionary mapping gesture names to their predicted
        probabilities. Keys are gesture names, plus "time"
        key containing the timestamps corresponding to each
        prediction.
    threshold : float
        An event is detected if the probability rises above this threshold
    debounce : float
        Minimum time (in seconds) between consecutive events

    Returns
    -------
    detected_events : pd.DataFrame
        DataFrame containing detected events with columns:
        - start: start time of the event
        - end: end time of the event
        - name: name of the gesture
        - time: time of the event
    """
    # 1. Thresholded Peak Detection
    events: list[tuple[str, float]] = []
    times = gesture_probabilities["time"]

    for name, probs in gesture_probabilities.items():
        if name == "time":
            continue

        assert probs.ndim == 1

        # If there's a threshold crossing on the first timestep, count it
        if probs[0] >= threshold:
            events.append((name, times[0]))

        # Find times[i] where probs[i-1] < threshold and probs[i] >= threshold
        threshold_crossings = times[1:][
            (probs[1:] >= threshold) & (probs[:-1] < threshold)
        ]
        events.extend((name, t) for t in threshold_crossings)

    events = sorted(events, key=lambda x: x[1])

    # 2. Apply debouncing
    events = debounce_events(events, debounce=debounce)

    # Assemble into a dataframe
    detected_events = pd.DataFrame(
        [
            {"start": time, "end": time, "time": time, "name": name}
            for name, time in events
        ]
    )

    return detected_events


def pad_dataframe(df: pd.DataFrame, start_pad: float, end_pad: float) -> pd.DataFrame:
    """
    Pads the start and end times in a DataFrame to create expanded time windows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'start' and 'end' time columns
    start_pad : float
        Amount to subtract from 'start' times (negative values extend window)
    end_pad : float
        Amount to add to 'end' times (positive values extend window)

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with padded 'start' and 'end' columns
    """
    # Create a copy to avoid modifying the original DataFrame
    padded_df = df.copy()

    # Apply padding to the start and end columns
    padded_df["start"] = padded_df["start"] + start_pad
    padded_df["end"] = padded_df["end"] + end_pad

    return padded_df


def get_start_end(df: pd.DataFrame) -> tuple[NDArray, NDArray]:
    """
    Get start and end time arrays from DataFrame with appropriate transformations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'start' and 'end' time columns

    Returns
    -------
    tuple[NDArray, NDArray]
        Processed start and end time arrays
    """
    start = df["start"].to_numpy()
    end = df["end"].to_numpy()

    start = np.maximum.accumulate(start)
    end = np.minimum.accumulate(end[::-1])[::-1]
    return start, end


@nb.njit(nogil=True)
def _calculate_cost(
    lnames: NDArray,
    ltimes: NDArray,
    rnames: NDArray,
    rtimes: NDArray,
    i: int,
    j: int,
) -> tuple[float, float]:
    """
    Calculates alignment cost between two gesture events.

    Parameters
    ----------
    lnames : NDArray
        Names (encoded as integers) for left sequence events
    ltimes : NDArray
        Times for left sequence events
    rnames : NDArray
        Names (encoded as integers) for right sequence events
    rtimes : NDArray
        Times for right sequence events
    i : int
        Index into left sequence
    j : int
        Index into right sequence

    Returns
    -------
    tuple[float, float]
        Cost tuple: (name_mismatch_cost, time_delta)
    """
    time_delta = abs(ltimes[i] - rtimes[j])

    if lnames[i] == rnames[j]:
        return (0.0, time_delta)
    else:
        return (1.0, time_delta)


@nb.njit(nogil=True)
def _matched_indices_implementation(
    lstart: NDArray,
    lend: NDArray,
    lnames: NDArray,
    ltimes: NDArray,
    rstart: NDArray,
    rend: NDArray,
    rnames: NDArray,
    rtimes: NDArray,
) -> list[tuple[int | None, int | None]]:
    """
    Align two sequences using a time-bounded Needleman-Wunsch algorithm.

    This is a Numba-compatible implementation that preserves the original algorithm
    logic but uses arrays instead of dictionaries for the dynamic programming grid.

    Parameters
    ----------
    lstart, lend : NDArray
        Start and end times for left sequence events
    lnames, ltimes : NDArray
        Names and times data for left sequence
    rstart, rend : NDArray
        Start and end times for right sequence events
    rnames, rtimes : NDArray
        Names and times data for right sequence

    Returns
    -------
    List[Tuple[Optional[int], Optional[int]]]
        Optimal alignment between the two sequences
    """
    left_len = len(lstart)
    right_len = len(rstart)

    # Create arrays to store the dynamic programming state
    # We'll use separate arrays for costs and opcodes
    # shape: (left_len+1, right_len+1) to account for empty prefix alignments

    # Initialize cost matrix for first component (name mismatch count)
    cost_mismatch = np.zeros((left_len + 1, right_len + 1), dtype=np.float64)
    # Initialize cost matrix for second component (time delta)
    cost_time = np.zeros((left_len + 1, right_len + 1), dtype=np.float64)
    # Initialize opcodes matrix
    opcodes = np.zeros((left_len + 1, right_len + 1), dtype=np.int64)

    # Initialize first row and column for aligning with empty sequences
    for i in range(1, left_len + 1):
        cost_mismatch[i, 0] = i
        opcodes[i, 0] = LEFT

    for j in range(1, right_len + 1):
        cost_mismatch[0, j] = j
        opcodes[0, j] = RIGHT

    # Fill the dynamic programming matrices
    # Note: We use 1-based indexing in the matrices for convenience
    # i,j in matrices correspond to i-1,j-1 in the input arrays
    for i in range(1, left_len + 1):
        for j in range(1, right_len + 1):
            i_idx = i - 1  # Convert to 0-based index for input arrays
            j_idx = j - 1

            # Check if intervals overlap
            if lend[i_idx] < rstart[j_idx] or rend[j_idx] < lstart[i_idx]:
                # No overlap - events cannot be matched
                # Choose based on cost
                left_cost = (cost_mismatch[i - 1, j] + 1, cost_time[i - 1, j])
                right_cost = (cost_mismatch[i, j - 1] + 1, cost_time[i, j - 1])

                # Lexicographic comparison
                if left_cost[0] < right_cost[0] or (
                    left_cost[0] == right_cost[0] and left_cost[1] <= right_cost[1]
                ):
                    cost_mismatch[i, j] = left_cost[0]
                    cost_time[i, j] = left_cost[1]
                    opcodes[i, j] = LEFT
                else:
                    cost_mismatch[i, j] = right_cost[0]
                    cost_time[i, j] = right_cost[1]
                    opcodes[i, j] = RIGHT
            else:
                # Intervals overlap - events can be matched
                # Calculate all three options
                match_cost = _calculate_cost(
                    lnames, ltimes, rnames, rtimes, i_idx, j_idx
                )

                left_cost = (cost_mismatch[i - 1, j] + 1, cost_time[i - 1, j])
                right_cost = (cost_mismatch[i, j - 1] + 1, cost_time[i, j - 1])
                both_cost = (
                    cost_mismatch[i - 1, j - 1] + match_cost[0],
                    cost_time[i - 1, j - 1] + match_cost[1],
                )

                # Choose the minimum cost option using lexicographic comparison
                if left_cost[0] < right_cost[0] and left_cost[0] < both_cost[0]:
                    cost_mismatch[i, j] = left_cost[0]
                    cost_time[i, j] = left_cost[1]
                    opcodes[i, j] = LEFT
                elif right_cost[0] < left_cost[0] and right_cost[0] < both_cost[0]:
                    cost_mismatch[i, j] = right_cost[0]
                    cost_time[i, j] = right_cost[1]
                    opcodes[i, j] = RIGHT
                elif both_cost[0] < left_cost[0] and both_cost[0] < right_cost[0]:
                    cost_mismatch[i, j] = both_cost[0]
                    cost_time[i, j] = both_cost[1]
                    opcodes[i, j] = BOTH
                # If there's a tie in the first cost component, use the second component
                elif left_cost[0] == right_cost[0] and left_cost[0] == both_cost[0]:
                    if left_cost[1] <= right_cost[1] and left_cost[1] <= both_cost[1]:
                        cost_mismatch[i, j] = left_cost[0]
                        cost_time[i, j] = left_cost[1]
                        opcodes[i, j] = LEFT
                    elif (
                        right_cost[1] <= left_cost[1] and right_cost[1] <= both_cost[1]
                    ):
                        cost_mismatch[i, j] = right_cost[0]
                        cost_time[i, j] = right_cost[1]
                        opcodes[i, j] = RIGHT
                    else:
                        cost_mismatch[i, j] = both_cost[0]
                        cost_time[i, j] = both_cost[1]
                        opcodes[i, j] = BOTH
                elif left_cost[0] == right_cost[0]:
                    if left_cost[1] <= right_cost[1]:
                        cost_mismatch[i, j] = left_cost[0]
                        cost_time[i, j] = left_cost[1]
                        opcodes[i, j] = LEFT
                    else:
                        cost_mismatch[i, j] = right_cost[0]
                        cost_time[i, j] = right_cost[1]
                        opcodes[i, j] = RIGHT
                elif left_cost[0] == both_cost[0]:
                    if left_cost[1] <= both_cost[1]:
                        cost_mismatch[i, j] = left_cost[0]
                        cost_time[i, j] = left_cost[1]
                        opcodes[i, j] = LEFT
                    else:
                        cost_mismatch[i, j] = both_cost[0]
                        cost_time[i, j] = both_cost[1]
                        opcodes[i, j] = BOTH
                elif right_cost[0] == both_cost[0]:
                    if right_cost[1] <= both_cost[1]:
                        cost_mismatch[i, j] = right_cost[0]
                        cost_time[i, j] = right_cost[1]
                        opcodes[i, j] = RIGHT
                    else:
                        cost_mismatch[i, j] = both_cost[0]
                        cost_time[i, j] = both_cost[1]
                        opcodes[i, j] = BOTH

    # Traceback to find the optimal alignment path
    path: list[tuple[int | None, int | None]] = []
    i, j = left_len, right_len

    while i > 0 or j > 0:
        if i > 0 and j > 0 and opcodes[i, j] == BOTH:
            path.append((i - 1, j - 1))  # Convert back to 0-based indexing
            i -= 1
            j -= 1
        elif i > 0 and opcodes[i, j] == LEFT:
            path.append((i - 1, None))  # Convert back to 0-based indexing
            i -= 1
        elif j > 0 and opcodes[i, j] == RIGHT:
            path.append((None, j - 1))  # Convert back to 0-based indexing
            j -= 1
        else:
            # This shouldn't happen with a properly filled matrix
            break

    # Return path in correct order (we built it backwards)
    return path[::-1]


def get_matched_indices(
    left: pd.DataFrame, right: pd.DataFrame
) -> list[tuple[int | None, int | None]]:
    """
    Get the indices defining the optimal alignment between `left` and `right`

    Parameters
    ----------
    left : pd.DataFrame
        The DataFrame representing the first sequence of events
    right : pd.DataFrame
        The DataFrame representing the second sequence of events

    Returns
    -------
    list[tuple[int | None, int | None]]
        A list of tuples of indices defining the alignment. Each tuple
        (left_index, right_index) represents the alignment between
        left[left_index] and right[right_index] (or against None).
    """
    lstart, lend = get_start_end(left)
    rstart, rend = get_start_end(right)

    # Extract names from DataFrames and convert to category codes
    lnames = left["name"]
    rnames = right["name"]

    # Create a categorical type from the union of unique names
    dtype = pd.CategoricalDtype(list(set(lnames.unique()) | set(rnames.unique())))

    # Convert to category codes for numerical comparison
    lnames_encoded = lnames.astype(dtype).cat.codes.to_numpy()
    rnames_encoded = rnames.astype(dtype).cat.codes.to_numpy()

    # Extract times
    ltimes = left["time"].to_numpy()
    rtimes = right["time"].to_numpy()

    # Call the Needleman-Wunsch implementation
    return _matched_indices_implementation(
        lstart, lend, lnames_encoded, ltimes, rstart, rend, rnames_encoded, rtimes
    )


def compute_confusion_matrix(
    events: pd.DataFrame,
    prompts_df: pd.DataFrame,
    tolerance: tuple[float, float],
) -> Counter:
    """
    Compute confusion matrix from detected events and ground truth labels.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing detected events with columns:
        - start: start time of the event
        - end: end time of the event
        - name: name of the gesture
        - time: time of the event
    prompts_df : pd.DataFrame
        Ground truth labels with at least 'name' and 'time' columns
    tolerance : Tuple[float, float], default=TOLERANCE
        Time window tolerance for matching predictions to labels

    Returns
    -------
    Counter
        Confusion matrix with counts of each confusion
    """

    # Prepare the labels DataFrame for alignment
    labels = prompts_df.copy()
    labels["start"] = labels["time"]
    labels["end"] = labels["time"]

    # First Needleman-Wunsch: aligning predicted events with labels
    padded_labels = pad_dataframe(labels, tolerance[0], tolerance[1])
    matches = get_matched_indices(events, padded_labels)

    keep = []
    state: Literal["NEUTRAL", "INDEX", "MIDDLE"] = "NEUTRAL"
    gt_names = labels["name"].to_list()
    pred_names = events["name"].to_list()
    for pred, gt in matches:
        # Check if the prediction should be kept
        if pred is not None:
            if state == "NEUTRAL" and pred_names[pred] not in {
                GestureType.index_release.name,
                GestureType.middle_release.name,
            }:
                keep.append(pred)
            elif (
                state == "INDEX" and pred_names[pred] != GestureType.middle_release.name
            ):
                keep.append(pred)
            elif (
                state == "MIDDLE" and pred_names[pred] != GestureType.index_release.name
            ):
                keep.append(pred)

        # Advance the state machine
        if gt is not None:
            if gt_names[gt] == GestureType.index_press.name:
                state = "INDEX"
            elif gt_names[gt] == GestureType.middle_press.name:
                state = "MIDDLE"
            else:
                state = "NEUTRAL"

    filtered_events = events.iloc[keep].reset_index(drop=True)

    # Second Needleman-Wunsch: aligning filtered events with labels
    filtered_matches = get_matched_indices(padded_labels, filtered_events)

    # Calculate confusion matrix with the filtered events
    confusion_matrix: Counter = Counter()
    for left_idx, right_idx in filtered_matches:
        if left_idx is None:
            pass
        elif right_idx is None:
            confusion_matrix[labels["name"].iloc[left_idx], None] += 1
        else:
            confusion_matrix[
                labels["name"].iloc[left_idx],
                filtered_events["name"].iloc[right_idx],
            ] += 1

    return confusion_matrix


def compute_cler(
    probabilities: NDArray,
    times: NDArray,
    prompts_df: pd.DataFrame,
    threshold: float = THRESHOLD,
    debounce: float = DEBOUNCE,
    tolerance: tuple[float, float] = TOLERANCE,
) -> float:
    """
    Compute Classification Error Rate (CLER) from model probabilities and
    ground truth labels.

    CLER is the proportion of events detected by the model that were assigned
    the incorrect gesture, in a balanced average across all gestures.

    Parameters
    ----------
    probabilities : NDArray
        Model output probabilities of shape (num_gestures, sequence_length)
    times : NDArray
        Timestamps corresponding to each prediction
    prompts_df : pd.DataFrame
        Ground truth labels with at least 'name' and 'time' columns
    threshold : float, default=THRESHOLD
        Threshold value for detecting gesture events
    debounce : float, default=DEBOUNCE
        Minimum time (in seconds) between consecutive events
    tolerance : Tuple[float, float], default=TOLERANCE
        Time window tolerance for matching predictions to labels

    Returns
    -------
    float
        The Classification Error Rate (CLER)
    """

    # Extract the probabilities of each gesture
    gesture_probabilities = map_gestures_to_probabilities(probabilities, times)

    # Detect discrete gesture events from predicted probabilities
    detected_events = detect_gesture_events(
        gesture_probabilities, threshold=threshold, debounce=debounce
    )

    # Calculate confusion matrix
    confusion_matrix = compute_confusion_matrix(
        detected_events, prompts_df, tolerance
    )

    # Calculate the classification error rate
    all_gestures = {gt for gt, _ in confusion_matrix.keys() if gt is not None}

    error_rates = []

    for gesture in all_gestures:
        correct = confusion_matrix.get((gesture, gesture), 0)

        total = 0
        for pred in list(all_gestures):
            total += confusion_matrix.get((gesture, pred), 0)

        if total > 0:
            # Error rate = 1 - accuracy
            error_rate = 1.0 - (correct / total)
            error_rates.append(error_rate)

    return sum(error_rates) / len(error_rates) if error_rates else 0.0
