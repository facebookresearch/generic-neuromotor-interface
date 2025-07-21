# Dictionary of expected values organized by task, data size, and metrics
EXPECTED_TEST_VALUES = {
    "wrist": {
        "full_data": {"test_mae_deg_per_sec": {"value": 11.2348, "atol": 1e-2}},
        "small_subset": {"test_mae_deg_per_sec": {"value": 11.088, "atol": 5e-2}},
    },
    "discrete_gestures": {
        "full_data": {"test_cler": {"value": 0.1819, "atol": 1e-3}},
        "small_subset": {"test_cler": {"value": 0.03321, "atol": 1e-3}},
    },
    "handwriting": {
        "full_data": {"test/CER": {"value": 30.0645, "atol": 1e-2}},
        "small_subset": {"test/CER": {"value": 65.1734, "atol": 5e-1}},
    },
}
