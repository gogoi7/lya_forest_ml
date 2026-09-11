import numpy as np
import pytest

from src.train_xgboost import XGB_DEFAULT_PARAMS, build_parser, make_model, score_pred, feature_columns, MAX_PK_BIN

def test_make_model_preserves_expected_config():
    model = make_model(seed=7)
    params = model.get_params()

    for name, value in XGB_DEFAULT_PARAMS.items():
        assert params[name] == value

    assert params["random_state"] == 7

def test_score_pred():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    scores = score_pred(y_true, y_pred, ["EX0", "EX1"])

    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["confusion_matrix"] == [[1, 1], [0, 2]]
    assert scores["classification_report"]["EX0"]["recall"] == pytest.approx(0.5)

def test_parser_accepts_pairwise_arguments():
    parser = build_parser()
    args = parser.parse_args([
        "--models", "EX2", "EX3",
        "--condition", "raw",
    ])

    assert args.models == ["EX2", "EX3"]
    assert args.condition == "raw"
    assert args.seed == 42
    assert args.max_pk_bin == MAX_PK_BIN

def test_feature_columns_keep_full_range_by_default():
    columns = feature_columns(MAX_PK_BIN)

    np.testing.assert_array_equal(columns, np.arange(92))  # Assuming N_BUNDLE_FEATURES is 92

def test_feature_columns_excludes_high_pk_bins():
    columns = feature_columns(max_pk_bin=14)
    dropped = np.array([32, 33, 34, 35, 36, 78, 79, 80, 81, 82])  # Example indices to drop

    assert columns.size == 82
    assert np.intersect1d(columns, dropped).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate([columns, dropped])),
        np.arange(92),
    )

@pytest.mark.parametrize("max_pk_bin", [-1, 20, 1.5])
def test_feature_columns_invalid_max_pk_bin(max_pk_bin):
    with pytest.raises(ValueError):
        feature_columns(max_pk_bin=max_pk_bin)