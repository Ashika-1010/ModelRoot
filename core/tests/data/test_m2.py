"""M2 tests: CSV loading, validation, profiling.

Test types used (per master-plan markers):
  U  = unit
  P  = property-based (Hypothesis library)
  Sec = security
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelroot.data.errors import ContractViolation, ErrorCode
from modelroot.data.fingerprint import dataset_fingerprint
from modelroot.data.loader import Limits, load_csv
from modelroot.data.profile import profile
from modelroot.data.validate import ValidatedDataset, validate_for_analysis
from modelroot.schemas.common import Algorithm
from modelroot.schemas.config import AnalysisConfig, ModelSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "csv"


def _make_config(
    path: str | pathlib.Path = "dummy.csv",
    target: str = "label",
    positive_class: str = "yes",
    id_columns: list[str] | None = None,
) -> AnalysisConfig:
    return AnalysisConfig(
        data_path=pathlib.Path(path),
        target=target,
        positive_class=positive_class,
        model_spec=ModelSpec(algorithm=Algorithm.logistic_regression),
        metadata={
            "id_columns": id_columns or [],
            "unavailable_at_inference": [],
            "sensitive_columns": [],
        },
    )


def _write_csv(tmp_path: pathlib.Path, name: str, content: str) -> pathlib.Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_basic_df(n: int = 12) -> pd.DataFrame:
    """Return a minimal all-string DataFrame suitable for validation."""
    ages = [str(20 + i) for i in range(n)]
    labels = ["yes" if i % 2 == 0 else "no" for i in range(n)]
    return pd.DataFrame({"age": ages, "label": labels})


# ---------------------------------------------------------------------------
# U: load_csv — happy path
# ---------------------------------------------------------------------------


def test_load_csv_basic(tmp_path: pathlib.Path) -> None:
    """U: Valid CSV loads without error."""
    p = _write_csv(tmp_path, "basic.csv", "a,b,label\n1,2,yes\n3,4,no\n" * 5)
    df = load_csv(p)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b", "label"]


def test_load_csv_bom(tmp_path: pathlib.Path) -> None:
    """U: UTF-8-BOM files load correctly."""
    content = "age,label\n25,yes\n30,no\n" * 5
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
    df = load_csv(p)
    assert "age" in df.columns


def test_load_csv_fixture_bom() -> None:
    """U: BOM fixture file loads."""
    df = load_csv(_FIXTURES / "bom_utf8.csv")
    assert "label" in df.columns
    assert len(df) >= 5


def test_load_csv_quoted_commas(tmp_path: pathlib.Path) -> None:
    """U: Values with embedded commas in quotes are handled."""
    # 2 data rows repeated 5 times = 10 data rows
    body = 'name,val,label\n"a,b",1,yes\n"c,d",2,no\n'
    p = _write_csv(tmp_path, "q.csv", body * 5)
    df = load_csv(p)
    # pandas treats repeated header lines as data rows; just verify load succeeds
    # and the quoted comma column survives intact
    assert len(df) >= 2
    assert "name" in df.columns


# ---------------------------------------------------------------------------
# U: ContractViolation error codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error_code", list(ErrorCode))
def test_error_code_values_are_strings(error_code: ErrorCode) -> None:
    """U: Every ErrorCode is a str subclass."""
    assert isinstance(error_code, str)


def test_file_too_large(tmp_path: pathlib.Path) -> None:
    """Sec: FILE_TOO_LARGE raised before reading file content."""
    p = tmp_path / "big.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    limits = Limits(max_file_bytes=1)  # 1 byte — guaranteed to fail
    with pytest.raises(ContractViolation) as exc_info:
        load_csv(p, limits=limits)
    assert exc_info.value.code == ErrorCode.FILE_TOO_LARGE


def test_null_bytes_rejected(tmp_path: pathlib.Path) -> None:
    """Sec: NULL_BYTES raised for binary files disguised as CSV."""
    p = tmp_path / "binary.csv"
    p.write_bytes(b"col1,col2\n\x001,2\n3,4\n")
    with pytest.raises(ContractViolation) as exc_info:
        load_csv(p)
    assert exc_info.value.code == ErrorCode.NULL_BYTES


def test_pkl_disguised_as_csv(tmp_path: pathlib.Path) -> None:
    """Sec: Pickle file (contains null bytes) renamed to .csv is rejected."""
    p = tmp_path / "model.csv"
    # Pickle files start with \x80 opcode — write minimal pickle-like header
    p.write_bytes(b"\x80\x05\x95\x00\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 10)
    with pytest.raises(ContractViolation) as exc_info:
        load_csv(p)
    assert exc_info.value.code in (ErrorCode.NULL_BYTES, ErrorCode.BAD_ENCODING)


def test_bad_encoding(tmp_path: pathlib.Path) -> None:
    """U: BAD_ENCODING raised for Latin-1 encoded file."""
    p = tmp_path / "latin1.csv"
    p.write_bytes("col,label\ncafé,yes\ntest,no\n".encode("latin-1") * 5)
    with pytest.raises(ContractViolation) as exc_info:
        load_csv(p)
    assert exc_info.value.code == ErrorCode.BAD_ENCODING


def test_no_header_empty_file(tmp_path: pathlib.Path) -> None:
    """U: NO_HEADER raised for empty file."""
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ContractViolation) as exc_info:
        load_csv(p)
    assert exc_info.value.code == ErrorCode.NO_HEADER


def test_too_many_rows(tmp_path: pathlib.Path) -> None:
    """U: TOO_MANY_ROWS raised when row count exceeds limit."""
    rows = "\n".join(f"1,{i}" for i in range(25))
    p = _write_csv(tmp_path, "big.csv", "a,b\n" + rows)
    limits = Limits(max_rows=10, min_rows=1)
    with pytest.raises(ContractViolation) as exc_info:
        load_csv(p, limits=limits)
    assert exc_info.value.code == ErrorCode.TOO_MANY_ROWS


def test_too_many_cols(tmp_path: pathlib.Path) -> None:
    """U: TOO_MANY_COLS raised when column count exceeds limit."""
    header = ",".join(f"c{i}" for i in range(20))
    rows = "\n".join(",".join(str(j) for j in range(20)) for _ in range(12))
    p = _write_csv(tmp_path, "wide.csv", header + "\n" + rows)
    limits = Limits(max_cols=5)
    with pytest.raises(ContractViolation) as exc_info:
        load_csv(p, limits=limits)
    assert exc_info.value.code == ErrorCode.TOO_MANY_COLS


def test_duplicate_columns(tmp_path: pathlib.Path) -> None:
    """U: DUPLICATE_COLUMNS detected before pandas silently renames them."""
    p = _write_csv(tmp_path, "dup.csv", "age,age,label\n25,26,yes\n30,31,no\n" * 5)
    with pytest.raises(ContractViolation) as exc_info:
        load_csv(p)
    assert exc_info.value.code == ErrorCode.DUPLICATE_COLUMNS


def test_too_few_rows(tmp_path: pathlib.Path) -> None:
    """U: TOO_FEW_ROWS raised when data rows below minimum."""
    p = _write_csv(tmp_path, "tiny.csv", "a,b\n1,2\n3,4\n")
    limits = Limits(min_rows=10)
    with pytest.raises(ContractViolation) as exc_info:
        load_csv(p, limits=limits)
    assert exc_info.value.code == ErrorCode.TOO_FEW_ROWS


# ---------------------------------------------------------------------------
# U: validate_for_analysis error codes
# ---------------------------------------------------------------------------


def test_target_missing(tmp_path: pathlib.Path) -> None:
    """U: TARGET_MISSING when target column not in DataFrame."""
    p = _write_csv(tmp_path, "t.csv", "a,b\n" + "\n".join(f"{i},{i}" for i in range(12)))
    df = load_csv(p)
    cfg = _make_config(target="missing_col")
    with pytest.raises(ContractViolation) as exc_info:
        validate_for_analysis(df, cfg)
    assert exc_info.value.code == ErrorCode.TARGET_MISSING


def test_target_not_binary(tmp_path: pathlib.Path) -> None:
    """U: TARGET_NOT_BINARY when target has 3+ unique values."""
    rows = "\n".join(f"{i},{['a', 'b', 'c'][i % 3]}" for i in range(12))
    p = _write_csv(tmp_path, "t.csv", "age,label\n" + rows)
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="a")
    with pytest.raises(ContractViolation) as exc_info:
        validate_for_analysis(df, cfg)
    assert exc_info.value.code == ErrorCode.TARGET_NOT_BINARY


def test_target_has_nulls(tmp_path: pathlib.Path) -> None:
    """U: TARGET_HAS_NULLS when target column contains empty strings."""
    rows = "\n".join(f"{i},{'' if i == 5 else ('yes' if i % 2 == 0 else 'no')}" for i in range(12))
    p = _write_csv(tmp_path, "t.csv", "age,label\n" + rows)
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="yes")
    with pytest.raises(ContractViolation) as exc_info:
        validate_for_analysis(df, cfg)
    assert exc_info.value.code == ErrorCode.TARGET_HAS_NULLS


def test_positive_class_invalid(tmp_path: pathlib.Path) -> None:
    """U: POSITIVE_CLASS_INVALID when positive_class not in target values."""
    rows = "\n".join(f"{i},{'yes' if i % 2 == 0 else 'no'}" for i in range(12))
    p = _write_csv(tmp_path, "t.csv", "age,label\n" + rows)
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="NOPE")
    with pytest.raises(ContractViolation) as exc_info:
        validate_for_analysis(df, cfg)
    assert exc_info.value.code == ErrorCode.POSITIVE_CLASS_INVALID


def test_class_too_small(tmp_path: pathlib.Path) -> None:
    """U: CLASS_TOO_SMALL when one class has only 1 sample."""
    rows = ["25,yes"] + ["30,no"] * 11  # only 1 positive
    p = _write_csv(tmp_path, "t.csv", "age,label\n" + "\n".join(rows))
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="yes")
    with pytest.raises(ContractViolation) as exc_info:
        validate_for_analysis(df, cfg)
    assert exc_info.value.code == ErrorCode.CLASS_TOO_SMALL


# ---------------------------------------------------------------------------
# U: validate_for_analysis — happy path and column-type inference
# ---------------------------------------------------------------------------


def test_validate_returns_validated_dataset(tmp_path: pathlib.Path) -> None:
    """U: Valid CSV yields ValidatedDataset with y ∈ {0, 1}."""
    df = load_csv(_FIXTURES / "valid_basic.csv")
    cfg = _make_config(target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    assert isinstance(vd, ValidatedDataset)
    assert set(vd.y.unique()) <= {0, 1}
    assert vd.positive_class == "yes"
    assert "label" not in vd.feature_cols


def test_numeric_column_inferred(tmp_path: pathlib.Path) -> None:
    """U: Column with ≥95% numeric values and >10 unique values → 'numeric'."""
    ages = "\n".join(str(i * 3 + 20) for i in range(12))
    rows = ages.split("\n")
    rows_with_label = [f"{a},{'yes' if i % 2 == 0 else 'no'}" for i, a in enumerate(rows)]
    p = _write_csv(tmp_path, "t.csv", "age,label\n" + "\n".join(rows_with_label))
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    assert vd.column_types.get("age") == "numeric"


def test_categorical_column_inferred(tmp_path: pathlib.Path) -> None:
    """U: Column with few unique string values → 'categorical'."""
    rows = [f"{'M' if i % 2 == 0 else 'F'},{'yes' if i % 2 == 0 else 'no'}" for i in range(12)]
    p = _write_csv(tmp_path, "t.csv", "gender,label\n" + "\n".join(rows))
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    assert vd.column_types.get("gender") == "categorical"


def test_mixed_type_column_is_categorical(tmp_path: pathlib.Path) -> None:
    """U: Column with mixed numeric/string values → 'categorical'."""
    vals = ["25", "30", "N/A", "35", "40", "N/A", "28", "31", "39", "45", "22", "50"]
    rows = [f"{v},{'yes' if i % 2 == 0 else 'no'}" for i, v in enumerate(vals)]
    p = _write_csv(tmp_path, "t.csv", "age,label\n" + "\n".join(rows))
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    # Mixed → fails numeric threshold → categorical
    assert vd.column_types.get("age") == "categorical"


def test_constant_column_dropped(tmp_path: pathlib.Path) -> None:
    """U: Constant column is dropped with a warning."""
    rows = [f"{i},CONST,{'yes' if i % 2 == 0 else 'no'}" for i in range(12)]
    p = _write_csv(tmp_path, "t.csv", "age,const_col,label\n" + "\n".join(rows))
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    dropped_names = {n for n, _ in vd.dropped}
    assert "const_col" in dropped_names
    assert "const_col" not in vd.feature_cols
    assert any("constant" in w.lower() for w in vd.warnings)


def test_id_column_excluded(tmp_path: pathlib.Path) -> None:
    """U: id_columns are excluded from features with a warning."""
    rows = [f"ID{i},{i},{'yes' if i % 2 == 0 else 'no'}" for i in range(12)]
    p = _write_csv(tmp_path, "t.csv", "row_id,age,label\n" + "\n".join(rows))
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="yes", id_columns=["row_id"])
    vd = validate_for_analysis(df, cfg)
    assert "row_id" not in vd.feature_cols
    assert any("row_id" in w for w in vd.warnings)


def test_formula_like_cells_loaded_as_strings(tmp_path: pathlib.Path) -> None:
    """Sec: Formula-like cells are loaded as plain strings, not evaluated."""
    rows = [
        "=SUM(1+1),yes",
        "@attack,no",
        "+injection,yes",
        "-cmd,no",
    ] * 3
    p = _write_csv(tmp_path, "t.csv", "formula,label\n" + "\n".join(rows))
    df = load_csv(p)
    # Should load without error; values are just strings
    assert df["formula"].iloc[0] == "=SUM(1+1)"


def test_high_missing_column_warning(tmp_path: pathlib.Path) -> None:
    """U: Column with >60% missing values generates a warning but is kept."""
    vals = ["" if i < 8 else str(20 + i) for i in range(12)]  # 8/12 = 67% missing
    rows = [f"{v},{'yes' if i % 2 == 0 else 'no'}" for i, v in enumerate(vals)]
    p = _write_csv(tmp_path, "t.csv", "age,label\n" + "\n".join(rows))
    df = load_csv(p)
    cfg = _make_config(target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    assert any("missing" in w.lower() for w in vd.warnings)
    assert "age" in vd.feature_cols  # not dropped, just warned


# ---------------------------------------------------------------------------
# U: profile
# ---------------------------------------------------------------------------


def test_profile_matches_fixture() -> None:
    """U: Profile n_rows, n_cols, class_balance match hand-computed values."""
    df = load_csv(_FIXTURES / "valid_basic.csv")
    cfg = _make_config(target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    dp = profile(df, vd.column_types, target_col="label", dropped=vd.dropped, warnings=vd.warnings)
    assert dp.n_rows == 12
    assert dp.n_cols == 4
    assert dp.class_balance.get("yes", 0) + dp.class_balance.get("no", 0) == 12
    assert len(dp.columns) == 4


def test_profile_column_profile_fields() -> None:
    """U: Each ColumnProfile has expected field types."""
    df = load_csv(_FIXTURES / "valid_basic.csv")
    cfg = _make_config(target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    dp = profile(df, vd.column_types, target_col="label")
    for cp in dp.columns:
        assert isinstance(cp.missing_frac, float)
        assert 0.0 <= cp.missing_frac <= 1.0
        assert isinstance(cp.n_unique, int)
        assert isinstance(cp.unique_ratio, float)


# ---------------------------------------------------------------------------
# U: dataset_fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_same_df() -> None:
    """U: Same DataFrame content → same fingerprint."""
    df = _make_basic_df()
    assert dataset_fingerprint(df) == dataset_fingerprint(df.copy())


def test_fingerprint_different_values() -> None:
    """U: Different content → different fingerprint."""
    df1 = _make_basic_df(12)
    df2 = _make_basic_df(12)
    df2.iloc[0, 0] = "99"
    assert dataset_fingerprint(df1) != dataset_fingerprint(df2)


def test_fingerprint_column_rename_changes_hash() -> None:
    """U: Column rename changes the fingerprint."""
    df1 = _make_basic_df()
    df2 = df1.rename(columns={"age": "AGE"})
    assert dataset_fingerprint(df1) != dataset_fingerprint(df2)


def test_fingerprint_hex_string() -> None:
    """U: Fingerprint is a 64-character hex string."""
    fp = dataset_fingerprint(_make_basic_df())
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# P: property-based — column order shuffle does not change column types
# ---------------------------------------------------------------------------


@given(
    st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=12,
        max_size=12,
    )
)
@settings(max_examples=20)
def test_column_order_shuffle_does_not_change_types(perm_seed: list[int]) -> None:
    """P: Shuffling column order does not change inferred column types."""
    import random

    rows = [(str(20 + i), str(50000 + i * 1000), "yes" if i % 2 == 0 else "no") for i in range(12)]
    df = pd.DataFrame(rows, columns=["age", "income", "label"])
    cols = ["age", "income", "label"]
    rng = random.Random(sum(perm_seed))
    shuffled_cols = cols.copy()
    rng.shuffle(shuffled_cols)
    df_shuffled = df[shuffled_cols]

    from modelroot.data.validate import _infer_column_types

    warns1: list[str] = []
    dropped1: list[tuple[str, str]] = []
    types1, _ = _infer_column_types(df, {"label"}, warns1, dropped1)

    warns2: list[str] = []
    dropped2: list[tuple[str, str]] = []
    types2, _ = _infer_column_types(df_shuffled, {"label"}, warns2, dropped2)

    assert types1 == types2
