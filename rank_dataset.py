"""
rank_dataset.py

PyTorch dataset and sampling utilities for the cross-sectional ranking model.

Role
----
This module consumes factor_pipeline.FeatureLabelBundle outputs:

    feature_panel: MultiIndex [trade_date, ts_code], columns = factors
    label_panel:   MultiIndex [trade_date, ts_code], columns = labels

It does:
1. Select valid signal dates.
2. Randomly sample N stocks without replacement for each date.
3. Build time-series tensor x_ts with shape [N, F_ts, T].
4. Build scalar tensor x_scalar with shape [N, F_scalar].
5. Dynamically generate local rank labels inside the sampled N-stock batch.
6. Return metadata such as date, codes, and raw forward returns.

It does NOT:
1. Compute factors.
2. Compute labels.
3. Read/write databases.
4. Train the model.

Recommended training usage
--------------------------
from torch.utils.data import DataLoader
import rank_dataset as rd

ds = rd.CrossSectionRankDataset.from_bundle(
    bundle,
    config=rd.CrossSectionDatasetConfig(
        sample_size=512,
        seq_len=128,
        samples_per_date=4,
        label_name="label_ret_t1_t6",
        target_mode="rank_pct",
    ),
)

loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=rd.cross_section_collate_fn)

for batch in loader:
    x_ts = batch["x_ts"]          # [B, N, F_ts, T]
    x_scalar = batch["x_scalar"]  # [B, N, F_scalar]
    y = batch["y"]                # [B, N]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore
        pass


EPS = 1e-12


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class CrossSectionDatasetConfig:
    """
    Dataset configuration.

    sample_size:
        Number of stocks N sampled per training item.

    seq_len:
        Time-series length T ending at signal date t.

    samples_per_date:
        Number of stochastic samples generated for each signal date in one epoch.

    require_full_history:
        If True, a date is usable only when at least seq_len dates exist in the
        feature_panel before and including that date. If False, missing left
        history is padded.

    target_mode:
        One of:
        - "rank_pct": local percentile rank, high return -> close to 1.0
        - "rank_centered": 2 * rank_pct - 1
        - "rank": local rank, low return -> 1, high return -> N
        - "zscore": local z-score of raw future return
        - "raw": raw future return

    return_tensors:
        - "torch": return torch.FloatTensor
        - "numpy": return np.ndarray
    """

    sample_size: int = 512
    seq_len: int = 128
    samples_per_date: int = 1
    label_name: str = "label_ret_t1_t6"
    label_valid_name: Optional[str] = "label_valid_t1_t6"
    target_mode: str = "rank_pct"

    require_full_history: bool = True
    min_valid_stocks: Optional[int] = None
    allow_smaller_sample: bool = False

    fill_value: float = 0.0
    return_tensors: str = "torch"

    random_seed: int = 42
    deterministic: bool = False

    candidate_pool_size: Optional[int] = None
    sort_codes: bool = True


# ============================================================
# Basic helpers
# ============================================================

def _ensure_torch_available() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for return_tensors='torch' or DataLoader usage.")


def _check_panel(panel: pd.DataFrame, name: str) -> None:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError(f"{name}.index must be MultiIndex [trade_date, ts_code]")
    index_names = list(panel.index.names)
    if "trade_date" not in index_names or "ts_code" not in index_names:
        raise ValueError(f"{name}.index must have levels named 'trade_date' and 'ts_code'")


def _unique_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(panel.index.get_level_values("trade_date").unique()).sort_values()


def _normalize_trade_date(x: Any) -> pd.Timestamp:
    return pd.Timestamp(x)


def _normalize_codes(codes: Iterable[Any], sort_codes: bool = True) -> list[str]:
    out = [str(c) for c in codes]
    if sort_codes:
        out = sorted(out)
    return out


def _to_numpy_float(df: pd.DataFrame | pd.Series) -> np.ndarray:
    return np.asarray(df, dtype=np.float32)


def _as_bool_series(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def _local_target_from_returns(
    returns: pd.Series,
    target_mode: str = "rank_pct",
    ascending: bool = True,
) -> pd.Series:
    """
    Dynamic local target computed inside a sampled cross-section.

    With ascending=True:
        lowest return has the lowest rank;
        highest return has the highest rank / rank_pct.
    """
    y = pd.to_numeric(returns, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)

    if target_mode == "raw":
        return y.rename("y")

    valid_count = int(y.notna().sum())
    if valid_count < 2:
        return pd.Series(np.nan, index=y.index, name="y")

    if target_mode == "rank":
        return y.rank(ascending=ascending, method="average", na_option="keep").rename("y")

    rank_pct = y.rank(ascending=ascending, method="average", pct=True, na_option="keep")

    if target_mode == "rank_pct":
        return rank_pct.rename("y")

    if target_mode == "rank_centered":
        return (2.0 * rank_pct - 1.0).rename("y")

    if target_mode == "zscore":
        mu = y.mean()
        sigma = y.std(ddof=0)
        if not np.isfinite(sigma) or sigma <= EPS:
            return pd.Series(np.nan, index=y.index, name="y")
        return ((y - mu) / sigma).rename("y")

    raise ValueError("target_mode must be one of: rank_pct, rank_centered, rank, zscore, raw")


def _convert_array(arr: np.ndarray, return_tensors: str) -> Any:
    if return_tensors == "numpy":
        return arr
    if return_tensors == "torch":
        _ensure_torch_available()
        return torch.from_numpy(arr.astype(np.float32, copy=False))
    raise ValueError("return_tensors must be 'torch' or 'numpy'")


# ============================================================
# Factor/label name helpers
# ============================================================

def infer_factor_groups_from_metadata(metadata: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """
    Infer ts/scalar factor names from bundle.metadata.

    Expected keys from factor_pipeline:
        ts_factor_names
        scalar_factor_names
    """
    if "ts_factor_names" not in metadata or "scalar_factor_names" not in metadata:
        raise KeyError("metadata must contain 'ts_factor_names' and 'scalar_factor_names'")
    return list(metadata["ts_factor_names"]), list(metadata["scalar_factor_names"])


def validate_factor_names(
    feature_panel: pd.DataFrame,
    ts_factor_names: Sequence[str],
    scalar_factor_names: Sequence[str],
) -> None:
    """Check that all requested factors exist in feature_panel columns."""
    missing_ts = [x for x in ts_factor_names if x not in feature_panel.columns]
    missing_scalar = [x for x in scalar_factor_names if x not in feature_panel.columns]
    if missing_ts or missing_scalar:
        raise KeyError(f"missing factors. ts={missing_ts}, scalar={missing_scalar}")


def validate_label_names(
    label_panel: pd.DataFrame,
    label_name: str,
    label_valid_name: Optional[str] = None,
) -> None:
    """Check that requested label columns exist."""
    if label_name not in label_panel.columns:
        raise KeyError(f"label_name not found in label_panel: {label_name}")
    if label_valid_name is not None and label_valid_name not in label_panel.columns:
        raise KeyError(f"label_valid_name not found in label_panel: {label_valid_name}")


# ============================================================
# Panel indexing helpers
# ============================================================

def get_date_subpanel(panel: pd.DataFrame, trade_date: Any) -> pd.DataFrame:
    """Return panel rows for one trade_date, indexed by ts_code."""
    _check_panel(panel, "panel")
    dt = _normalize_trade_date(trade_date)
    try:
        return panel.xs(dt, level="trade_date")
    except KeyError:
        return pd.DataFrame(columns=panel.columns)


def get_valid_codes_for_date(
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    trade_date: Any,
    label_name: str,
    label_valid_name: Optional[str] = None,
    candidate_codes: Optional[Iterable[str]] = None,
    sort_codes: bool = True,
) -> list[str]:
    """
    Return codes that have usable current features and labels on trade_date.
    """
    f = get_date_subpanel(feature_panel, trade_date)
    l = get_date_subpanel(label_panel, trade_date)

    if f.empty or l.empty:
        return []

    feature_ok = f.notna().any(axis=1)
    label_ok = l[label_name].notna() & np.isfinite(pd.to_numeric(l[label_name], errors="coerce"))

    valid = feature_ok.reindex(label_ok.index).fillna(False) & label_ok

    if label_valid_name is not None and label_valid_name in l.columns:
        valid &= l[label_valid_name].fillna(False).astype(bool)

    codes = valid[valid].index.astype(str).tolist()

    if candidate_codes is not None:
        candidate_set = set(str(c) for c in candidate_codes)
        codes = [c for c in codes if c in candidate_set]

    return _normalize_codes(codes, sort_codes=sort_codes)


def build_valid_codes_by_date(
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    label_name: str,
    label_valid_name: Optional[str] = None,
    signal_dates: Optional[Iterable[Any]] = None,
    candidate_codes_by_date: Optional[Mapping[Any, Iterable[str]]] = None,
    sort_codes: bool = True,
) -> dict[pd.Timestamp, list[str]]:
    """
    Build valid code lists for each signal date.
    """
    _check_panel(feature_panel, "feature_panel")
    _check_panel(label_panel, "label_panel")
    validate_label_names(label_panel, label_name, label_valid_name)

    if signal_dates is None:
        f_dates = set(_unique_dates(feature_panel))
        l_dates = set(_unique_dates(label_panel))
        dates = sorted(f_dates & l_dates)
    else:
        dates = sorted(pd.Timestamp(x) for x in signal_dates)

    out: dict[pd.Timestamp, list[str]] = {}

    for dt in dates:
        candidate_codes = None
        if candidate_codes_by_date is not None:
            candidate_codes = _lookup_candidate_codes(candidate_codes_by_date, dt)

        codes = get_valid_codes_for_date(
            feature_panel=feature_panel,
            label_panel=label_panel,
            trade_date=dt,
            label_name=label_name,
            label_valid_name=label_valid_name,
            candidate_codes=candidate_codes,
            sort_codes=sort_codes,
        )
        out[pd.Timestamp(dt)] = codes

    return out


def _lookup_candidate_codes(
    candidate_codes_by_date: Mapping[Any, Iterable[str]],
    trade_date: pd.Timestamp,
) -> Optional[Iterable[str]]:
    """Lookup candidate codes with flexible date key handling."""
    if trade_date in candidate_codes_by_date:
        return candidate_codes_by_date[trade_date]

    ymd = trade_date.strftime("%Y%m%d")
    if ymd in candidate_codes_by_date:
        return candidate_codes_by_date[ymd]

    ymd_dash = trade_date.strftime("%Y-%m-%d")
    if ymd_dash in candidate_codes_by_date:
        return candidate_codes_by_date[ymd_dash]

    return None


def filter_signal_dates(
    all_dates: Sequence[pd.Timestamp],
    valid_codes_by_date: Mapping[pd.Timestamp, Sequence[str]],
    min_valid_stocks: int,
    seq_len: int,
    require_full_history: bool = True,
) -> list[pd.Timestamp]:
    """
    Keep signal dates with enough valid stocks and optional full history.
    """
    date_index = pd.DatetimeIndex(all_dates).sort_values()
    out: list[pd.Timestamp] = []

    for dt in date_index:
        valid_count = len(valid_codes_by_date.get(pd.Timestamp(dt), []))
        if valid_count < min_valid_stocks:
            continue

        if require_full_history:
            pos = date_index.get_loc(dt)
            if int(pos) < int(seq_len) - 1:
                continue

        out.append(pd.Timestamp(dt))

    return out


# ============================================================
# Tensor builders
# ============================================================

def build_time_series_array(
    feature_panel: pd.DataFrame,
    all_feature_dates: Sequence[pd.Timestamp],
    trade_date: Any,
    codes: Sequence[str],
    ts_factor_names: Sequence[str],
    seq_len: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Build x_ts array with shape [N, F_ts, T].

    The sequence ends at trade_date.
    Missing left history is padded with NaN and then filled.
    """
    _check_panel(feature_panel, "feature_panel")
    dt = pd.Timestamp(trade_date)
    date_index = pd.DatetimeIndex(all_feature_dates).sort_values()

    if dt not in date_index:
        raise KeyError(f"trade_date not found in feature dates: {dt}")

    pos = int(date_index.get_loc(dt))
    start_pos = max(0, pos - int(seq_len) + 1)
    seq_dates = date_index[start_pos:pos + 1]
    pad_len = int(seq_len) - len(seq_dates)

    codes = [str(c) for c in codes]
    n = len(codes)
    f_ts = len(ts_factor_names)

    if n == 0:
        return np.empty((0, f_ts, int(seq_len)), dtype=np.float32)

    if f_ts == 0:
        return np.empty((n, 0, int(seq_len)), dtype=np.float32)

    idx = pd.MultiIndex.from_product([seq_dates, codes], names=["trade_date", "ts_code"])
    sub = feature_panel.reindex(idx)[list(ts_factor_names)]
    arr = sub.to_numpy(dtype=float).reshape(len(seq_dates), n, f_ts)  # [T_actual, N, F_ts]
    arr = np.transpose(arr, (1, 2, 0))  # [N, F_ts, T_actual]

    if pad_len > 0:
        pad = np.full((n, f_ts, pad_len), np.nan, dtype=float)
        arr = np.concatenate([pad, arr], axis=2)

    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr.astype(np.float32)


def build_scalar_array(
    feature_panel: pd.DataFrame,
    trade_date: Any,
    codes: Sequence[str],
    scalar_factor_names: Sequence[str],
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Build x_scalar array with shape [N, F_scalar].
    """
    _check_panel(feature_panel, "feature_panel")
    dt = pd.Timestamp(trade_date)
    codes = [str(c) for c in codes]
    n = len(codes)
    f_scalar = len(scalar_factor_names)

    if n == 0:
        return np.empty((0, f_scalar), dtype=np.float32)

    if f_scalar == 0:
        return np.empty((n, 0), dtype=np.float32)

    idx = pd.MultiIndex.from_product([[dt], codes], names=["trade_date", "ts_code"])
    sub = feature_panel.reindex(idx)[list(scalar_factor_names)]
    arr = sub.to_numpy(dtype=float).reshape(n, f_scalar)
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr.astype(np.float32)


def build_label_arrays(
    label_panel: pd.DataFrame,
    trade_date: Any,
    codes: Sequence[str],
    label_name: str,
    target_mode: str = "rank_pct",
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build local target y and raw forward return arrays.

    Returns
    -------
    y:
        Local target according to target_mode, shape [N].

    y_raw:
        Raw forward return from label_name, shape [N].
    """
    _check_panel(label_panel, "label_panel")
    dt = pd.Timestamp(trade_date)
    codes = [str(c) for c in codes]

    idx = pd.MultiIndex.from_product([[dt], codes], names=["trade_date", "ts_code"])
    raw = label_panel.reindex(idx)[label_name]
    raw.index = codes
    raw = pd.to_numeric(raw, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)

    y = _local_target_from_returns(raw, target_mode=target_mode, ascending=True)

    y_raw_arr = raw.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)

    y_raw_arr = np.nan_to_num(y_raw_arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    y_arr = np.nan_to_num(y_arr, nan=fill_value, posinf=fill_value, neginf=fill_value)

    return y_arr.astype(np.float32), y_raw_arr.astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class CrossSectionRankDataset(Dataset):
    """
    Stochastic cross-sectional ranking dataset.

    Each item:
        - selects one signal date
        - samples sample_size stocks without replacement
        - returns tensors and metadata
    """

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        ts_factor_names: Sequence[str],
        scalar_factor_names: Sequence[str],
        config: CrossSectionDatasetConfig = CrossSectionDatasetConfig(),
        signal_dates: Optional[Iterable[Any]] = None,
        candidate_codes_by_date: Optional[Mapping[Any, Iterable[str]]] = None,
        valid_codes_by_date: Optional[Mapping[Any, Sequence[str]]] = None,
    ) -> None:
        _check_panel(feature_panel, "feature_panel")
        _check_panel(label_panel, "label_panel")

        if config.return_tensors == "torch":
            _ensure_torch_available()

        validate_factor_names(feature_panel, ts_factor_names, scalar_factor_names)
        validate_label_names(label_panel, config.label_name, config.label_valid_name)

        if int(config.sample_size) <= 0:
            raise ValueError("sample_size must be positive")
        if int(config.seq_len) <= 0:
            raise ValueError("seq_len must be positive")
        if int(config.samples_per_date) <= 0:
            raise ValueError("samples_per_date must be positive")

        self.feature_panel = feature_panel.sort_index()
        self.label_panel = label_panel.sort_index()
        self.ts_factor_names = list(ts_factor_names)
        self.scalar_factor_names = list(scalar_factor_names)
        self.config = config
        self.epoch = 0

        self.all_feature_dates = _unique_dates(self.feature_panel)

        if valid_codes_by_date is None:
            self.valid_codes_by_date = build_valid_codes_by_date(
                feature_panel=self.feature_panel,
                label_panel=self.label_panel,
                label_name=config.label_name,
                label_valid_name=config.label_valid_name,
                signal_dates=signal_dates,
                candidate_codes_by_date=candidate_codes_by_date,
                sort_codes=config.sort_codes,
            )
        else:
            self.valid_codes_by_date = {
                pd.Timestamp(k): _normalize_codes(v, sort_codes=config.sort_codes)
                for k, v in valid_codes_by_date.items()
            }

        min_valid = config.min_valid_stocks
        if min_valid is None:
            min_valid = config.sample_size if not config.allow_smaller_sample else 2

        raw_dates = sorted(self.valid_codes_by_date.keys()) if signal_dates is None else sorted(pd.Timestamp(x) for x in signal_dates)

        self.signal_dates = filter_signal_dates(
            all_dates=raw_dates,
            valid_codes_by_date=self.valid_codes_by_date,
            min_valid_stocks=int(min_valid),
            seq_len=int(config.seq_len),
            require_full_history=bool(config.require_full_history),
        )

        if len(self.signal_dates) == 0:
            raise ValueError(
                "No usable signal dates. Consider lowering min_valid_stocks, "
                "setting allow_smaller_sample=True, setting require_full_history=False, "
                "or checking label/feature panels."
            )

    @classmethod
    def from_bundle(
        cls,
        bundle: Any,
        config: CrossSectionDatasetConfig = CrossSectionDatasetConfig(),
        signal_dates: Optional[Iterable[Any]] = None,
        candidate_codes_by_date: Optional[Mapping[Any, Iterable[str]]] = None,
    ) -> "CrossSectionRankDataset":
        """
        Construct dataset from factor_pipeline.FeatureLabelBundle.
        """
        if not hasattr(bundle, "feature_panel") or not hasattr(bundle, "label_panel"):
            raise TypeError("bundle must have feature_panel and label_panel attributes")
        if not hasattr(bundle, "metadata"):
            raise TypeError("bundle must have metadata attribute")

        ts_names, scalar_names = infer_factor_groups_from_metadata(bundle.metadata)

        return cls(
            feature_panel=bundle.feature_panel,
            label_panel=bundle.label_panel,
            ts_factor_names=ts_names,
            scalar_factor_names=scalar_names,
            config=config,
            signal_dates=signal_dates,
            candidate_codes_by_date=candidate_codes_by_date,
        )

    def __len__(self) -> int:
        return len(self.signal_dates) * int(self.config.samples_per_date)

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for deterministic-but-changing stochastic sampling.
        """
        self.epoch = int(epoch)

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.config.deterministic:
            seed = int(self.config.random_seed) + int(idx)
        else:
            seed = int(self.config.random_seed) + int(idx) + 1_000_003 * int(self.epoch)
        return np.random.default_rng(seed)

    def _date_for_index(self, idx: int) -> pd.Timestamp:
        date_idx = int(idx) // int(self.config.samples_per_date)
        date_idx = date_idx % len(self.signal_dates)
        return pd.Timestamp(self.signal_dates[date_idx])

    def _sample_codes(self, dt: pd.Timestamp, rng: np.random.Generator) -> list[str]:
        codes = list(self.valid_codes_by_date.get(pd.Timestamp(dt), []))
        if len(codes) == 0:
            raise RuntimeError(f"no valid codes for date {dt}")

        # Optional candidate-pool downsampling, e.g. approximate a 2048-stock universe.
        if self.config.candidate_pool_size is not None and len(codes) > int(self.config.candidate_pool_size):
            idx = rng.choice(len(codes), size=int(self.config.candidate_pool_size), replace=False)
            codes = [codes[i] for i in idx]

        n_available = len(codes)
        n_sample = int(self.config.sample_size)

        if n_available < n_sample:
            if not self.config.allow_smaller_sample:
                raise RuntimeError(f"date {dt} has only {n_available} codes, sample_size={n_sample}")
            n_sample = n_available

        chosen_idx = rng.choice(n_available, size=n_sample, replace=False)
        chosen = [codes[i] for i in chosen_idx.tolist()]

        if self.config.sort_codes:
            chosen = sorted(chosen)

        return chosen

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dt = self._date_for_index(int(idx))
        rng = self._rng_for_index(int(idx))
        codes = self._sample_codes(dt, rng)

        x_ts = build_time_series_array(
            feature_panel=self.feature_panel,
            all_feature_dates=self.all_feature_dates,
            trade_date=dt,
            codes=codes,
            ts_factor_names=self.ts_factor_names,
            seq_len=int(self.config.seq_len),
            fill_value=float(self.config.fill_value),
        )

        x_scalar = build_scalar_array(
            feature_panel=self.feature_panel,
            trade_date=dt,
            codes=codes,
            scalar_factor_names=self.scalar_factor_names,
            fill_value=float(self.config.fill_value),
        )

        y, y_raw = build_label_arrays(
            label_panel=self.label_panel,
            trade_date=dt,
            codes=codes,
            label_name=self.config.label_name,
            target_mode=self.config.target_mode,
            fill_value=float(self.config.fill_value),
        )

        sample = {
            "x_ts": _convert_array(x_ts, self.config.return_tensors),
            "x_scalar": _convert_array(x_scalar, self.config.return_tensors),
            "y": _convert_array(y, self.config.return_tensors),
            "y_raw": _convert_array(y_raw, self.config.return_tensors),
            "codes": codes,
            "trade_date": dt,
            "date_index": self.signal_dates.index(dt),
            "sample_index": int(idx),
            "ts_factor_names": self.ts_factor_names,
            "scalar_factor_names": self.scalar_factor_names,
            "label_name": self.config.label_name,
            "target_mode": self.config.target_mode,
        }

        return sample

    def summary(self) -> pd.Series:
        """Return a compact dataset summary."""
        counts = pd.Series({dt: len(self.valid_codes_by_date.get(dt, [])) for dt in self.signal_dates})
        return pd.Series({
            "signal_date_count": len(self.signal_dates),
            "samples_per_date": int(self.config.samples_per_date),
            "dataset_length": len(self),
            "sample_size": int(self.config.sample_size),
            "seq_len": int(self.config.seq_len),
            "ts_factor_count": len(self.ts_factor_names),
            "scalar_factor_count": len(self.scalar_factor_names),
            "min_valid_codes": int(counts.min()),
            "median_valid_codes": float(counts.median()),
            "max_valid_codes": int(counts.max()),
        })


# ============================================================
# Collate and split helpers
# ============================================================

def cross_section_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate samples from CrossSectionRankDataset.

    Works for batch_size >= 1 if all samples have the same N.
    If allow_smaller_sample=True creates variable N, use batch_size=1.
    """
    if len(batch) == 0:
        return {}

    tensor_mode = torch is not None and isinstance(batch[0]["x_ts"], torch.Tensor)

    out: dict[str, Any] = {}

    for key in ("x_ts", "x_scalar", "y", "y_raw"):
        values = [item[key] for item in batch]
        if tensor_mode:
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = np.stack(values, axis=0)

    out["codes"] = [item["codes"] for item in batch]
    out["trade_date"] = [item["trade_date"] for item in batch]
    out["date_index"] = [item["date_index"] for item in batch]
    out["sample_index"] = [item["sample_index"] for item in batch]
    out["ts_factor_names"] = batch[0]["ts_factor_names"]
    out["scalar_factor_names"] = batch[0]["scalar_factor_names"]
    out["label_name"] = batch[0]["label_name"]
    out["target_mode"] = batch[0]["target_mode"]

    return out


def split_dates_by_time(
    dates: Sequence[Any],
    train_end: Optional[Any] = None,
    valid_end: Optional[Any] = None,
    valid_ratio: float = 0.2,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """
    Time-based train/valid split.

    If train_end is provided:
        train dates <= train_end
        valid dates > train_end and <= valid_end if provided

    Otherwise:
        last valid_ratio fraction is validation.
    """
    ds = sorted(pd.Timestamp(x) for x in dates)
    if len(ds) == 0:
        return [], []

    if train_end is not None:
        train_end_ts = pd.Timestamp(train_end)
        valid_end_ts = pd.Timestamp(valid_end) if valid_end is not None else None

        train = [d for d in ds if d <= train_end_ts]
        valid = [d for d in ds if d > train_end_ts and (valid_end_ts is None or d <= valid_end_ts)]
        return train, valid

    n_valid = max(1, int(round(len(ds) * float(valid_ratio))))
    n_train = max(0, len(ds) - n_valid)
    return ds[:n_train], ds[n_train:]


def make_train_valid_datasets(
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    ts_factor_names: Sequence[str],
    scalar_factor_names: Sequence[str],
    config: CrossSectionDatasetConfig,
    train_end: Optional[Any] = None,
    valid_end: Optional[Any] = None,
    valid_ratio: float = 0.2,
    candidate_codes_by_date: Optional[Mapping[Any, Iterable[str]]] = None,
) -> tuple[CrossSectionRankDataset, CrossSectionRankDataset]:
    """
    Build train and validation datasets with a time split.
    """
    valid_codes = build_valid_codes_by_date(
        feature_panel=feature_panel,
        label_panel=label_panel,
        label_name=config.label_name,
        label_valid_name=config.label_valid_name,
        candidate_codes_by_date=candidate_codes_by_date,
        sort_codes=config.sort_codes,
    )

    all_dates = sorted(valid_codes.keys())
    train_dates, valid_dates = split_dates_by_time(
        all_dates,
        train_end=train_end,
        valid_end=valid_end,
        valid_ratio=valid_ratio,
    )

    train_ds = CrossSectionRankDataset(
        feature_panel=feature_panel,
        label_panel=label_panel,
        ts_factor_names=ts_factor_names,
        scalar_factor_names=scalar_factor_names,
        config=config,
        signal_dates=train_dates,
        valid_codes_by_date={d: valid_codes[d] for d in train_dates},
    )

    valid_config = CrossSectionDatasetConfig(
        sample_size=config.sample_size,
        seq_len=config.seq_len,
        samples_per_date=1,
        label_name=config.label_name,
        label_valid_name=config.label_valid_name,
        target_mode=config.target_mode,
        require_full_history=config.require_full_history,
        min_valid_stocks=config.min_valid_stocks,
        allow_smaller_sample=config.allow_smaller_sample,
        fill_value=config.fill_value,
        return_tensors=config.return_tensors,
        random_seed=config.random_seed,
        deterministic=True,
        candidate_pool_size=config.candidate_pool_size,
        sort_codes=config.sort_codes,
    )

    valid_ds = CrossSectionRankDataset(
        feature_panel=feature_panel,
        label_panel=label_panel,
        ts_factor_names=ts_factor_names,
        scalar_factor_names=scalar_factor_names,
        config=valid_config,
        signal_dates=valid_dates,
        valid_codes_by_date={d: valid_codes[d] for d in valid_dates},
    )

    return train_ds, valid_ds


def make_train_valid_datasets_from_bundle(
    bundle: Any,
    config: CrossSectionDatasetConfig,
    train_end: Optional[Any] = None,
    valid_end: Optional[Any] = None,
    valid_ratio: float = 0.2,
    candidate_codes_by_date: Optional[Mapping[Any, Iterable[str]]] = None,
) -> tuple[CrossSectionRankDataset, CrossSectionRankDataset]:
    """Build train/valid datasets directly from a FeatureLabelBundle."""
    ts_names, scalar_names = infer_factor_groups_from_metadata(bundle.metadata)
    return make_train_valid_datasets(
        feature_panel=bundle.feature_panel,
        label_panel=bundle.label_panel,
        ts_factor_names=ts_names,
        scalar_factor_names=scalar_names,
        config=config,
        train_end=train_end,
        valid_end=valid_end,
        valid_ratio=valid_ratio,
        candidate_codes_by_date=candidate_codes_by_date,
    )


__all__ = [
    "CrossSectionDatasetConfig",
    "infer_factor_groups_from_metadata",
    "validate_factor_names",
    "validate_label_names",
    "get_date_subpanel",
    "get_valid_codes_for_date",
    "build_valid_codes_by_date",
    "filter_signal_dates",
    "build_time_series_array",
    "build_scalar_array",
    "build_label_arrays",
    "CrossSectionRankDataset",
    "cross_section_collate_fn",
    "split_dates_by_time",
    "make_train_valid_datasets",
    "make_train_valid_datasets_from_bundle",
]
