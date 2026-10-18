"""Versioned, fail-closed configuration and immutable chronological splits."""

import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .io import digest, read_json


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, hide_input_in_errors=True)


class Period(StrictModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def chronological(self):
        if self.start.tzinfo or self.end.tzinfo:
            raise ValueError("Data timestamps must be timezone-naive exchange-local Asia/Shanghai time")
        if self.start >= self.end:
            raise ValueError("A split must have start < end (half-open)")
        return self


class Splits(StrictModel):
    train: Period
    validation: Period
    final: Period

    @model_validator(mode="after")
    def disjoint(self):
        if self.train.end > self.validation.start or self.validation.end > self.final.start:
            raise ValueError("Require train.end <= validation.start <= validation.end <= final.start")
        return self


class DataConfig(StrictModel):
    bars_path: Path
    labels_path: Path
    instruments_path: Path | None = None
    instrument_map_path: Path | None = None
    cache_dir: Path | None = None
    outside_universe_labels: Literal["exclude", "error"] = "exclude"
    history_days: int = Field(default=10, ge=0)
    allowed_start: datetime = datetime(2020, 1, 1)
    allowed_end: datetime = datetime(2025, 1, 1)
    feature_columns: tuple[str, ...] = ("close", "volume")
    lookback: int = Field(default=30, ge=1, le=10000)
    min_history: int = Field(default=1, ge=1)
    missing_policy: Literal["zero", "ffill"] = "ffill"
    max_ffill_steps: int = Field(default=5, ge=0)
    label_horizon_minutes: Literal[30] = 30
    sample_times: tuple[str, ...] = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
    training_sample_times: tuple[str, ...] | None = None
    price_scale: float = Field(default=1.0, gt=0)
    missing_sentinel: float | None = -1.0
    source: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_features(self):
        if self.allowed_start.tzinfo or self.allowed_end.tzinfo or self.allowed_start >= self.allowed_end:
            raise ValueError("Invalid allowed development interval")
        forbidden = {"target", "label", "vwap_return", "vwap", "end_price", "label_end", "date", "instrument"}
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be nonempty and unique")
        if forbidden.intersection(self.feature_columns):
            raise ValueError("Keys and future label fields must not be features")
        if self.min_history > self.lookback:
            raise ValueError("min_history exceeds lookback")
        if not self.sample_times or len(set(self.sample_times)) != len(self.sample_times):
            raise ValueError("sample_times must be nonempty and unique")
        if self.training_sample_times is not None and (
                not self.training_sample_times or len(set(self.training_sample_times)) != len(self.training_sample_times)
                or not set(self.training_sample_times).issubset(self.sample_times)):
            raise ValueError("training_sample_times must be a nonempty unique subset of sample_times")
        for t in self.sample_times:
            if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", t):
                raise ValueError("sample_times requires HH:MM")
            minutes = int(t[:2]) * 60 + int(t[3:])
            if not (570 <= minutes <= 660 or minutes == 690 or 780 <= minutes <= 870):
                raise ValueError("30-minute target requires a supported trading-clock time; "
                                 "11:30 uses the declared 13:30 lunch exception")
        return self


class ModelConfig(StrictModel):
    kind: Literal["ridge", "mean", "modified_tlob"] = "ridge"
    alpha: float = Field(default=1.0, gt=0)
    hidden_dim: int = Field(default=40, ge=4, le=512)
    num_layers: int = Field(default=4, ge=1, le=12)
    num_heads: int = Field(default=1, ge=1, le=8)
    is_sin_emb: bool = True
    epochs: int = Field(default=2, ge=1, le=1000)
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.0001, gt=0)
    target_scaling: Literal["none", "train_std"] = "none"
    time_loss_weighting: Literal["none", "train_variance", "date_decay_10d"] = "none"
    book_representation: Literal["raw", "relative_mid_bps"] = "raw"
    trade_features: Literal["zero", "log1p"] = "zero"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    cpu_threads: int = Field(default=1, ge=1, le=64)
    log_every_steps: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def valid_embedding(self):
        if self.kind != "modified_tlob" and (self.book_representation != "raw" or self.trade_features != "zero"):
            raise ValueError("LOB input transformations require modified_tlob")
        if self.kind == "modified_tlob" and self.is_sin_emb and self.hidden_dim % 2:
            raise ValueError("Sinusoidal embedding requires even hidden_dim")
        return self


class BudgetConfig(StrictModel):
    wall_seconds: float = Field(default=120, gt=0)
    max_rss_mb: float = Field(default=2048, gt=0)
    max_train_samples: int = Field(default=100000, ge=1)


class EvaluationConfig(StrictModel):
    quantile: float = Field(default=0.2, gt=0, le=0.5)
    periods_per_year: int = Field(default=252, ge=1)
    min_cross_section: int = Field(default=3, ge=2)
    constant_portfolios: Literal["neutral", "undefined"] = "neutral"


class AcceptanceConfig(StrictModel):
    min_coverage: float = Field(default=0.6, gt=0, le=1)
    repro_atol: float = Field(default=1e-10, ge=0)
    repro_rtol: float = Field(default=1e-8, ge=0)


class LoggingConfig(StrictModel):
    resource_interval_seconds: float = Field(default=5, gt=0)
    learning_curves: bool = False
    learning_curve_every_steps: int | None = Field(default=None, ge=1, strict=True)


class FinalAccess(StrictModel):
    purpose: str = ""
    max_views: int = Field(default=1, ge=1)


class RunConfig(StrictModel):
    schema_version: Literal[1] = 1
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    hypothesis: str = Field(min_length=1)
    output_root: Path = Path("runs")
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    data: DataConfig
    splits: Splits
    evaluation_split: Literal["validation", "final"] = "validation"
    model: ModelConfig = ModelConfig()
    budget: BudgetConfig = BudgetConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    acceptance: AcceptanceConfig = AcceptanceConfig()
    logging: LoggingConfig = LoggingConfig()
    final_access: FinalAccess = FinalAccess()

    @model_validator(mode="after")
    def final_is_deliberate(self):
        if self.model.kind == "modified_tlob":
            from modified_tlob.schema import validate_input_schema
            validate_input_schema(self.data.feature_columns, self.model.book_representation, self.model.trade_features)
            if self.data.lookback < 4:
                raise ValueError("modified_tlob requires lookback >= 4")
        if self.splits.train.start < self.data.allowed_start or self.splits.final.end > self.data.allowed_end:
            raise ValueError("All splits must stay within the declared allowed development interval")
        if self.evaluation_split == "final" and not self.final_access.purpose.strip():
            raise ValueError("Final split requires a recorded purpose; default is validation only")
        return self

    def snapshot(self) -> dict:
        return self.model_dump(mode="json")

    @property
    def checksum(self) -> str:
        return digest(self.snapshot())


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    raw = read_json(path)
    cfg = RunConfig.model_validate(raw)
    data = cfg.data.model_dump()
    for key in ("bars_path", "labels_path", "instruments_path", "instrument_map_path", "cache_dir"):
        if data[key] is not None:
            data[key] = (path.parent / data[key]).resolve()
    return RunConfig.model_validate({**cfg.model_dump(), "data": data,
                                     "output_root": (path.parent / cfg.output_root).resolve()})
