"""Pandera schema validation for normalised data frames.

Validation is defensive: it catches malformed upstream payloads early and produces
clear errors instead of silent bad numbers flowing into the optimiser.
"""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# A settlement period is 1..50 (50 on the long autumn DST day).
_SP_CHECK = Check.in_range(1, 50)

PRICE_FRAME_SCHEMA = DataFrameSchema(
    {
        "settlement_period": Column(int, _SP_CHECK, coerce=True),
        "price": Column(float, nullable=True, coerce=True),
    },
    strict=False,
    coerce=True,
)

SNAPSHOT_SCHEMA = DataFrameSchema(
    {
        "settlement_period": Column(int, _SP_CHECK, coerce=True),
        "wholesale_price": Column(float, nullable=True, coerce=True),
        "system_price": Column(float, nullable=True, coerce=True),
        "demand_forecast_mw": Column(float, nullable=True, coerce=True),
        "wind_forecast_mw": Column(float, nullable=True, coerce=True),
        "solar_forecast_mw": Column(float, nullable=True, coerce=True),
    },
    strict=False,
    coerce=True,
)


def validate_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    return PRICE_FRAME_SCHEMA.validate(df)


def validate_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    return SNAPSHOT_SCHEMA.validate(df)
