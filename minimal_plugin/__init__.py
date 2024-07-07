from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from minimal_plugin.utils import register_plugin, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

def pig_latinnify(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        symbol="pig_latinnify",
        is_elementwise=True,
        lib=lib,
    )

def noop(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="noop",
        is_elementwise=True,
    )

def abs_i64(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="abs_i64",
        is_elementwise=True,
    )

def abs_numeric(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="abs_numeric",
        is_elementwise=True,
    )

def sum_i64(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr, other],
        lib=lib,
        symbol="sum_i64",
        is_elementwise=True,
    )

def sum_numeric(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr, other],
        lib=lib,
        symbol="sum_numeric",
        is_elementwise=True,
    )

# Not element wise!
def cum_sum(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="cum_sum",
        is_elementwise=False,
    )

def pig_latinnify_bad(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="pig_latinnify_bad",
        is_elementwise=True,
    )

def pig_latinnify_ok(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="pig_latinnify",
        is_elementwise=True,
    )

def snowball_stem(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="snowball_stem",
        is_elementwise=True,
    )

def abs_i64_fast(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="abs_i64_fast",
        is_elementwise=True,
    )

def sum_i64_fast(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr, other],
        lib=lib,
        symbol="sum_i64_fast",
        is_elementwise=True,
    )

def add_suffix(expr: IntoExpr, *, suffix: str) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="add_suffix",
        is_elementwise=True,
        kwargs={"suffix": suffix},
    )

def weighted_mean(expr: IntoExpr, weights: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr, weights],
        lib=lib,
        symbol="weighted_mean",
        is_elementwise=True,
    )

def non_zero_indices(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="non_zero_indices",
        is_elementwise=True,
    )

def shift_struct(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="shift_struct",
        is_elementwise=True,
    )

def reverse_geocode(lat: IntoExpr, long: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[lat, long],
        lib=lib,
        symbol="reverse_geocode",
        is_elementwise=True,
    )

def sum_of_row(*expr) -> pl.Expr:
    return register_plugin(
        args=list(expr),
        lib=lib,
        symbol="sum_of_row",
        is_elementwise=True,
    )

"""
# Impl. panics
def sum_8_neighbors(*expr) -> pl.Expr:
    return register_plugin(
        args=list(expr),
        lib=lib,
        symbol="sum_8_neighbors",
        is_elementwise=True,
    )
"""

def sum_row_above(expr) -> pl.Expr:
    # Note: didn't see a difference passing expr vs [expr]
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="sum_row_above",
        is_elementwise=False,
    )

def sum_nbrs_above(left, mid, right) -> pl.Expr:
    return register_plugin(
        args=[left, mid, right],
        lib=lib,
        symbol="sum_nbrs_above",
        is_elementwise=False,
    )

def sum_nbrs(left, mid, right) -> pl.Expr:
    return register_plugin(
        args=[left, mid, right],
        lib=lib,
        symbol="sum_nbrs",
        is_elementwise=False,
    )

def iterate_life(left, mid, right) -> pl.Expr:
    return register_plugin(
        args=[left, mid, right],
        lib=lib,
        symbol="iterate_life",
        is_elementwise=False,
    )




































