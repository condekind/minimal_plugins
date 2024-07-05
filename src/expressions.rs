#![allow(clippy::unused_unit)]

use crate::utils::binary_amortized_elementwise;
use polars::export::num::zero;
use std::borrow::Cow;
use std::fmt::Write;

use polars::prelude::arity::binary_elementwise;
use polars::prelude::*;

use polars_arrow::array::MutablePlString;
use polars_core::utils::align_chunks_binary;

use pyo3_polars::derive::polars_expr;
//use polars::export::num::Signed;
use pyo3_polars::export::polars_core::export::num::Signed;
use pyo3_polars::export::polars_core::utils::CustomIterTools;

use reverse_geocoder::ReverseGeocoder;
use rust_stemmers::{Algorithm, Stemmer};
use serde::Deserialize;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

#[polars_expr(output_type_func=same_output_type)]
fn noop(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    Ok(s.clone())
}

#[polars_expr(output_type=Int64)]
fn abs_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;
    let out = ca.apply(|opt_v: Option<i64>| opt_v.map(|v: i64| v.abs()));
    Ok(out.into_series())
}

fn impl_abs_numeric<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Signed,
{
    ca.apply(|opt_v: Option<T::Native>| opt_v.map(|v: T::Native| v.abs()))
}

#[polars_expr(output_type_func=same_output_type)]
fn abs_numeric(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let out = match s.dtype() {
        DataType::Int32 => impl_abs_numeric(s.i32().unwrap()).into_series(),
        DataType::Int64 => impl_abs_numeric(s.i64().unwrap()).into_series(),
        DataType::Float32 => impl_abs_numeric(s.f32().unwrap()).into_series(),
        DataType::Float64 => impl_abs_numeric(s.f64().unwrap()).into_series(),
        dtype => polars_bail!(InvalidOperation:format!("dtype {dtype} not supported")),
    };
    Ok(out)
}

#[polars_expr(output_type=Int64)]
fn sum_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let lhs = inputs[0].i64()?;
    let rhs = inputs[1].i64()?;
    let out: Int64Chunked =
        binary_elementwise(lhs, rhs, |lhs: Option<i64>, rhs: Option<i64>| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => Some(lhs + rhs),
                (Some(lhs), _) => Some(lhs),
                (_, Some(rhs)) => Some(rhs),
                _ => None,
            }
        });
    Ok(out.into_series())
}

fn impl_sum_numeric<T>(lhs: &ChunkedArray<T>, rhs: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Signed + std::ops::Add<Output = T::Native>,
{
    lhs + rhs
}

#[polars_expr(output_type_func=same_output_type)]
fn sum_numeric(inputs: &[Series]) -> PolarsResult<Series> {
    let lhs_is_float = match &inputs[0].dtype() {
        DataType::Float32 | DataType::Float64 => true,
        _ => false,
    };
    let rhs_is_float = match &inputs[1].dtype() {
        DataType::Float32 | DataType::Float64 => true,
        _ => false,
    };
    let res: Series;
    // If both are float, cast the Series to Float64, otherwise try Int64
    if lhs_is_float && rhs_is_float {
        let lhs = inputs[0].cast(&DataType::Float64);
        let rhs = inputs[1].cast(&DataType::Float64);
        res = match (lhs, rhs) {
            (Ok(lhs_f64), Ok(rhs_f64)) => {
                impl_sum_numeric(lhs_f64.f64()?, rhs_f64.f64()?).into_series()
            }
            _ => {
                polars_bail!(InvalidOperation:format!("Either lhs or rhs could not be converted to f64"))
            }
        };
    } else {
        let lhs = inputs[0].cast(&DataType::Int64);
        let rhs = inputs[1].cast(&DataType::Int64);
        res = match (lhs, rhs) {
            (Ok(lhs_i64), Ok(rhs_i64)) => {
                impl_sum_numeric(lhs_i64.i64()?, rhs_i64.i64()?).into_series()
            }
            _ => {
                polars_bail!(InvalidOperation:format!("Either lhs or rhs could not be converted to i64"))
            }
        };
    }
    Ok(res)
}

#[polars_expr(output_type_func=same_output_type)]
fn cum_sum(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;
    let out: Int64Chunked = ca
        .iter()
        .scan(0_i64, |state: &mut i64, elem: Option<i64>| match elem {
            Some(elem) => {
                *state += elem;
                Some(Some(*state))
            }
            None => Some(None),
        })
        .collect_trusted();
    let out = out.with_name(ca.name());
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn pig_latinnify_bad(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.str()?;
    let out: StringChunked = ca.apply(|opt_v: Option<&str>| {
        opt_v.map(|value: &str| {
            // Not the recommended way to do it,
            // see below for a better way!
            if let Some(first_char) = value.chars().next() {
                Cow::Owned(format!("{}{}ay", &value[1..], first_char))
            } else {
                Cow::Borrowed(value)
            }
        })
    });
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out = ca.apply_to_buffer(|value: &str, output: &mut String| {
        if let Some(first_char) = value.chars().next() {
            write!(output, "{}{}ay", &value[1..], first_char).unwrap()
        }
    });
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn snowball_stem(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let en_stemmer = Stemmer::create(Algorithm::English);
    let out = ca.apply_to_buffer(|value: &str, output: &mut String| {
        write!(output, "{}", en_stemmer.stem(value)).unwrap()
    });
    Ok(out.into_series())
}

/*
Stretch goal of chapter 6:
TODO: Browse through crates.io and pick a crate to make my own plugin out of
...
*/

/*
// Old abs_i64:
#[polars_expr(output_type=Int64)]
fn abs_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;
    let out = ca.apply(
        |opt_v: Option<i64>| opt_v.map(|v: i64| v.abs())
    );
    Ok(out.into_series())
}
*/

// Better version
#[polars_expr(output_type=Int64)]
fn abs_i64_fast(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;
    let out = ca.apply_values(|x| x.abs());
    Ok(out.into_series())
}

/*
// Old sum_i64:
#[polars_expr(output_type=Int64)]
fn sum_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let lhs = inputs[0].i64()?;
    let rhs = inputs[1].i64()?;
    let out: Int64Chunked = binary_elementwise(
        lhs,
        rhs,
        |lhs: Option<i64>, rhs: Option<i64>| match (lhs, rhs) {
            (Some(lhs), Some(rhs)) => Some(lhs + rhs),
            (Some(lhs), _) => Some(lhs),
            (_, Some(rhs)) => Some(rhs),
            _ => None,
        },
    );
    Ok(out.into_series())
}
 */

// Hopefully better (?)
#[polars_expr(output_type=Int64)]
fn sum_i64_fast(inputs: &[Series]) -> PolarsResult<Series> {
    let lhs = inputs[0].i64()?;
    let rhs = inputs[1].i64()?;
    let ca = lhs + rhs;
    Ok(ca.into_series())
}

#[derive(Deserialize)]
struct AddSuffixKwargs {
    suffix: String,
}

#[polars_expr(output_type=String)]
fn add_suffix(inputs: &[Series], kwargs: AddSuffixKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.str()?;
    let out = ca.apply_to_buffer(|value, output| {
        write!(output, "{}{}", value, kwargs.suffix).unwrap();
    });
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn weighted_mean(inputs: &[Series]) -> PolarsResult<Series> {
    // Why do I need & in the weights? (making it a &&ListChunked)
    let values = inputs[0].list()?;
    let weights = &inputs[1].list()?;

    let out: Float64Chunked =
        binary_amortized_elementwise(values, weights, |values_inner, weights_inner| {
            let values_inner = values_inner.i64().unwrap();
            let weights_inner = weights_inner.f64().unwrap();
            if values_inner.len() == 0 {
                return None;
            }
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            values_inner
                .iter()
                .zip(weights_inner.iter())
                .for_each(|(v, w)| {
                    if let (Some(v), Some(w)) = (v, w) {
                        numerator += v as f64 * w;
                        denominator += w;
                    }
                });
            Some(numerator / denominator)
        });
    Ok(out.into_series())
}

/*
// TODO: Could you implement a weighted standard deviation calculator?
// https://marcogorelli.github.io/polars-plugins-tutorial/lists/#gimme-chocolate-challenge
*/

fn list_idx_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(input_fields[0].name(), DataType::List(Box::new(IDX_DTYPE)));
    Ok(field.clone())
}

#[polars_expr(output_type_func=list_idx_dtype)]
fn non_zero_indices(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;

    let out: ListChunked = ca.apply_amortized(|s| {
        let s = s.as_ref();
        let ca = s.i64().unwrap();
        let out: IdxCa = ca
            .iter()
            .enumerate()
            .filter(|(_idx, opt_val)| opt_val != &Some(0))
            .map(|(idx, _opt_val)| Some(idx as IdxSize))
            .collect_ca("");
        out.into_series()
    });
    Ok(out.into_series())
}

fn shifted_struct(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.data_type() {
        DataType::Struct(fields) => {
            let mut field_0 = fields[0].clone();
            let name = field_0.name().clone();
            field_0.set_name(fields[fields.len() - 1].name().clone());
            let mut fields = fields[1..]
                .iter()
                .zip(fields[0..fields.len() - 1].iter())
                .map(|(fld, name)| Field::new(name.name(), fld.data_type().clone()))
                .collect::<Vec<_>>();
            fields.push(field_0);
            Ok(Field::new(&name, DataType::Struct(fields)))
        }
        _ => unreachable!(),
    }
}

#[polars_expr(output_type_func=shifted_struct)]
fn shift_struct(inputs: &[Series]) -> PolarsResult<Series> {
    let struct_ = inputs[0].struct_()?;
    let fields = struct_.fields();
    if fields.is_empty() {
        return Ok(inputs[0].clone());
    }
    let mut field_0 = fields[0].clone();
    field_0.rename(fields[fields.len() - 1].name());
    let mut fields = fields[1..]
        .iter()
        .zip(fields[..fields.len() - 1].iter())
        .map(|(s, name)| {
            let mut s = s.clone();
            s.rename(name.name());
            s
        })
        .collect::<Vec<_>>();
    fields.push(field_0);
    StructChunked::new(struct_.name(), &fields).map(|ca| ca.into_series())
}

#[polars_expr(output_type=String)]
fn reverse_geocode(inputs: &[Series]) -> PolarsResult<Series> {
    let lat = inputs[0].f64()?;
    let lon = inputs[1].f64()?;
    let geocoder = ReverseGeocoder::new();

    let (lhs, rhs) = align_chunks_binary(lat, lon);
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lat_arr, lon_arr)| {
            let mut mutarr = MutablePlString::with_capacity(lat_arr.len());

            for (lat_opt_val, lon_opt_val) in lat_arr.iter().zip(lon_arr.iter()) {
                match (lat_opt_val, lon_opt_val) {
                    (Some(lat_val), Some(lon_val)) => {
                        let res = &geocoder.search((*lat_val, *lon_val)).record.name;
                        mutarr.push(Some(res))
                    }
                    _ => mutarr.push_null(),
                }
            }

            mutarr.freeze().boxed()
        })
        .collect();
    let out: StringChunked = unsafe { ChunkedArray::from_chunks("placeholder", chunks) };
    Ok(out.into_series())
}

#[polars_expr(output_type=Int64)]
fn sum_of_row(inputs: &[Series]) -> PolarsResult<Series> {
    let first: &Int64Chunked = inputs[0].i64()?;
    let total = inputs
        .iter()
        .skip(1)
        .fold(first.clone(), |acc, series| &acc + series.i64().unwrap());
    Ok(total.into_series())
}

/*
// GPT attempt
#[polars_expr(output_type=Int64)]
fn sum_8_neighbors(inputs: &[Series]) -> PolarsResult<Series> {
    let current = inputs[1].i64()?;
    let prev = inputs.get(0).map(|s| s.i64().expect("Series to be of type Int64"));
    let next = inputs.get(2).map(|s| s.i64().expect("Series to be of type Int64"));
    let len = current.len();

    let mut result = Vec::with_capacity(len);

    // panics if 0..len
    for i in 1..len-1 {
        let mut sum = 0;
        for &neighbor in &[
            (i.checked_sub(1), prev.as_ref()),
            (Some(i), prev.as_ref()),
            (i.checked_add(1), prev.as_ref()),
            (i.checked_sub(1), Some(&current)),
            (i.checked_add(1), Some(&current)),
            (i.checked_sub(1), next.as_ref()),
            (Some(i), next.as_ref()),
            (i.checked_add(1), next.as_ref()),
        ] {
            if let (Some(idx), Some(col)) = neighbor {
                if let Some(value) = col.get(idx) {
                    sum += value;
                }
            }
        }
        result.push(sum);
    }

    Ok(Int64Chunked::new(inputs[1].name(), &result).into_series())
}
*/

/*
// Ref.
#[polars_expr(output_type_func=same_output_type)]
fn cum_sum(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;
    let out: Int64Chunked = ca
        .iter()
        .scan(0_i64, |state: &mut i64, elem: Option<i64>| match elem {
            Some(elem) => {
                *state += elem;
                Some(Some(*state))
            }
            None => Some(None),
        })
        .collect_trusted();
    let out = out.with_name(ca.name());
    Ok(out.into_series())
}
 */
#[polars_expr(output_type=Int64)]
fn sum_row_above(inputs: &[Series]) -> PolarsResult<Series> {
    // Note: GPT made this - from it, I was able to modify it. See fns below
    let s = &inputs[0];
    let ca = s.i64()?;
    let len = ca.len();
    let mut out: Int64Chunked = ca
        .into_no_null_iter()
        .enumerate()
        .map(|(idx, val)| {
            let prev = if 0 == idx {
                ca.get(len - 1).unwrap_or(0)
            } else {
                ca.get(idx - 1).unwrap_or(0)
            };
            Some(val + prev)
        })
        .collect_trusted();
    out.rename(ca.name());
    Ok(out.into_series())
}

#[polars_expr(output_type=Int64)]
fn sum_nbrs_above(inputs: &[Series]) -> PolarsResult<Series> {
    let (left, curr, right) = (&inputs[0], &inputs[1], &inputs[2]);
    let (ca_lf, ca_curr, ca_rt) = (left.i64()?, curr.i64()?, right.i64()?);
    let len = ca_curr.len();

    let mut out: Int64Chunked = ca_curr
        .into_no_null_iter()
        .enumerate()
        .map(|(idx, val)| {
            let prev = if 0 == idx {
                ca_lf.get(len - 1).unwrap_or(0)
                    + ca_curr.get(len - 1).unwrap_or(0)
                    + ca_rt.get(len - 1).unwrap_or(0)
            } else {
                ca_lf.get(idx - 1).unwrap_or(0)
                    + ca_curr.get(idx - 1).unwrap_or(0)
                    + ca_rt.get(idx - 1).unwrap_or(0)
            };
            Some(val + prev)
        })
        .collect_trusted();
    out.rename(ca_curr.name());
    Ok(out.into_series())
}

#[polars_expr(output_type=Int64)]
fn sum_nbrs(inputs: &[Series]) -> PolarsResult<Series> {
    let (left, curr, right) = (&inputs[0], &inputs[1], &inputs[2]);
    let (ca_lf, ca_curr, ca_rt) = (left.i64()?, curr.i64()?, right.i64()?);
    let len = ca_curr.len();

    let mut out: Int64Chunked = ca_curr
        .into_no_null_iter()
        .enumerate()
        .map(|(idx, val)| {
            let prev_row = if 0 == idx {
                ca_lf.get(len - 1).unwrap_or(0)
                    + ca_curr.get(len - 1).unwrap_or(0)
                    + ca_rt.get(len - 1).unwrap_or(0)
            } else {
                ca_lf.get(idx - 1).unwrap_or(0)
                    + ca_curr.get(idx - 1).unwrap_or(0)
                    + ca_rt.get(idx - 1).unwrap_or(0)
            };
            let curr_row = ca_lf.get(idx).unwrap_or(0) + ca_rt.get(idx).unwrap_or(0);
            let next_row = if len - 1 == idx {
                ca_lf.get(0).unwrap_or(0) + ca_curr.get(0).unwrap_or(0) + ca_rt.get(0).unwrap_or(0)
            } else {
                ca_lf.get(idx + 1).unwrap_or(0)
                    + ca_curr.get(idx + 1).unwrap_or(0)
                    + ca_rt.get(idx + 1).unwrap_or(0)
            };
            Some(match (val, prev_row + curr_row + next_row + val) {
                (1, alive) if alive == 3 || alive == 4 => 1,
                (0, alive) if alive == 3 => 1,
                _ => 0,
            })
            //Some(prev_row + curr_row + next_row)
        })
        .collect_trusted();
    out.rename(ca_curr.name());
    Ok(out.into_series())
}
