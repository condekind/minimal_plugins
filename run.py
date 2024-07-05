from itertools import tee, islice
import polars as pl
import minimal_plugin as mp
from minimal_plugin import pig_latinnify, noop, abs_i64, abs_numeric, sum_i64, sum_numeric, cum_sum

life_board_str = {
    '00': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '01': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '02': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '03': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '04': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '05': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","1","0","0","0","0","0","0","1","1","0","0","0","0","0","0","0","0","0","0","0","0","1","1","0","0","0","0"],
    '06': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","1","0","0","0","0","1","1","0","0","0","0","0","0","0","0","0","0","0","0","1","1","0","0","0","0"],
    '07': ["0","0","0","0","1","1","0","0","0","0","0","0","0","0","1","0","0","0","0","0","1","0","0","0","1","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '08': ["0","0","0","0","1","1","0","0","0","0","0","0","0","0","1","0","0","0","1","0","1","1","0","0","0","0","1","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '09': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","1","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '10': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '11': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '12': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '13': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    '14': ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
}
life_board = {
    '00': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '01': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '02': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '03': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '04': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '05': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    '06': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    '07': [0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '08': [0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '09': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '10': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '11': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '12': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '13': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    '14': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
}

df = pl.DataFrame({
    'english': ['this', 'is', 'not', 'pig', 'latin'],
})
result = df.with_columns(pig_latin = pig_latinnify('english'))
print(result)

df = pl.DataFrame({
    'a': [1, 1, None],
    'b': [4.1, 5.2, 6.3],
    'c': ['hello', 'everybody!', '!']
})
result = df.with_columns(
    noop(
        pl.all()
    ).name.suffix('_noop')
)
print(result)

df = pl.DataFrame({
    'a': [1, -1, None],
    'b': [4.1, 5.2, -6.3],
    'c': ['hello', 'everybody!', '!']
})
result = df.with_columns(abs_i64('a').name.suffix('_abs'))
print(result)

df = pl.DataFrame({
    'a': [1, -1, None],
    'b': [4.1, 5.2, -6.3],
    'c': ['hello', 'everybody!', '!']
})
result = df.with_columns(abs_numeric(pl.col('a', 'b')).name.suffix('_abs_num'))
print(result)

df = pl.DataFrame({
    'a': [1, 5, 2],
    'b': [3, None, -1]
})
result = df.with_columns(a_plus_b=sum_i64('a', 'b'))
print(result)

# Sum Numeric
df = pl.DataFrame({
    'a': [1.3, 0.2, 2],
    'b': [2.9, 4.0, 2.2]
})
result = df.with_columns(a_plus_b=sum_numeric('a', 'b'))
print(result)
df = pl.DataFrame({
    'a': [1, 0, 2],
    'b': [2, 4, 2]
})
result = df.with_columns(a_plus_b=sum_numeric('a', 'b'))
print(result)

df = pl.DataFrame({
    'a': [1, 5, 2],
    'b': [3, None, -1]
})
result = df.with_columns(cum_sum=cum_sum('b'))
print(result)

print("If is_elementwise is not properly set, the window for operations such "
    "as groupby is not respected\n")

df = pl.DataFrame({'a': ["I", "love", "pig", "latin"]})
print(df.with_columns(pig_latin_bad=mp.pig_latinnify_bad('a')))

df = pl.DataFrame({'a': ["I", "love", "pig", "latin"]})
print(df.with_columns(pig_latin_ok=mp.pig_latinnify_ok('a')))

df = pl.DataFrame({'word': ["fearlessly", "littleness", "lovingly", "devoted"]})
print(df.with_columns(b=mp.snowball_stem('word')))

df = pl.DataFrame({
    'a': [1, 5, 2],
    'b': [3, None, -1]
})
result = df.with_columns(a_plus_b_fast=mp.sum_i64_fast('a', 'b'))
print(result)

df = pl.DataFrame({'a': ['bob', 'billy']})
print(df.with_columns(a_with_suffix=mp.add_suffix('a', suffix='-billy')))

df = pl.DataFrame({
    'values': [[1, 3, 2], [5, 7]],
    'weights': [[.5, .3, .2], [.1, .9]]
})
print(df.with_columns(weighted_mean = mp.weighted_mean('values', 'weights')))

df = pl.DataFrame({'dense': [[0, 9], [8, 6, 0, 9], None, [3, 3]]})
print(df)
print(df.with_columns(indices=mp.non_zero_indices('dense')))

df = pl.DataFrame(
    {
        "a": [1, 3, 8],
        "b": [2.0, 3.1, 2.5],
        "c": ["3", "7", "3"],
    }
).select(abc=pl.struct("a", "b", "c"))
print(df.with_columns(abc_shifted=mp.shift_struct("abc")))

import pprint
pprint.pprint(df.with_columns(abc_shifted=mp.shift_struct("abc")).schema)
print('')

df = pl.DataFrame({
    'lat': [37.7749, 51.01, 52.5],
    'lon': [-122.4194, -3.9, -.91]
})
print(df.with_columns(city=mp.reverse_geocode('lat', 'lon')))

df = pl.DataFrame({
    'a': [1, 0, 2],
    'b': [2, 4, 2],
    'c': [3, 1, -4],
})
result = df.with_columns(total=mp.sum_of_row('a', 'b', 'c'))
print(result)

"""
# Impl. panics
df = pl.DataFrame({
    'a': [1, 0, 1],
    'b': [1, 1, 1],
    'c': [0, 1, 0],
})
result = df.with_columns(total=mp.sum_8_neighbors('a', 'b', 'c'))
print(result)
"""

df = pl.DataFrame({
    'a': [1, 0, 1, 0, 1, 1, 0, 1],
    'b': [3, 30, 10, 100, None, 17, 0, 7],
})
result = df.with_columns(sum_row_above=mp.sum_row_above('b'))
print(result)

df = pl.DataFrame({
    'a': [1, 0, 1, 0, 1, 1, 0, 1],
    'b': [3, 30, 10, 100, None, 17, 0, 7],
    'c': [0, 1, 0, 1, 1, 0, 1, 0],
})
result = df.with_columns(sum_nbrs=mp.sum_nbrs_above('a', 'b', 'c'))
print(result)

df = pl.DataFrame({
    'a': [1, 0, 1, 0, 1, 1, 0, 1],
    'b': [3, 30, 10, 100, None, 17, 0, 7],
    'c': [0, 1, 0, 1, 1, 0, 1, 0],
})
result = df.with_columns(sum_nbrs=mp.sum_nbrs('a', 'b', 'c'))
print(result)

with pl.Config(tbl_rows=-1, tbl_cols=-1):
    df = pl.DataFrame(life_board)
    result = df.with_columns(
        sum_08_09_10=mp.sum_nbrs('08', '09', '10'),
        sum_09_10_11=mp.sum_nbrs('09', '10', '11'),
    )
    print(result)

def nwise(iterable, n):
    """Return overlapping n-tuples from an iterable."""
    iterators = tee(iterable, n)
    return [list(z) for z in zip(*(islice(it, i, None) for i, it in enumerate(iterators)))]


# colnums: [['00', '01', '02'], ['01', '02', '03'], ... ]
colnums = nwise([f'{idx:02}' for idx in range(len(life_board))], 3)

# colnames: ['00_01_02', '01_02_03', '02_03_04', ... ]
colnames = ['_'.join(cols) for cols in colnums]

# colvalues: [<Expr ['col("00")./home/…'] at 0x7B7C253C7E60>, ... ]
colvalues = [mp.sum_nbrs(*tuple(cols)) for cols in colnums]

with pl.Config(tbl_rows=-1, tbl_cols=-1):
    df = pl.DataFrame(life_board)
    result = df.with_columns(
        **dict(zip(colnames, colvalues))
    )
    print(result)

# Question: how to expand kwargs in with_columns with keys:values?































