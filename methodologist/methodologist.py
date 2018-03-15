#!/usr/local/bin/python3
#
# methodologist.py
# Functions for static lexical analysis over economists' code.
# Uses Pygments, which miraculously has lexers for everything (except SPSS)!
#
# Author: Alex Kindel
# Last edited: 11 March 2018

import sys
from collections import defaultdict, Counter

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

from pygments import lex
from pygments.lexers import StataLexer, OctaveLexer, SASLexer, SLexer, NumPyLexer, FortranLexer
from pygments.token import Operator, Keyword, Name, Text
from pygments.lexer import words, include

# Improvements to Pygments lexers

# The SASLexer class was kinda busted
# Perhaps I'm the only one who's ever used it??
class SuperSASLexer(SASLexer):
    # Add missing function tokens
    builtins_functions = SASLexer.builtins_functions + ("rxchange", "rxparse")

    # Respecify regexes for operators, builtins
    tokens = SASLexer.tokens
    tokens['operators'] = [(r'(-|=|<=|>=|<|>|<>|&|!=|\||\*|\+|\^|/|!|~|~=)', Operator)]
    tokens['keywords'] = [(words(SASLexer.builtins_statements,
                           prefix=r'\b',
                           suffix=r'\b'),
                           Keyword),
                          (words(SASLexer.builtins_sql,
                           prefix=r'\b',
                           suffix=r'\b'),
                           Keyword),
                          (words(SASLexer.builtins_conditionals,
                           prefix=r'\b',
                           suffix=r'\b'),
                           Keyword),
                          (words(SASLexer.builtins_macros,
                           prefix=r'\%',
                           suffix=r'\b'),
                           Name.Builtin),
                          (words(builtins_functions,
                           prefix=r'\b'),
                           Name.Builtin),
                          ]
    tokens['general'] = [include('keywords'),
                         include('vars-strings'),
                         include('special'),
                         include('numbers'),
                         include('operators')
                         ]

# Enable more of the StataLexer class
# In general this could be improved to take better account of e.g. r() objects
class SuperStataLexer(StataLexer):
    tokens = StataLexer.tokens
    tokens["root"] = [include('comments'),
                      include('vars-strings'),
                      include('numbers'),
                      include('keywords'),
                      include('operators'),
                      (r'.', Text)
                      ]

# NumPyLexer doesn't have Pandas or Scipy names in it
# We also add names from pandas.DataFrame
# TODO: Bootstrap this the way DynamicRLexer works
class SuperNumPyLexer(NumPyLexer):
    pandas_names = ['Categorical', 'CategoricalIndex', 'DataFrame', 'DateOffset', 'DatetimeIndex', 'ExcelFile', 'ExcelWriter', 'Expr', 'Float64Index', 'Grouper', 'HDFStore', 'Index', 'IndexSlice', 'Int64Index', 'Interval', 'IntervalIndex', 'MultiIndex', 'NaT', 'Panel', 'Panel4D', 'Period', 'PeriodIndex', 'RangeIndex', 'Series', 'SparseArray', 'SparseDataFrame', 'SparseList', 'SparseSeries', 'Term', 'TimeGrouper', 'Timedelta', 'TimedeltaIndex', 'Timestamp', 'UInt64Index', 'WidePanel', '_DeprecatedModule', '__builtins__', '__cached__', '__doc__', '__docformat__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '__version__', '_hashtable', '_lib', '_libs', '_np_version_under1p10', '_np_version_under1p11', '_np_version_under1p12', '_np_version_under1p13', '_np_version_under1p14', '_np_version_under1p15', '_tslib', '_version', 'api', 'bdate_range', 'compat', 'concat', 'core', 'crosstab', 'cut', 'date_range', 'datetime', 'datetools', 'describe_option', 'errors', 'eval', 'ewma', 'ewmcorr', 'ewmcov', 'ewmstd', 'ewmvar', 'ewmvol', 'expanding_apply', 'expanding_corr', 'expanding_count', 'expanding_cov', 'expanding_kurt', 'expanding_max', 'expanding_mean', 'expanding_median', 'expanding_min', 'expanding_quantile', 'expanding_skew', 'expanding_std', 'expanding_sum', 'expanding_var', 'factorize', 'get_dummies', 'get_option', 'get_store', 'groupby', 'infer_freq', 'interval_range', 'io', 'isna', 'isnull', 'json', 'lib', 'lreshape', 'match', 'melt', 'merge', 'merge_asof', 'merge_ordered', 'notna', 'notnull', 'np', 'offsets', 'option_context', 'options', 'ordered_merge', 'pandas', 'parser', 'period_range', 'pivot', 'pivot_table', 'plot_params', 'plotting', 'pnow', 'qcut', 'read_clipboard', 'read_csv', 'read_excel', 'read_feather', 'read_fwf', 'read_gbq', 'read_hdf', 'read_html', 'read_json', 'read_msgpack', 'read_parquet', 'read_pickle', 'read_sas', 'read_sql', 'read_sql_query', 'read_sql_table', 'read_stata', 'read_table', 'reset_option', 'rolling_apply', 'rolling_corr', 'rolling_count', 'rolling_cov', 'rolling_kurt', 'rolling_max', 'rolling_mean', 'rolling_median', 'rolling_min', 'rolling_quantile', 'rolling_skew', 'rolling_std', 'rolling_sum', 'rolling_var', 'rolling_window', 'scatter_matrix', 'set_eng_float_format', 'set_option', 'show_versions', 'stats', 'test', 'testing', 'timedelta_range', 'to_datetime', 'to_msgpack', 'to_numeric', 'to_pickle', 'to_timedelta', 'tools', 'tseries', 'tslib', 'unique', 'util', 'value_counts', 'wide_to_long']

    pandas_df_names = ['T', '_AXIS_ALIASES', '_AXIS_IALIASES', '_AXIS_LEN', '_AXIS_NAMES', '_AXIS_NUMBERS', '_AXIS_ORDERS', '_AXIS_REVERSED', '_AXIS_SLICEMAP', '__abs__', '__add__', '__and__', '__array__', '__array_wrap__', '__bool__', '__bytes__', '__class__', '__contains__', '__copy__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__div__', '__doc__', '__eq__', '__finalize__', '__floordiv__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__', '__imod__', '__imul__', '__init__', '__init_subclass__', '__invert__', '__ior__', '__ipow__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__lt__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__nonzero__', '__or__', '__pow__', '__radd__', '__rand__', '__rdiv__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__unicode__', '__weakref__', '__xor__', '_accessors', '_add_numeric_operations', '_add_series_only_operations', '_add_series_or_dataframe_operations', '_agg_by_level', '_agg_doc', '_aggregate', '_aggregate_multiple_funcs', '_align_frame', '_align_series', '_apply_broadcast', '_apply_empty_result', '_apply_raw', '_apply_standard', '_at', '_box_col_values', '_box_item_values', '_builtin_table', '_check_inplace_setting', '_check_is_chained_assignment_possible', '_check_percentile', '_check_setitem_copy', '_clear_item_cache', '_clip_with_one_bound', '_clip_with_scalar', '_combine_const', '_combine_frame', '_combine_match_columns', '_combine_match_index', '_combine_series', '_combine_series_infer', '_compare_frame', '_compare_frame_evaluate', '_consolidate', '_consolidate_inplace', '_construct_axes_dict', '_construct_axes_dict_for_slice', '_construct_axes_dict_from', '_construct_axes_from_arguments', '_constructor', '_constructor_expanddim', '_constructor_sliced', '_convert', '_count_level', '_create_indexer', '_cython_table', '_deprecations', '_dir_additions', '_dir_deletions', '_drop_axis', '_ensure_valid_index', '_expand_axes', '_flex_compare_frame', '_from_arrays', '_from_axes', '_get_agg_axis', '_get_axis', '_get_axis_name', '_get_axis_number', '_get_axis_resolvers', '_get_block_manager_axis', '_get_bool_data', '_get_cacher', '_get_index_resolvers', '_get_item_cache', '_get_numeric_data', '_get_valid_indices', '_get_value', '_get_values', '_getitem_array', '_getitem_column', '_getitem_frame', '_getitem_multilevel', '_getitem_slice', '_gotitem', '_iat', '_iget_item_cache', '_iloc', '_indexed_same', '_info_axis', '_info_axis_name', '_info_axis_number', '_info_repr', '_init_dict', '_init_mgr', '_init_ndarray', '_internal_names', '_internal_names_set', '_is_builtin_func', '_is_cached', '_is_cython_func', '_is_datelike_mixed_type', '_is_mixed_type', '_is_numeric_mixed_type', '_is_view', '_ix', '_ixs', '_join_compat', '_loc', '_maybe_cache_changed', '_maybe_update_cacher', '_metadata', '_needs_reindex_multi', '_obj_with_exclusions', '_protect_consolidate', '_reduce', '_reindex_axes', '_reindex_axis', '_reindex_columns', '_reindex_index', '_reindex_multi', '_reindex_with_indexers', '_repr_data_resource_', '_repr_fits_horizontal_', '_repr_fits_vertical_', '_repr_html_', '_repr_latex_', '_reset_cache', '_reset_cacher', '_sanitize_column', '_selected_obj', '_selection', '_selection_list', '_selection_name', '_series', '_set_as_cached', '_set_axis', '_set_axis_name', '_set_is_copy', '_set_item', '_set_value', '_setitem_array', '_setitem_frame', '_setitem_slice', '_setup_axes', '_shallow_copy', '_slice', '_stat_axis', '_stat_axis_name', '_stat_axis_number', '_take', '_to_dict_of_blocks', '_try_aggregate_string_function', '_typ', '_unpickle_frame_compat', '_unpickle_matrix_compat', '_update_inplace', '_validate_dtype', '_values', '_where', '_xs', 'abs', 'add', 'add_prefix', 'add_suffix', 'agg', 'aggregate', 'align', 'all', 'any', 'append', 'apply', 'applymap', 'as_blocks', 'as_matrix', 'asfreq', 'asof', 'assign', 'astype', 'at', 'at_time', 'axes', 'between_time', 'bfill', 'blocks', 'bool', 'boxplot', 'clip', 'clip_lower', 'clip_upper', 'columns', 'combine', 'combine_first', 'compound', 'consolidate', 'convert_objects', 'copy', 'corr', 'corrwith', 'count', 'cov', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'div', 'divide', 'dot', 'drop', 'drop_duplicates', 'dropna', 'dtypes', 'duplicated', 'empty', 'eq', 'equals', 'eval', 'ewm', 'expanding', 'ffill', 'fillna', 'filter', 'first', 'first_valid_index', 'floordiv', 'from_csv', 'from_dict', 'from_items', 'from_records', 'ftypes', 'ge', 'get', 'get_dtype_counts', 'get_ftype_counts', 'get_value', 'get_values', 'groupby', 'gt', 'head', 'hist', 'iat', 'idxmax', 'idxmin', 'iloc', 'infer_objects', 'info', 'insert', 'interpolate', 'is_copy', 'isin', 'isna', 'isnull', 'items', 'iteritems', 'iterrows', 'itertuples', 'ix', 'join', 'keys', 'kurt', 'kurtosis', 'last', 'last_valid_index', 'le', 'loc', 'lookup', 'lt', 'mad', 'mask', 'max', 'mean', 'median', 'melt', 'memory_usage', 'merge', 'min', 'mod', 'mode', 'mul', 'multiply', 'ndim', 'ne', 'nlargest', 'notna', 'notnull', 'nsmallest', 'nunique', 'pct_change', 'pipe', 'pivot', 'pivot_table', 'plot', 'pop', 'pow', 'prod', 'product', 'quantile', 'query', 'radd', 'rank', 'rdiv', 'reindex', 'reindex_axis', 'reindex_like', 'rename', 'rename_axis', 'reorder_levels', 'replace', 'resample', 'reset_index', 'rfloordiv', 'rmod', 'rmul', 'rolling', 'round', 'rpow', 'rsub', 'rtruediv', 'sample', 'select', 'select_dtypes', 'sem', 'set_axis', 'set_index', 'set_value', 'shape', 'shift', 'size', 'skew', 'slice_shift', 'sort_index', 'sort_values', 'sortlevel', 'squeeze', 'stack', 'std', 'style', 'sub', 'subtract', 'sum', 'swapaxes', 'swaplevel', 'tail', 'take', 'to_clipboard', 'to_csv', 'to_dense', 'to_dict', 'to_excel', 'to_feather', 'to_gbq', 'to_hdf', 'to_html', 'to_json', 'to_latex', 'to_msgpack', 'to_panel', 'to_parquet', 'to_period', 'to_pickle', 'to_records', 'to_sparse', 'to_sql', 'to_stata', 'to_string', 'to_timestamp', 'to_xarray', 'transform', 'transpose', 'truediv', 'truncate', 'tshift', 'tz_convert', 'tz_localize', 'unstack', 'update', 'values', 'var', 'where', 'xs']
    # Removed 'index' (kwarg)

    scipy_names = ['ALLOW_THREADS', 'AxisError', 'BUFSIZE', 'CLIP', 'ComplexWarning', 'DataSource', 'ERR_CALL', 'ERR_DEFAULT', 'ERR_IGNORE', 'ERR_LOG', 'ERR_PRINT', 'ERR_RAISE', 'ERR_WARN', 'FLOATING_POINT_SUPPORT', 'FPE_DIVIDEBYZERO', 'FPE_INVALID', 'FPE_OVERFLOW', 'FPE_UNDERFLOW', 'False_', 'Inf', 'Infinity', 'LowLevelCallable', 'MAXDIMS', 'MAY_SHARE_BOUNDS', 'MAY_SHARE_EXACT', 'MachAr', 'ModuleDeprecationWarning', 'NAN', 'NINF', 'NZERO', 'NaN', 'PINF', 'PZERO', 'PackageLoader', 'RAISE', 'RankWarning', 'SHIFT_DIVIDEBYZERO', 'SHIFT_INVALID', 'SHIFT_OVERFLOW', 'SHIFT_UNDERFLOW', 'ScalarType', 'TooHardError', 'True_', 'UFUNC_BUFSIZE_DEFAULT', 'UFUNC_PYVALS_NAME', 'VisibleDeprecationWarning', 'WRAP', '__SCIPY_SETUP__', '__all__', '__builtins__', '__cached__', '__config__', '__doc__', '__file__', '__loader__', '__name__', '__numpy_version__', '__package__', '__path__', '__spec__', '__version__', '_distributor_init', '_lib', 'absolute', 'absolute_import', 'add', 'add_docstring', 'add_newdoc', 'add_newdoc_ufunc', 'add_newdocs', 'alen', 'all', 'allclose', 'alltrue', 'amax', 'amin', 'angle', 'any', 'append', 'apply_along_axis', 'apply_over_axes', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argpartition', 'argsort', 'argwhere', 'around', 'array', 'array2string', 'array_equal', 'array_equiv', 'array_repr', 'array_split', 'array_str', 'asanyarray', 'asarray', 'asarray_chkfinite', 'ascontiguousarray', 'asfarray', 'asfortranarray', 'asmatrix', 'asscalar', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'bartlett', 'base_repr', 'binary_repr', 'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'blackman', 'block', 'bmat', 'bool8', 'bool_', 'broadcast', 'broadcast_arrays', 'broadcast_to', 'busday_count', 'busday_offset', 'busdaycalendar', 'byte', 'byte_bounds', 'bytes0', 'bytes_', 'c_', 'can_cast', 'cast', 'cbrt', 'cdouble', 'ceil', 'cfloat', 'char', 'character', 'chararray', 'choose', 'clip', 'clongdouble', 'clongfloat', 'column_stack', 'common_type', 'compare_chararrays', 'complex128', 'complex256', 'complex64', 'complex_', 'complexfloating', 'compress', 'concatenate', 'conj', 'conjugate', 'convolve', 'copy', 'copysign', 'copyto', 'corrcoef', 'correlate', 'cos', 'cosh', 'count_nonzero', 'cov', 'cross', 'csingle', 'ctypeslib', 'cumprod', 'cumproduct', 'cumsum', 'datetime64', 'datetime_as_string', 'datetime_data', 'deg2rad', 'degrees', 'delete', 'deprecate', 'deprecate_with_doc', 'diag', 'diag_indices', 'diag_indices_from', 'diagflat', 'diagonal', 'diff', 'digitize', 'disp', 'divide', 'division', 'divmod', 'dot', 'double', 'dsplit', 'dstack', 'dtype', 'e', 'ediff1d', 'einsum', 'einsum_path', 'emath', 'empty', 'empty_like', 'equal', 'errstate', 'euler_gamma', 'exp', 'exp2', 'expand_dims', 'expm1', 'extract', 'eye', 'fabs', 'fastCopyAndTranspose', 'fft', 'fill_diagonal', 'find_common_type', 'finfo', 'fix', 'flatiter', 'flatnonzero', 'flexible', 'flip', 'fliplr', 'flipud', 'float128', 'float16', 'float32', 'float64', 'float_', 'float_power', 'floating', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod', 'format_float_positional', 'format_float_scientific', 'format_parser', 'frexp', 'frombuffer', 'fromfile', 'fromfunction', 'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'full', 'full_like', 'fv', 'generic', 'genfromtxt', 'geomspace', 'get_array_wrap', 'get_include', 'get_printoptions', 'getbufsize', 'geterr', 'geterrcall', 'geterrobj', 'gradient', 'greater', 'greater_equal', 'half', 'hamming', 'hanning', 'heaviside', 'histogram', 'histogram2d', 'histogramdd', 'hsplit', 'hstack', 'hypot', 'i0', 'identity', 'ifft', 'iinfo', 'imag', 'in1d', 'index_exp', 'indices', 'inexact', 'inf', 'info', 'infty', 'inner', 'insert', 'int0', 'int16', 'int32', 'int64', 'int8', 'int_', 'int_asbuffer', 'intc', 'integer', 'interp', 'intersect1d', 'intp', 'invert', 'ipmt', 'irr', 'is_busday', 'isclose', 'iscomplex', 'iscomplexobj', 'isfinite', 'isfortran', 'isin', 'isinf', 'isnan', 'isnat', 'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar', 'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'iterable', 'ix_', 'kaiser', 'kron', 'ldexp', 'left_shift', 'less', 'less_equal', 'lexsort', 'linspace', 'little_endian', 'load', 'loads', 'loadtxt', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logn', 'logspace', 'long', 'longcomplex', 'longdouble', 'longfloat', 'longlong', 'lookfor', 'ma', 'mafromtxt', 'mask_indices', 'mat', 'math', 'matmul', 'matrix', 'maximum', 'maximum_sctype', 'may_share_memory', 'mean', 'median', 'memmap', 'meshgrid', 'mgrid', 'min_scalar_type', 'minimum', 'mintypecode', 'mirr', 'mod', 'modf', 'moveaxis', 'msort', 'multiply', 'nan', 'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanstd', 'nansum', 'nanvar', 'nbytes', 'ndarray', 'ndenumerate', 'ndfromtxt', 'ndim', 'ndindex', 'nditer', 'negative', 'nested_iters', 'newaxis', 'nextafter', 'nonzero', 'not_equal', 'nper', 'npv', 'number', 'obj2sctype', 'object0', 'object_', 'ogrid', 'ones', 'ones_like', 'outer', 'packbits', 'pad', 'partition', 'percentile', 'pi', 'piecewise', 'pkgload', 'place', 'pmt', 'poly', 'poly1d', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint', 'polymul', 'polysub', 'polyval', 'positive', 'power', 'ppmt', 'print_function', 'prod', 'product', 'promote_types', 'ptp', 'put', 'putmask', 'pv', 'r_', 'rad2deg', 'radians', 'rand', 'randn', 'random', 'rank', 'rate', 'ravel', 'ravel_multi_index', 'real', 'real_if_close', 'rec', 'recarray', 'recfromcsv', 'recfromtxt', 'reciprocal', 'record', 'remainder', 'repeat', 'require', 'reshape', 'resize', 'result_type', 'right_shift', 'rint', 'roll', 'rollaxis', 'roots', 'rot90', 'round_', 'row_stack', 's_', 'safe_eval', 'save', 'savetxt', 'savez', 'savez_compressed', 'sctype2char', 'sctypeDict', 'sctypeNA', 'sctypes', 'searchsorted', 'select', 'set_numeric_ops', 'set_printoptions', 'set_string_function', 'setbufsize', 'setdiff1d', 'seterr', 'seterrcall', 'seterrobj', 'setxor1d', 'shape', 'shares_memory', 'short', 'show_config', 'show_numpy_config', 'sign', 'signbit', 'signedinteger', 'sin', 'sinc', 'single', 'singlecomplex', 'sinh', 'size', 'sometrue', 'sort', 'sort_complex', 'source', 'spacing', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'std', 'str0', 'str_', 'string_', 'subtract', 'sum', 'swapaxes', 'take', 'tan', 'tanh', 'tensordot', 'test', 'tile', 'timedelta64', 'trace', 'tracemalloc_domain', 'transpose', 'trapz', 'tri', 'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros', 'triu', 'triu_indices', 'triu_indices_from', 'true_divide', 'trunc', 'typeDict', 'typeNA', 'typecodes', 'typename', 'ubyte', 'ufunc', 'uint', 'uint0', 'uint16', 'uint32', 'uint64', 'uint8', 'uintc', 'uintp', 'ulonglong', 'unicode', 'unicode_', 'union1d', 'unique', 'unpackbits', 'unravel_index', 'unsignedinteger', 'unwrap', 'ushort', 'vander', 'var', 'vdot', 'vectorize', 'version', 'void', 'void0', 'vsplit', 'vstack', 'where', 'who', 'zeros', 'zeros_like']

    EXTRA_KEYWORDS = set(list(NumPyLexer.EXTRA_KEYWORDS) + pandas_names + pandas_df_names + scipy_names)

# Minor change to get the operator lexer working
class SuperFortranFixedLexer(FortranLexer):
    tokens = FortranLexer.tokens
    tokens['root'].insert(0, (r'\.(eq|ne|lt|le|gt|ge|not|and|or|eqv|neqv)\.', Operator))

# R lexer with namespace boostrapping from file imports
class DynamicRLexer(SLexer):
    # Set base keyword tokenizer
    tokens = SLexer.tokens
    tokens['keywords'] = [(words(SLexer.builtins_base, suffix=r'(?![\w. =])'), Keyword.Pseudo),
                          (r'(if|else|for|while|repeat|in|next|break|return|switch|function)'
                           r'(?![\w.])',
                           Keyword.Reserved),
                          (r'(array|category|character|complex|double|function|integer|list|'
                           r'logical|matrix|numeric|vector|data.frame|c)'
                           r'(?![\w.])',
                           Keyword.Type),
                          (r'(?<=(library|require)\()\w*(?=\))',
                           Keyword.Namespace)]

    builtins_base = SLexer.builtins_base
    EXTRA_KEYWORDS = set(list(builtins_base))

    def get_tokens_unprocessed(self, text):
        for index, token, value in SLexer.get_tokens_unprocessed(self, text):
            if token is Text and value in self.EXTRA_KEYWORDS:
                # Rescue misclassified tokens from imports
                yield index, Keyword.Pseudo, value
            else:
                yield index, token, value

    def expand_namespace(self, pkg):
        # Set up to install R packages as needed
        base = importr('base')
        utils = importr('utils')
        devtools = importr('devtools')
        utils.chooseCRANmirror(ind=1)

        # Load or install as necessary
        pkg_ix = None
        try:
            pkg_ix = importr(pkg, on_conflict="warn")
        except Exception:
            utils.install_packages(StrVector([pkg]))
            try:
                pkg_ix = importr(pkg)
            except Exception:
                if pkg in ["randomForestCI", "causalForest"]:
                    try:
                        robjects.r('install_github("swager/{}")'.format(pkg))
                        pkg_ix = importr(pkg)
                    except Exception:
                        pass  # Give up

        # List names in this package
        ls = []
        try:
            ls = list(base.ls("package:{}".format(pkg)))
        except Exception:
            print("Could not get names for R package: {}.".format(pkg))
        utility = list(base.ls("package:{}".format('utils')))

        # Add names to keyword list
        self.EXTRA_KEYWORDS = set(list(self.EXTRA_KEYWORDS) + ls + utility)



# Encoding methods

def encode_stata(filename, noisy):
    stata_methods = ["Token.Keyword", "Token.Name.Function", "Token.Operator"]

    # Read file
    stata_code = None
    with open(filename, 'r', encoding="utf8", errors="ignore") as f:
        stata_code = f.read()

    # Lex code
    tokens = lex(stata_code, SuperStataLexer())

    # Classify tokens by type
    tks = defaultdict(list)
    script_order = []
    for tktype, value in tokens:
        val = value.replace("\t", "").replace("\n", "").replace(' ', "")  # Strip whitespace

        # We generally want to know what a user looked at in the response object
        if str(tktype) == "Token.Name.Function":
            if val == "r(":
                nv = None
                while nv != ")":
                    try:
                        tp, nv = next(tokens)
                        val += nv
                    except StopIteration:
                        print(filename)
                        print(val)
            else:
                val = val.rstrip("(")

        # Group consecutive text-type tokens
        # TODO: I'm not 100% satisfied that I'm not losing information here -- come back to this later
        # if str(tktype) == "Token.Text":
        #     tp = str(tktype)
        #     while str(tp) == "Token.Text":
        #         tp, nv =
        #         val += nv

        # Store this token by type and by order, if interesting
        tks[str(tktype)].append(val)
        if str(tktype) in stata_methods:
            script_order.append(val)

    # Print token counts and script order
    if noisy:
        print("Token typology")
        print(tks.keys())
        print()
        for tp in r_methods:
            print(tp)
            print(Counter(tks[tp]))
            print()
        print('Reduced script:')
        print(script_order)
        print()

    # Return reduced-form script length
    return script_order


def encode_matlab(filename, noisy):
    matlab_methods = ['Token.Operator', 'Token.Name.Builtin', 'Token.Keyword']

    # Read file
    matlab_code = None
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        matlab_code = f.read()

    # Lex code
    # OctaveLexer seems to have better performance than MatlabLexer?
    tokens = lex(matlab_code, OctaveLexer())

    # Classify tokens by type
    tks = defaultdict(list)
    script_order = []
    for tktype, value in tokens:

        # TODO: I'd like to get a better sense for how user-defined operations are being used
        # i.e. if a user-defined function is reused, maybe I should count the operators twice?

        # Store token by type and order
        tks[str(tktype)].append(value)
        if str(tktype) in matlab_methods:
            script_order.append(value)

    # Print token counts and script order
    if noisy:
        print("Token typology")
        print(tks.keys())
        print()
        for tp in r_methods:
            print(tp)
            print(Counter(tks[tp]))
            print()
        print('Reduced script:')
        print(script_order)
        print()

    # Return reduced-form script length
    return script_order


def encode_sas(filename, noisy):
    sas_methods = ['Token.Keyword', "Token.Keyword.Reserved", "Token.Operator", "Token.Name.Function", "Token.Name.Builtin"]

    # Read file
    sas_code = None
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        sas_code = f.read()

    # Lex code
    tokens = lex(sas_code, SuperSASLexer())

    # Classify tokens by type
    tks = defaultdict(list)
    script_order = []
    for tktype, value in tokens:
        val = value.replace("\t", "").replace("\n", "").replace(' ', "").replace(';', "")  # Strip whitespace

        # Store token by type and order
        tks[str(tktype)].append(val)
        if str(tktype) in sas_methods:
            script_order.append(val)

    # Print token counts and script order
    if noisy:
        print("Token typology")
        print(tks.keys())
        print()
        for tp in r_methods:
            print(tp)
            print(Counter(tks[tp]))
            print()
        print('Reduced script:')
        print(script_order)
        print()

    # Return reduced-form script length
    return script_order


def encode_r(filename, noisy):
    r_methods = ['Token.Keyword.Pseudo', 'Token.Operator', 'Token.Keyword.Namespace']

    # Read file
    r_code = None
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        r_code = f.read()

    # Lex code
    lexer = DynamicRLexer()
    tks = lex(r_code, lexer)

    # Bootstrap lexer namespace
    packages = []
    for tktype, pkg in tks:
        if str(tktype) == 'Token.Keyword.Namespace':
            lexer.expand_namespace(pkg)
            if noisy:
                print("Added {} namespace to keywords.".format(pkg))

    # Re-lex
    tokens = lex(r_code, lexer)

    # Classify tokens by type
    tks = defaultdict(list)
    script_order = []
    for tktype, value in tokens:
        val = value.replace("\t", "").replace("\n", "").replace(' ', "").replace(';', "")  # Strip whitespace

        # print("{}: {}".format(str(tktype), val))

        if val in [':', '<-', '=', '$']:
            tktype = "Token.Text"  # Ignore range, class and assignment operators

        # Store token by type and order
        tks[str(tktype)].append(val)
        if str(tktype) in r_methods:
            script_order.append(val)

    # Print token counts and script order
    if noisy:
        print("Token typology")
        print(tks.keys())
        print()
        for tp in r_methods:
            print(tp)
            print(Counter(tks[tp]))
            print()
        print('Reduced script:')
        print(script_order)
        print()

    # Return reduced-form script length
    return script_order


def encode_python(filename, noisy):
    python_methods = ['Token.Name.Namespace', 'Token.Operator', 'Token.Keyword.Pseudo']

    # Read file
    python_code = None
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        python_code = f.read()

    # Lex code
    tokens = lex(python_code, SuperNumPyLexer())

    # Classify tokens by type
    tks = defaultdict(list)
    script_order = []
    for tktype, value in tokens:
        val = value.replace("\t", "").replace("\n", "").replace(' ', "").replace(';', "")  # Strip whitespace

        if val in ['.', '=']:
            tktype = "Token.Text"  # Ignore dot and assignment operators

        # Store token by type and order
        tks[str(tktype)].append(val)
        if str(tktype) in python_methods:
            script_order.append(val)

    # Print token counts and script order
    if noisy:
        print("Token typology")
        print(tks.keys())
        print()
        for tp in r_methods:
            print(tp)
            print(Counter(tks[tp]))
            print()
        print('Reduced script:')
        print(script_order)
        print()

    # Return reduced-form script length
    return script_order


def encode_fortran(filename, noisy):
    fortran_methods = ['Token.Keyword', 'Token.Name.Builtin', 'Token.Keyword.Type', 'Token.Operator']

    # Read file
    fortran_code = None
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        fortran_code = f.read()

    # Lex code
    tokens = lex(fortran_code, SuperFortranFixedLexer())

    # Classify tokens by type
    tks = defaultdict(list)
    script_order = []
    for tktype, value in tokens:
        val = value.replace("\t", "").replace("\n", "").replace(' ', "").replace(';', "")  # Strip whitespace

        if val in ['=']:
            tktype = "Token.Text"  # Ignore dot and assignment operators

        # Store token by type and order
        tks[str(tktype)].append(val)
        if str(tktype) in fortran_methods:
            script_order.append(val)

    # Print token counts and script order
    if noisy:
        print("Token typology")
        print(tks.keys())
        print()
        for tp in r_methods:
            print(tp)
            print(Counter(tks[tp]))
            print()
        print('Reduced script:')
        print(script_order)
        print()

    # Return reduced-form script length
    return script_order


# Dispatch method to language-specific lexing
# These are pretty repetitive but idiosyncratic enough that I want to split them up now
def lexify(filename, noisy=False):
    ext = filename.rsplit(".")[-1]
    if ext in ["do", "ado"]:
        return encode_stata(filename, noisy)
    elif ext == "m":
        return encode_matlab(filename, noisy)
    elif ext == "sas":
        return encode_sas(filename, noisy)
    elif ext == "py":
        return encode_python(filename, noisy)
    elif ext in ["r", "R"]:
        return encode_r(filename, noisy)
    elif ext in ["f", "f90"]:
        return encode_fortran(filename, noisy)
    elif ext == "sps":
        raise NotImplementedError("SPSS files not currently handled.")
    else:
        raise NotImplementedError("Methodologist doesn't know this file type: *.{}".format(ext))


if __name__ == "__main__":
    script_len = len(lexify(sys.argv[1], noisy=True))
    print("Read a script with {} functions.".format(str(script_len)))
