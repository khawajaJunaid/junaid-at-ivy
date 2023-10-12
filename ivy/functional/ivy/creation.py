# global
from __future__ import annotations
import functools
from numbers import Number
from typing import (
    Union,
    Tuple,
    Optional,
    List,
    Sequence,
    Callable,
    Protocol,
    TypeVar,
    Iterable,
    Dict,
)
import numpy as np
from numpy.core._multiarray_umath import _load_from_filelike
import os 
import operator
import contextlib
from numpy.core import overrides
# local
import ivy
from ivy import to_ivy
from ivy.utils.exceptions import handle_exceptions
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_array_function,
    infer_device,
    infer_dtype,
    handle_out_argument,
    outputs_to_ivy_arrays,
    inputs_to_native_arrays,
    inputs_to_native_shapes,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_device_shifting,
    handle_backend_invalid,
)

# Helpers #
# --------#


def _ensure_ndmin_ndarray_check_param(ndmin):
    """Just checks if the param ndmin is supported on
        _ensure_ndmin_ndarray. It is intended to be used as
        verification before running anything expensive.
        e.g. loadtxt, genfromtxt
    """
    # Check correctness of the values of `ndmin`
    if ndmin not in [0, 1, 2]:
        raise ValueError(f"Illegal value of ndmin keyword: {ndmin}")

def _ensure_ndmin_ndarray(a, *, ndmin: int):
    """This is a helper function of loadtxt and genfromtxt to ensure
        proper minimum dimension as requested

        ndim : int. Supported values 1, 2, 3
                    ^^ whenever this changes, keep in sync with
                       _ensure_ndmin_ndarray_check_param
    """
    # Verify that the array has at least dimensions `ndmin`.
    # Tweak the size and shape of the arrays - remove extraneous dimensions
    if a.ndim > ndmin:
        a = ivy.squeeze(a)
    # and ensure we have the minimum number of dimensions asked for
    # - has to be in this order for the odd case ndmin=1, a.squeeze().ndim=0
    if a.ndim < ndmin:
        if ndmin == 1:
            a = ivy.atleast_1d(a)
        elif ndmin == 2:
            a = ivy.atleast_2d(a).T

    return a

def _check_nonneg_int(value, name="argument"):
    try:
        operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")

def _preprocess_comments(iterable, comments, encoding):
    """
    Generator that consumes a line iterated iterable and strips out the
    multiple (or multi-character) comments from lines.
    This is a pre-processing step to achieve feature parity with loadtxt
    (we assume that this feature is a nieche feature).
    """
    for line in iterable:
        if isinstance(line, bytes):
            # Need to handle conversion here, or the splitting would fail
            line = line.decode(encoding)

        for c in comments:
            line = line.split(c, 1)[0]

        yield line

_loadtxt_chunksize = 50000
def _read(fname, *, delimiter=',', comment='#', quote='"',
          imaginary_unit='j', usecols=None, skiplines=0,
          max_rows=None, converters=None, ndmin=None, unpack=False,
          dtype=ivy.float64, encoding="bytes"):
    r"""
    Read a NumPy array from a text file.
    This is a helper function for loadtxt.

    Parameters
    ----------
    fname : file, str, or pathlib.Path
        The filename or the file to be read.
    delimiter : str, optional
        Field delimiter of the fields in line of the file.
        Default is a comma, ','.  If None any sequence of whitespace is
        considered a delimiter.
    comment : str or sequence of str or None, optional
        Character that begins a comment.  All text from the comment
        character to the end of the line is ignored.
        Multiple comments or multiple-character comment strings are supported,
        but may be slower and `quote` must be empty if used.
        Use None to disable all use of comments.
    quote : str or None, optional
        Character that is used to quote string fields. Default is '"'
        (a double quote). Use None to disable quote support.
    imaginary_unit : str, optional
        Character that represent the imaginary unit `sqrt(-1)`.
        Default is 'j'.
    usecols : array_like, optional
        A one-dimensional array of integer column numbers.  These are the
        columns from the file to be included in the array.  If this value
        is not given, all the columns are used.
    skiplines : int, optional
        Number of lines to skip before interpreting the data in the file.
    max_rows : int, optional
        Maximum number of rows of data to read.  Default is to read the
        entire file.
    converters : dict or callable, optional
        A function to parse all columns strings into the desired value, or
        a dictionary mapping column number to a parser function.
        E.g. if column 0 is a date string: ``converters = {0: datestr2num}``.
        Converters can also be used to provide a default value for missing
        data, e.g. ``converters = lambda s: float(s.strip() or 0)`` will
        convert empty fields to 0.
        Default: None
    ndmin : int, optional
        Minimum dimension of the array returned.
        Allowed values are 0, 1 or 2.  Default is 0.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = read(...)``.  When used with a structured
        data-type, arrays are returned for each field.  Default is False.
    dtype : numpy data type
        A NumPy dtype instance, can be a structured dtype to map to the
        columns of the file.
    encoding : str, optional
        Encoding used to decode the inputfile. The special value 'bytes'
        (the default) enables backwards-compatible behavior for `converters`,
        ensuring that inputs to the converter functions are encoded
        bytes objects. The special value 'bytes' has no additional effect if
        ``converters=None``. If encoding is ``'bytes'`` or ``None``, the
        default system encoding is used.

    Returns
    -------
    ndarray
        NumPy array.
    """
    # Handle special 'bytes' keyword for encoding
    byte_converters = False
    if encoding == 'bytes':
        encoding = None
        byte_converters = True

    if dtype is None:
        raise TypeError("a dtype must be provided.")
    dtype = ivy.dtype(dtype)

    read_dtype_via_object_chunks = None
    if dtype.kind in 'SUM' and (
            dtype == "S0" or dtype == "U0" or dtype == "M8" or dtype == 'm8'):
        # This is a legacy "flexible" dtype.  We do not truly support
        # parametric dtypes currently (no dtype discovery step in the core),
        # but have to support these for backward compatibility.
        read_dtype_via_object_chunks = dtype
        dtype = ivy.dtype(object)

    if usecols is not None:
        # Allow usecols to be a single int or a sequence of ints, the C-code
        # handles the rest
        try:
            usecols = list(usecols)
        except TypeError:
            usecols = [usecols]

    _ensure_ndmin_ndarray_check_param(ndmin)

    if comment is None:
        comments = None
    else:
        # assume comments are a sequence of strings
        if "" in comment:
            raise ValueError(
                "comments cannot be an empty string. Use comments=None to "
                "disable comments."
            )
        comments = tuple(comment)
        comment = None
        if len(comments) == 0:
            comments = None  # No comments at all
        elif len(comments) == 1:
            # If there is only one comment, and that comment has one character,
            # the normal parsing can deal with it just fine.
            if isinstance(comments[0], str) and len(comments[0]) == 1:
                comment = comments[0]
                comments = None
        else:
            # Input validation if there are multiple comment characters
            if delimiter in comments:
                raise TypeError(
                    f"Comment characters '{comments}' cannot include the "
                    f"delimiter '{delimiter}'"
                )

    # comment is now either a 1 or 0 character string or a tuple:
    if comments is not None:
        # Note: An earlier version support two character comments (and could
        #       have been extended to multiple characters, we assume this is
        #       rare enough to not optimize for.
        if quote is not None:
            raise ValueError(
                "when multiple comments or a multi-character comment is "
                "given, quotes are not supported.  In this case quotechar "
                "must be set to None.")

    if len(imaginary_unit) != 1:
        raise ValueError('len(imaginary_unit) must be 1.')

    _check_nonneg_int(skiplines)
    if max_rows is not None:
        _check_nonneg_int(max_rows)
    else:
        # Passing -1 to the C code means "read the entire file".
        max_rows = -1

    fh_closing_ctx = contextlib.nullcontext()
    filelike = False
    try:
        if isinstance(fname, os.PathLike):
            fname = os.fspath(fname)
        if isinstance(fname, str):
            fh = np.lib._datasource.open(fname, 'rt', encoding=encoding)
            if encoding is None:
                encoding = getattr(fh, 'encoding', 'latin1')

            fh_closing_ctx = contextlib.closing(fh)
            data = fh
            filelike = True
        else:
            if encoding is None:
                encoding = getattr(fname, 'encoding', 'latin1')
            data = iter(fname)
    except TypeError as e:
        raise ValueError(
            f"fname must be a string, filehandle, list of strings,\n"
            f"or generator. Got {type(fname)} instead.") from e

    with fh_closing_ctx:
        if comments is not None:
            if filelike:
                data = iter(data)
                filelike = False
            data = _preprocess_comments(data, comments, encoding)

        if read_dtype_via_object_chunks is None:
            arr = _load_from_filelike(
                data, delimiter=delimiter, comment=comment, quote=quote,
                imaginary_unit=imaginary_unit,
                usecols=usecols, skiplines=skiplines, max_rows=max_rows,
                converters=converters, dtype=dtype,
                encoding=encoding, filelike=filelike,
                byte_converters=byte_converters)

        else:
            # This branch reads the file into chunks of object arrays and then
            # casts them to the desired actual dtype.  This ensures correct
            # string-length and datetime-unit discovery (like `arr.astype()`).
            # Due to chunking, certain error reports are less clear, currently.
            if filelike:
                data = iter(data)  # cannot chunk when reading from file

            c_byte_converters = False
            if read_dtype_via_object_chunks == "S":
                c_byte_converters = True  # Use latin1 rather than ascii

            chunks = []
            while max_rows != 0:
                if max_rows < 0:
                    chunk_size = _loadtxt_chunksize
                else:
                    chunk_size = min(_loadtxt_chunksize, max_rows)

                next_arr = _load_from_filelike(
                    data, delimiter=delimiter, comment=comment, quote=quote,
                    imaginary_unit=imaginary_unit,
                    usecols=usecols, skiplines=skiplines, max_rows=max_rows,
                    converters=converters, dtype=dtype,
                    encoding=encoding, filelike=filelike,
                    byte_converters=byte_converters,
                    c_byte_converters=c_byte_converters)
                # Cast here already.  We hope that this is better even for
                # large files because the storage is more compact.  It could
                # be adapted (in principle the concatenate could cast).
                chunks.append(next_arr.astype(read_dtype_via_object_chunks))

                skiprows = 0  # Only have to skip for first chunk
                if max_rows >= 0:
                    max_rows -= chunk_size
                if len(next_arr) < chunk_size:
                    # There was less data than requested, so we are done.
                    break

            # Need at least one chunk, but if empty, the last one may have
            # the wrong shape.
            if len(chunks) > 1 and len(chunks[-1]) == 0:
                del chunks[-1]
            if len(chunks) == 1:
                arr = chunks[0]
            else:
                arr = ivy.concat(chunks, axis=0)

    # NOTE: ndmin works as advertised for structured dtypes, but normally
    #       these would return a 1D result plus the structured dimension,
    #       so ndmin=2 adds a third dimension even when no squeezing occurs.
    #       A `squeeze=False` could be a better solution (pandas uses squeeze).
    arr = _ensure_ndmin_ndarray(arr, ndmin=ndmin)

    if arr.shape:
        if arr.shape[0] == 0:
            print( f'loadtxt: input contained no data: "{fname}"')
    if unpack:
        # Unpack structured dtypes if requested:
        dt = arr.dtype
        if dt.names is not None:
            # For structured arrays, return an array for each field.
            return [arr[field] for field in dt.names]
        else:
            return arr.T
    else:
        return arr


def _loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding='bytes', max_rows=None, *, quotechar=None,
            like=None):

    if isinstance(delimiter, bytes):
        delimiter.decode("latin1")

    if dtype is None:
        dtype = ivy.float64

    comment = comments
    # Control character type conversions for Py3 convenience
    if comment is not None:
        if isinstance(comment, (str, bytes)):
            comment = [comment]
        comment = [
            x.decode('latin1') if isinstance(x, bytes) else x for x in comment]
    if isinstance(delimiter, bytes):
        delimiter = delimiter.decode('latin1')

    arr = _read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
                converters=converters, skiplines=skiprows, usecols=usecols,
                unpack=unpack, ndmin=ndmin, encoding=encoding,
                max_rows=max_rows, quote=quotechar)

    return arr

def _asarray_handle_nestable(fn: Callable) -> Callable:
    fn_name = fn.__name__

    @functools.wraps(fn)
    def _asarray_handle_nestable_wrapper(*args, **kwargs):
        """
        Call `fn` with the *nestable* property of the function correctly handled. This
        means mapping the function to the container leaves if any containers are passed
        in the input.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with the nestable property handled correctly.
        """
        # This decorator should only be applied to ivy.asarray, so we know where
        # the container must be if there is one.
        cont_fn = getattr(ivy.Container, "static_" + fn_name)
        if isinstance(args[0], ivy.Container):
            return cont_fn(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an ivy or a native array.
        return fn(*args, **kwargs)

    _asarray_handle_nestable_wrapper.handle_nestable = True
    return _asarray_handle_nestable_wrapper


def _ivy_to_native(x):
    # checks the first element of the leaf list and
    # converts it to a native array if it is an ivy array
    # assumes that either all elements in a leaf list are ivy arrays
    # or none of them are
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x = list(x) if isinstance(x, tuple) else x
            x[i] = _ivy_to_native(item)
    else:
        if (isinstance(x, (list, tuple)) and len(x) > 0) and ivy.is_ivy_array(x[0]):
            x = ivy.to_native(x, nested=True)
        elif ivy.is_ivy_array(x):
            x = ivy.to_native(x)
    return x


def _shape_to_native(x):
    # checks the first element of the leaf list and
    # converts it to a native array if it is an ivy array
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x = list(x) if isinstance(x, tuple) else x
            x[i] = _shape_to_native(item)
    else:
        if (isinstance(x, (list, tuple)) and len(x) > 0) and (
            isinstance(x[0], ivy.Shape) and ivy.array_mode
        ):
            x = ivy.nested_map(lambda x: x.shape if isinstance(x, ivy.Shape) else x, x)
        elif isinstance(x, ivy.Shape) and ivy.array_mode:
            x = x.shape
    return x


def _flatten_nest(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten_nest(x)
        else:
            yield x


def _remove_np_bfloat16(obj):
    # unlike other frameworks, torch and paddle do not support creating tensors
    # from numpy arrays that have bfloat16 dtype using any extension because
    # bfloat16 in not supported natively by numpy (as of version <=1.25)
    if isinstance(obj, np.ndarray) and obj.dtype.name == "bfloat16":
        return obj.tolist()
    return obj


def _asarray_to_native_arrays_and_back(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_to_native_arrays_and_back_wrapper(*args, dtype=None, **kwargs):
        """
        Wrap `fn` so that input arrays are all converted to `ivy.NativeArray` instances
        and return arrays are all converted to `ivy.Array` instances.

        This wrapper is specifically for the backend implementations of
        asarray.

        It assumes either all the elements in a leaf list are ivy arrays
        or none of them are. It checks the first element of all the leaf
        list. If it is an ivy array, it converts all the elements in the
        leaf list to native otherwise it skips that leaf list.
        """
        new_arg = _ivy_to_native(args[0])
        new_args = (new_arg,) + args[1:]
        if dtype is not None:
            dtype = ivy.default_dtype(dtype=dtype, as_native=True)
        return to_ivy(fn(*new_args, dtype=dtype, **kwargs))

    return _asarray_to_native_arrays_and_back_wrapper


def _asarray_infer_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_infer_dtype_wrapper(*args, dtype=None, **kwargs):
        """
        Determine the correct `dtype`, and then calls the function with the `dtype`
        passed explicitly. This wrapper is specifically for the backend implementations
        of asarray.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        dtype
            The dtype for the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with `dtype` passed explicitly.
        """

        def _infer_dtype(obj):
            if isinstance(obj, ivy.NativeShape):
                obj = list(obj)
            if hasattr(obj, "dtype"):
                return obj.dtype.name if isinstance(obj, np.ndarray) else obj.dtype
            else:
                return ivy.default_dtype(item=obj)

        if not ivy.exists(dtype):
            arr = args[0]
            # get default dtypes for all elements
            dtype_list = [ivy.nested_map(lambda x: _infer_dtype(x), arr, shallow=False)]
            # flatten the nested structure
            dtype_list = _flatten_nest(dtype_list)
            # keep unique dtypes
            dtype_list = list(set(dtype_list))
            if len(dtype_list) != 0:  # handle the case of empty input
                # promote all dtypes to a single dtype
                dtype = dtype_list[0]
                # we disable precise mode to avoid wider than necessary casting
                # that might result from the mixing of int32 and float32
                with ivy.PreciseMode(False):
                    for dt in dtype_list[1:]:
                        dtype = ivy.promote_types(dtype, dt)
            else:
                dtype = ivy.default_float_dtype()
            dtype = ivy.as_native_dtype(dtype)
        # call the function with dtype provided explicitly
        return fn(*args, dtype=dtype, **kwargs)

    _asarray_infer_dtype_wrapper.infer_dtype = True
    return _asarray_infer_dtype_wrapper


def _asarray_infer_device(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_infer_device_wrapper(*args, device=None, **kwargs):
        """
        Determine the correct `device`, and then calls the function with the `device`
        passed explicitly. This wrapper is specifically for the backend implementations
        of asarray.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        device
            The device for the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with `device` passed explicitly.
        """
        if isinstance(args[0], list):
            return fn(
                *args, device=ivy.default_device(device, as_native=True), **kwargs
            )

        # find the first array argument, if required
        arr = None if ivy.exists(device) else args[0]
        # infer the correct device
        device = ivy.default_device(device, item=arr, as_native=True)
        # call the function with device provided explicitly
        return fn(*args, device=device, **kwargs)

    _asarray_infer_device_wrapper.infer_device = True
    return _asarray_infer_device_wrapper


def _asarray_inputs_to_native_shapes(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_native_shapes(*args, **kwargs):
        new_arg = _shape_to_native(args[0])
        new_args = (new_arg,) + args[1:]
        return fn(*new_args, **kwargs)

    _inputs_to_native_shapes.inputs_to_native_shapes = True
    return _inputs_to_native_shapes


# Type hints #
# -----------#

SupportsBufferProtocol = TypeVar("SupportsBufferProtocol")
_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> Union[_T_co, NestedSequence[_T_co]]: ...

    def __len__(self, /) -> int: ...


# Array API Standard #
# -------------------#


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@outputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
@infer_device
def arange(
    start: Number,
    /,
    stop: Optional[Number] = None,
    step: Number = 1,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return evenly spaced values within a given interval, with the spacing being
    specified.

    Values are generated within the half-open interval [start, stop) (in other words,
    the interval including start but excluding stop). For integer arguments the function
    is equivalent to the Python built-in range function, but returns an array in the
    chosen ml_framework rather than a list.

    See :math:`linspace` for a certain number of evenly spaced values in an interval.

    Parameters
    ----------
    start
        if stop is specified, the start of interval (inclusive); otherwise, the end of
        the interval (exclusive). If stop is not specified, the default starting value
        is 0.
    stop
        the end of the interval. Default: ``None``.
    step
        the distance between two adjacent elements (out[i+1] - out[i]). Must not be 0;
        may be negative, this results in an empty array if stop >= start. Default: 1.
    dtype
        output array data type. If dtype is None, the output array data type must be
        inferred from start, stop and step. If those are all integers, the output array
        dtype must be the default integer dtype; if one or more have type float, then
        the output array dtype must be the default floating-point data type. Default:
        None.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        a one-dimensional array containing evenly spaced values. The length of the
        output array must be ceil((stop-start)/step) if stop - start and step have the
        same sign, and length 0 otherwise.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.arange.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> stop = 5
    >>> x = ivy.arange(stop)
    >>> print(x)
    ivy.array([0, 1, 2, 3, 4])

    >>> start = 1
    >>> stop = 5
    >>> x = ivy.arange(start, stop)
    >>> print(x)
    ivy.array([1, 2, 3, 4])

    >>> start = 1
    >>> stop = 10
    >>> step = 2
    >>> x = ivy.arange(start, stop, step)
    >>> print(x)
    ivy.array([1, 3, 5, 7, 9])

    >>> start = 1
    >>> stop = 10
    >>> step = 2
    >>> dtype = "float64"
    >>> device = "cpu"
    >>> x = ivy.arange(start, stop, step, dtype=dtype, device=device)
    >>> print(x, x.dtype, x.device)
    ivy.array([1., 3., 5., 7., 9.]) float64 cpu
    """
    return current_backend().arange(
        start, stop, step, dtype=dtype, device=device, out=out
    )


@handle_backend_invalid
@handle_array_like_without_promotion
@handle_out_argument
@handle_array_function
@handle_device_shifting
def asarray(
    obj: Union[
        ivy.Array,
        ivy.NativeArray,
        ivy.Shape,
        ivy.NativeShape,
        bool,
        int,
        float,
        NestedSequence,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Convert the input to an array.

    Parameters
    ----------
    obj
        input data, in any form that can be converted to an array. This includes lists,
        lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    copy
        boolean, indicating whether or not to copy the input. Default: ``None``.
    dtype
       output array data type. If ``dtype`` is ``None``, the output array data type must
       be the default floating-point data type. Default  ``None``.
    device
       device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array interpretation of x.

    Examples
    --------
    With list of lists as input:

    >>> ivy.asarray([[1,2],[3,4]])
    ivy.array([[1, 2],
               [3, 4]])

    With tuple of lists as input:

    >>> ivy.asarray(([1.4,5.6,5.5],[3.1,9.1,7.5]))
    ivy.array([[1.39999998, 5.5999999 , 5.5       ],
               [3.0999999 , 9.10000038, 7.5       ]])

    With ndarray as input:

    >>> x = ivy.np.ndarray(shape=(2,2), order='C')
    >>> ivy.asarray(x)
    ivy.array([[6.90786433e-310, 6.90786433e-310],
               [6.90786433e-310, 6.90786433e-310]])

    With :class:`ivy.Container` as input:

    >>> x = ivy.Container(a = [(1,2),(3,4),(5,6)], b = ((1,2,3),(4,5,6)))
    >>> ivy.asarray(x)
    {
        a: ivy.array([[1, 2],[3, 4], [5, 6]]),
        b: ivy.array([[1, 2, 3],
                   [4, 5, 6]])
    }

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.asarray.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend().asarray(
        obj, copy=copy, dtype=dtype, device=device, out=out
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@outputs_to_ivy_arrays
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def zeros(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape
       output array shape.
    dtype
       output array data type. If ``dtype`` is ``None``, the output array data type must
       be the default floating-point data type. Default  ``None``.
    device
       device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing zeros.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.zeros.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.NativeShape` input:
    >>> shape = (3, 5)
    >>> x = ivy.zeros(shape)
    >>> print(x)
    ivy.array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])

    >>> x = ivy.zeros(5)
    >>> print(x)
    ivy.array([0., 0., 0., 0., 0.])
    """
    return current_backend().zeros(shape, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@outputs_to_ivy_arrays
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def ones(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a new array having a specified ``shape`` and filled with ones.

    .. note::

        An output array having a complex floating-point data type must contain complex
        numbers having a real component equal to one and an imaginary component equal to
        zero (i.e., ``1 + 0j``).

    Parameters
    ----------
    shape
        output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be the default floating-point data type. Default  ``None``.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing ones.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.ones.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Shape` input:

    >>> shape = (2,2)
    >>> x = ivy.ones(shape)
    >>> print(x)
    ivy.array([[1., 1.],
           [1., 1.]])

    With :class:`ivy.Dtype` input:

    >>> shape = (3,2)
    >>> d_type = ivy.int64
    >>> y = ivy.ones(shape, dtype=d_type)
    >>> print(y)
    ivy.array([[1, 1],
           [1, 1],
           [1, 1]])

    With :class:`ivy.Device` input:

    >>> shape = (3,2)
    >>> y = ivy.ones(shape, device="cpu")
    >>> print(y)
    ivy.array([[1., 1.],
           [1., 1.],
           [1., 1.]])

    With :class:`ivy.Array` input:

    >>> shape = (1, 5, 2)
    >>> x = ivy.zeros(shape)
    >>> ivy.ones(shape, out=x)
    >>> print(x)
    ivy.array([[[1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.]]])
    """
    return current_backend().ones(shape, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def full_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    fill_value: Number,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a new array filled with ``fill_value`` and having the same ``shape`` as an
    input array ``x`` .

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    fill_value
        Scalar fill value
    dtype
        output array data type. If ``dtype`` is `None`, the output array data type must
        be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and where every element is equal to
        ``fill_value``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.full_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :code:`int` datatype:

    >>> x = ivy.array([1, 2, 3, 4, 5, 6])
    >>> fill_value = 1
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([1, 1, 1, 1, 1, 1])

    >>> fill_value = 0.000123
    >>> x = ivy.ones(5)
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

    With float datatype:

    >>> x = ivy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> fill_value = 0.000123
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([3.0, 8.0])
    >>> fill_value = 0.000123
    >>> y = ivy.full_like(x,fill_value)
    >>> print(y)
    ivy.array([0.000123, 0.000123])

    >>> x = ivy.native_array([[3., 8., 2.], [2., 8., 3.]])
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([[0.000123, 0.000123, 0.000123],
               [0.000123, 0.000123, 0.000123]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.2, 2.2324, 3.234]),
    ...                   b=ivy.array([4.123, 5.23, 6.23]))
    >>> fill_value = 15.0
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    {
        a: ivy.array([15., 15., 15.]),
        b: ivy.array([15., 15., 15.])
    }
    """
    return current_backend(x).full_like(
        x, fill_value, dtype=dtype, device=device, out=out
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def ones_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a new array filled with ones and having the same shape as an input array
    ``x``.

    .. note::

        An output array having a complex floating-point data type must contain complex
        numbers having a real component equal to one and an imaginary component equal
        to zero (i.e., ``1 + 0j``).

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default  ``None``.
    device
        device on which to place the created array. If device is ``None``, the output
        array device must be inferred from ``x``. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and filled with ``ones``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.ones_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3, 4, 5, 6])
    >>> y = ivy.ones_like(x)
    >>> print(y)
    ivy.array([1, 1, 1, 1, 1, 1])

    >>> x = ivy.array([[0, 1, 2],[3, 4, 5]], dtype = ivy.float32)
    >>> y = ivy.ones_like(x)
    >>> print(y)
    ivy.array([[1., 1., 1.],
           [1., 1., 1.]])

    >>> x = ivy.array([3., 2., 1.])
    >>> y = ivy.zeros(3)
    >>> ivy.ones_like(x, out=y)
    >>> print(y)
    ivy.array([1., 1., 1.])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[3, 8, 2],[2, 8, 3]])
    >>> y = ivy.ones_like(x)
    >>> print(y)
    ivy.array([[1, 1, 1],
           [1, 1, 1]])

    >>> x = ivy.native_array([3, 8, 2, 0, 0, 2])
    >>> y = ivy.ones_like(x, dtype=ivy.IntDtype('int32'), device=ivy.Device('cpu'))
    >>> print(y)
    ivy.array([1, 1, 1, 1, 1, 1])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([3, 2, 1]), b=ivy.array([8, 2, 3]))
    >>> y = ivy.ones_like(x)
    >>> print(y)
    {
        a: ivy.array([1, 1, 1]),
        b: ivy.array([1, 1, 1])
    }

    With :class:`ivy.Array` input:

    >>> x = ivy.array([2, 3, 8, 2, 1])
    >>> y = x.ones_like()
    >>> print(y)
    ivy.array([1, 1, 1, 1, 1])

    With :class:'ivy.Container' input:

    >>> x = ivy.Container(a=ivy.array([3., 8.]), b=ivy.array([2., 2.]))
    >>> y = x.ones_like()
    >>> print(y)
    {
        a: ivy.array([1., 1.]),
        b: ivy.array([1., 1.])
    }
    """
    return current_backend(x).ones_like(x, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def zeros_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a new array filled with zeros and having the same ``shape`` as an input array
    ``x``.

    Parameters
    ----------
    x
         input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and filled with ``zeros``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.zeros_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3, 4, 5, 6])
    >>> y = ivy.zeros_like(x)
    >>> print(y)
    ivy.array([0, 0, 0, 0, 0, 0])

    >>> x = ivy.array([[0, 1, 2],[3, 4, 5]], dtype = ivy.float32)
    >>> y = ivy.zeros_like(x)
    >>> print(y)
    ivy.array([[0., 0., 0.],
            [0., 0., 0.]])

    >>> x = ivy.array([3., 2., 1.])
    >>> y = ivy.ones(3)
    >>> ivy.zeros_like(x, out=y)
    >>> print(y)
    ivy.array([0., 0., 0.])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[3, 8, 2],[2, 8, 3]])
    >>> y = ivy.zeros_like(x)
    >>> print(y)
    ivy.array([[0, 0, 0],[0, 0, 0]])


    >>> x = ivy.native_array([3, 8, 2, 0, 0, 2])
    >>> y = ivy.zeros_like(x, dtype=ivy.IntDtype('int32'), device=ivy.Device('cpu'))
    >>> print(y)
    ivy.array([0, 0, 0, 0, 0, 0])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([3, 2, 1]), b=ivy.array([8, 2, 3]))
    >>> y = ivy.zeros_like(x)
    >>> print(y)
    {
        a: ivy.array([0, 0, 0]),
        b: ivy.array([0, 0, 0])
    }


    With :class:`ivy.Array` input:

    >>> x = ivy.array([2, 3, 8, 2, 1])
    >>> y = x.zeros_like()
    >>> print(y)
    ivy.array([0, 0, 0, 0, 0])

    With :class:'ivy.Container' input:

    >>> x = ivy.Container(a=ivy.array([3., 8.]), b=ivy.array([2., 2.]))
    >>> y = x.zeros_like()
    >>> print(y)
    {
        a: ivy.array([0., 0.]),
        b: ivy.array([0., 0.])
    }
    """
    return current_backend(x).zeros_like(x, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def tril(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the lower triangular part of a matrix (or a stack of matrices) ``x``.

    .. note::

        The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i``
        on the interval ``[0, min(M, N) - 1]``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.
    k
        diagonal above which to zero elements. If k = 0, the diagonal is the main
        diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the lower triangular part(s). The returned array must have
        the same shape and data type as x. All elements above the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.tril.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).tril(x, k=k, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def triu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the upper triangular part of a matrix (or a stack of matrices) ``x``.

    .. note::

        The upper triangular part of the matrix is defined as the elements
        on and above the specified diagonal ``k``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.    *,
    k
        diagonal below which to zero elements. If k = 0, the diagonal is the main
        diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the upper triangular part(s). The returned array must have
        the same shape and data type as x. All elements below the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.triu.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).triu(x, k=k, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@outputs_to_ivy_arrays
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def empty(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape
       output array shape.
    dtype
        output array data type. If dtype is None, the output array data type must be the
        default floating-point data type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an uninitialized array having a specified shape


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.empty.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend().empty(shape, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def empty_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return an uninitialized array with the same shape as an input array x.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If dtype is None, the output array data type must be
        inferred from x. Deafult: ``None``.
    device
        device on which to place the created array. If device is None, the output array
        device must be inferred from x. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as x and containing uninitialized data.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.empty_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).empty_like(x, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@outputs_to_ivy_arrays
@handle_array_function
@infer_dtype
@infer_device
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a two-dimensional array with ones on the k diagonal and zeros elsewhere.

    Parameters
    ----------
    n_rows
        number of rows in the output array.
    n_cols
        number of columns in the output array. If None, the default number of columns in
        the output array is equal to n_rows. Default: ``None``.
    k
        index of the diagonal. A positive value refers to an upper diagonal, a negative
        value to a lower diagonal, and 0 to the main diagonal. Default: ``0``.
    batch_shape
        optional input that determines returning identity array shape.
        Default: ``None``.
    dtype
        output array data type. If dtype is None, the output array data type must be the
        default floating-point data type. Default: ``None``.
    device
        the device on which to place the created array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        device on which to place the created array. Default: ``None``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.eye.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances as a replacement to any of the arguments.

    Examples
    --------
    With :'n_rows' input:

    >>> x = ivy.eye(3)
    >>> print(x)
    ivy.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])


    With :'n_cols' input:

    >>> x = ivy.eye(3,4)
    >>> print(x)
    ivy.array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.]])


    With :'k' input:

    >>> x = ivy.eye(3, k=1)
    >>> print(x)
    ivy.array([[0., 1., 0.],
               [0., 0., 1.],
               [0., 0., 0.]])


    With :'dtype' input:

    >>> x = ivy.eye(4, k=2, dtype=ivy.IntDtype('int32'))
    >>> print(x)
    ivy.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])


    With :'batch_shape' input:

    >>> x = ivy.eye(2, 3, batch_shape=[3])
    >>> print(x)
    ivy.array([[[1., 0., 0.],
                [0., 1., 0.]],

                [[1., 0., 0.],
                [0., 1., 0.]],

                [[1., 0., 0.],
                [0., 1., 0.]]])


    With :'out' input:

    >>> y = ivy.ones((3, 3))
    >>> ivy.eye(3, out=y)
    >>> print(y)
    ivy.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])


    With :'device' input:

    >>> x = ivy.eye(3, device=ivy.Device('cpu'))
    >>> print(x)
    ivy.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
    """
    return current_backend().eye(
        n_rows,
        n_cols,
        k=k,
        batch_shape=batch_shape,
        dtype=dtype,
        device=device,
        out=out,
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def linspace(
    start: Union[ivy.Array, ivy.NativeArray, float],
    stop: Union[ivy.Array, ivy.NativeArray, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Generate a certain number of evenly-spaced values in an interval along a given axis.

    See :math:`arange` that allows to specify the step size of evenly spaced values in
    an interval.

    Parameters
    ----------
    start
        First entry in the range.
    stop
        Final entry in the range.
    num
        Number of values to generate.
    axis
        Axis along which the operation is performed.
    endpoint
        If True, stop is the last sample. Otherwise, it is not included.
    dtype
        output array data type.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of evenly-spaced values.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.linspace.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With float input:

    >>> x = ivy.linspace(1, 2, 3)
    >>> print(x)
    ivy.array([1. , 1.5, 2. ])

    >>> x = ivy.linspace(1, 2, 4, endpoint=False)
    >>> print(x)
    ivy.array([1., 1.25, 1.5 , 1.75])

    >>> x = ivy.linspace(1, 10, 4, dtype="int32")
    >>> print(x)
    ivy.array([ 1,  4,  7, 10])

    >>> x = ivy.linspace(1, 2, 4, device= "cpu")
    >>> print(x)
    ivy.array([1., 1.33333337, 1.66666663, 2.])

    >>> y = ivy.array([0,0,0,0])
    >>> ivy.linspace(1, 2, 4, out= y)
    >>> print(y)
    ivy.array([1, 1, 1, 2])

    With :class:`ivy.Array` input:

    >>> x = ivy.array([1,2])
    >>> y = ivy.array([4,5])
    >>> z = ivy.linspace(x, y, 4, axis = 0)
    >>> print(z)
    ivy.array([[1, 2],
               [2, 3],
               [3, 4],
               [4, 5]])
    """
    return current_backend(start).linspace(
        start,
        stop,
        num,
        axis=axis,
        endpoint=endpoint,
        dtype=dtype,
        device=device,
        out=out,
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def meshgrid(
    *arrays: Union[ivy.Array, ivy.NativeArray],
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[ivy.Array] = None,
) -> List[ivy.Array]:
    """
    Return coordinate matrices from coordinate vectors.

    Parameters
    ----------
    arrays
        an arbitrary number of one-dimensional arrays representing grid coordinates.
        Each array should have the same numeric data type.
    sparse
        if True, a sparse grid is returned in order to conserve memory.
        Default: ``False``.
    indexing
        Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or
        one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases,
        respectively), the ``indexing`` keyword has no effect and should be ignored.
        Default: ``'xy'``.

    Returns
    -------
    ret
        list of N arrays, where ``N`` is the number of provided one-dimensional input
        arrays. Each returned array must have rank ``N``. For ``N`` one-dimensional
        arrays having lengths ``Ni = len(xi)``,

        - if matrix indexing ``ij``, then each returned array must have the shape
          ``(N1, N2, N3, ..., Nn)``.
        - if Cartesian indexing ``xy``, then each returned array must have shape
          ``(N2, N1, N3, ..., Nn)``.

        Accordingly, for the two-dimensional case with input one-dimensional arrays of
        length ``M`` and ``N``, if matrix indexing ``ij``, then each returned array must
        have shape ``(M, N)``, and, if Cartesian indexing ``xy``, then each returned
        array must have shape ``(N, M)``.

        Similarly, for the three-dimensional case with input one-dimensional arrays of
        length ``M``, ``N``, and ``P``, if matrix indexing ``ij``, then each returned
        array must have shape ``(M, N, P)``, and, if Cartesian indexing ``xy``, then
        each returned array must have shape ``(N, M, P)``.

        Each returned array should have the same data type as the input arrays.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of
    the `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.meshgrid.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2])
    >>> y = ivy.array([3, 4])
    >>> xv, yv = ivy.meshgrid(x, y)
    >>> print(xv)
    ivy.array([[1, 2],
            [1, 2]])

    >>> print(yv)
    ivy.array([[3, 3],
            [4, 4]])

    >>> x = ivy.array([1, 2, 5])
    >>> y = ivy.array([4, 1])
    >>> xv, yv = ivy.meshgrid(x, y, indexing='ij')
    >>> print(xv)
    ivy.array([[1, 1],
            [2, 2],
            [5, 5]])

    >>> print(yv)
    ivy.array([[4, 1],
            [4, 1],
            [4, 1]])

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> xv, yv = ivy.meshgrid(x, y, sparse=True)
    >>> print(xv)
    ivy.array([[1, 2, 3]])

    >>> print(yv)
    ivy.array([[4], [5], [6]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([1, 2])
    >>> y = ivy.native_array([3, 4])
    >>> xv, yv = ivy.meshgrid(x, y)
    >>> print(xv)
    ivy.array([[1, 2],
            [1, 2]])

    >>> print(yv)
    ivy.array([[3, 3],
            [4, 4]])
    """
    return current_backend().meshgrid(
        *arrays, sparse=sparse, indexing=indexing, out=out
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@inputs_to_native_arrays
@outputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
@infer_device
def full(
    shape: Union[ivy.Shape, ivy.NativeShape],
    fill_value: Union[float, bool],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a new array having a specified ``shape`` and filled with ``fill_value``.

    Parameters
    ----------
    shape
        output array shape.
    fill_value
        fill value.
    dtype
        output array data type. If ``dtype`` is `None`, the output array data type must
        be inferred from ``fill_value``. If the fill value is an ``int``, the output
        array data type must be the default integer data type. If the fill value is a
        ``float``, the output array data type must be the default floating-point data
        type. If the fill value is a ``bool``, the output array must have boolean data
        type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array where every element is equal to `fill_value`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.full.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Shape` input:

    >>> shape = ivy.Shape((2,2))
    >>> fill_value = 8.6
    >>> x = ivy.full(shape, fill_value)
    >>> print(x)
    ivy.array([[8.6, 8.6],
               [8.6, 8.6]])

    With :class:`ivy.NativeShape` input:

    >>> shape = ivy.NativeShape((2, 2, 2))
    >>> fill_value = True
    >>> dtype = ivy.bool
    >>> device = ivy.Device('cpu')
    >>> x = ivy.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    ivy.array([[[True,  True],
                [True,  True]],
               [[True,  True],
                [True,  True]]])

    With :class:`ivy.NativeDevice` input:

    >>> shape = ivy.NativeShape((1, 2))
    >>> fill_value = 0.68
    >>> dtype = ivy.float64
    >>> device = ivy.NativeDevice('cpu')
    >>> x = ivy.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    ivy.array([[0.68, 0.68]])

    With :class:`ivy.Container` input:

    >>> shape = ivy.Container(a=ivy.NativeShape((2, 1)), b=ivy.Shape((2, 1, 2)))
    >>> fill_value = ivy.Container(a=0.99, b=False)
    >>> dtype = ivy.Container(a=ivy.float64, b=ivy.bool)
    >>> device = ivy.Container(a=ivy.NativeDevice('cpu'), b=ivy.Device('cpu'))
    >>> x = ivy.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    {
        a: ivy.array([[0.99],
                      [0.99]]),
        b: ivy.array([[[False, False]],
                      [[False, False]]])
    }
    """
    return current_backend().full(
        shape, fill_value, dtype=dtype, device=device, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def to_dlpack(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
):
    """
    Return PyCapsule Object.

    Parameters
    ----------
    x  object
        input (array) object.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Return PyCapsule Object.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See
           :ref:`data-interchange` for details.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.from_dlpack.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).to_dlpack(x, out=out)


@handle_backend_invalid
def from_dlpack(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Return a new array containing the data from another (array) object with a
    ``__dlpack__`` method.

    Parameters
    ----------
    x  object
        input (array) object.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the data in `x`.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See
           :ref:`data-interchange` for details.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.from_dlpack.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).from_dlpack(x, out=out)


# Extra #
# ------#


array = asarray


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_arrays
@handle_array_function
@handle_device_shifting
def copy_array(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    to_ivy_array: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Copy an array.

    Parameters
    ----------
    x
        array, input array containing elements to copy.
    to_ivy_array
        boolean, if True the returned array will be an ivy.Array object otherwise
        returns an ivy.NativeArray object (i.e. a torch.tensor, np.array, etc.,
        depending on the backend), defaults to True.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        a copy of the input array ``x``.

    Examples
    --------
    With one :class:`ivy.Array` input:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.copy_array(x)
    >>> print(y)
    ivy.array([-1, 0, 1])

    >>> x = ivy.array([1, 0, 1, 1])
    >>> y = ivy.copy_array(x)
    >>> print(y)
    ivy.array([1, 0, 1, 1])

    >>> x = ivy.array([1, 0, 1, -1])
    >>> y = ivy.zeros((1, 4))
    >>> ivy.copy_array(x, out=y)
    >>> print(y)
    ivy.array([1, 0, 1, -1])

    >>> x = ivy.array([1, 0, 1, 1])
    >>> ivy.copy_array(x, out=x)
    >>> print(x)
    ivy.array([1, 0, 1, 1])

    With one :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]))
    >>> y = ivy.copy_array(x)
    >>> print(y)
    {
        a: ivy.array([-1, 0, 1])
    }

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),b=ivy.array([-1, 0, 1, 1, 1, 0]))
    >>> y = ivy.copy_array(x)
    >>> print(y)
    {
        a: ivy.array([-1, 0, 1]),
        b: ivy.array([-1, 0, 1, 1, 1, 0])
    }

    With one :class:`ivy.Container` static method:

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),b=ivy.array([-1, 0, 1, 1, 1, 0]))
    >>> y = ivy.Container.static_copy_array(x)
    >>> print(y)
    {
        a: ivy.array([-1, 0, 1]),
        b: ivy.array([-1, 0, 1, 1, 1, 0])
    }

    With one :class:`ivy.Array` instance method:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = x.copy_array()
    >>> print(y)
    ivy.array([-1, 0, 1])

    >>> x = ivy.array([1, 0, 1, 1])
    >>> y = x.copy_array()
    >>> print(y)
    ivy.array([1, 0, 1, 1])

    With :class:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([1, 0, 1]),b=ivy.array([-1, 0, 1, 1]))
    >>> y = x.copy_array()
    >>> print(y)
    {
        a: ivy.array([1, 0, 1]),
        b: ivy.array([-1, 0, 1, 1])
    }
    """
    return current_backend(x).copy_array(x, to_ivy_array=to_ivy_array, out=out)


@handle_backend_invalid
@handle_array_like_without_promotion
def native_array(
    x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number], np.ndarray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.NativeArray:
    """
    Convert the input to a native array.

    Parameters
    ----------
    x
        input data, in any form that can be converted to an array. This includes lists,
        lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    dtype
        datatype, optional. Datatype is inferred from the input data.
    device
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        A native array interpretation of x.

    Examples
    --------
    With :class:`List[Number]` input:

    >>> x = [1, 2, 3]
    >>> x_native = ivy.native_array(x)
    >>> print(x_native)
    [1 2 3]

    With :class:`np.ndarray` input:
    >>> y = np.array([4, 5, 6])
    >>> y_native = ivy.native_array(y)
    >>> print(y_native)
    [4 5 6]

    With :class:`ivy.Array` input:
    >>> z = ivy.array([7, 8, 9])
    >>> z_native = ivy.native_array(z)
    >>> print(z_native)
    [7 8 9]
    """
    # ToDo: Make this more efficient,
    # ideally without first converting to ivy.Array with ivy.asarray and then
    # converting back to native with ivy.to_native

    return ivy.to_native(ivy.asarray(x, dtype=dtype, device=device))


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
@infer_device
def one_hot(
    indices: Union[ivy.Array, ivy.NativeArray],
    depth: int,
    /,
    *,
    on_value: Optional[Number] = None,
    off_value: Optional[Number] = None,
    axis: Optional[int] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a one-hot array. The locations represented by indices in the parameter
    indices take value on_value, while all other locations take value off_value.

    Parameters
    ----------
    indices
        Indices for where the ones should be scattered *[batch_shape, dim]*
    depth
        Scalar defining the depth of the one-hot dimension.
    on_value
        Scalar defining the value to fill in output when indices[j] == i.
        Default: ``1``.
    off_value
        Scalar defining the value to fill in output when indices[j] != i.
        Default: ``0``.
    axis
        Axis to scatter on. The default is ``-1``, a new inner-most axis is created.
    dtype
        The data type of the output tensor.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if
        None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of zeros with the same shape and type as a, unless dtype provided which
        overrides.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([3, 1])
    >>> y = 5
    >>> z = x.one_hot(5)
    >>> print(z)
    ivy.array([[0., 0., 0., 1., 0.],
    ...    [0., 1., 0., 0., 0.]])

    >>> x = ivy.array([0])
    >>> y = 5
    >>> ivy.one_hot(x, y)
    ivy.array([[1., 0., 0., 0., 0.]])

    >>> x = ivy.array([0])
    >>> y = 5
    >>> ivy.one_hot(x, 5, out=z)
    ivy.array([[1., 0., 0., 0., 0.]])
    >>> print(z)
    ivy.array([[1., 0., 0., 0., 0.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 2]), \
        b=ivy.array([3, 1]), c=ivy.array([2, 3]))
    >>> y = 5
    >>> z = x.one_hot(y)
    >>> print(z)
    {
        a: ivy.array([[0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.]]),
        b: ivy.array([[0., 0., 0., 1., 0.],
                    [0., 1., 0., 0., 0.]]),
        c: ivy.array([[0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.]])
    }

    >>> x = ivy.Container(a=ivy.array([2]), \
        b=ivy.array([]), c=ivy.native_array([4]))
    >>> y = 7
    >>> z = x.one_hot(y)
    >>> print(z)
    {
        a: ivy.array([[0., 0., 1., 0., 0., 0., 0.]]),
        b: ivy.array([], shape=(0, 7)),
        c: ivy.array([[0., 0., 0., 0., 1., 0., 0.]])
    }
    """
    return current_backend(indices).one_hot(
        indices,
        depth,
        on_value=on_value,
        off_value=off_value,
        axis=axis,
        dtype=dtype,
        device=device,
        out=out,
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@infer_device
def logspace(
    start: Union[ivy.Array, ivy.NativeArray, float],
    stop: Union[ivy.Array, ivy.NativeArray, float],
    /,
    num: int,
    *,
    base: float = 10.0,
    axis: int = 0,
    endpoint: bool = True,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Generate a certain number of evenly-spaced values in log space, in an interval along
    a given axis.

    Parameters
    ----------
    start
        First value in the range in log space. base ** start is the starting value in
        the sequence. Can be an array or a float.
    stop
        Last value in the range in log space. base ** stop is the final value in the
        sequence. Can be an array or a float.
    num
        Number of values to generate.
    base
        The base of the log space. Default is 10.0
    axis
        Axis along which the operation is performed. Relevant only if start or stop are
        array-like. Default is 0.
    endpoint
        If True, stop is the last sample. Otherwise, it is not included. Default is
        True.
    dtype
        The data type of the output tensor. If None, the dtype of on_value is used or if
        that is None, the dtype of off_value is used, or if that is None, defaults to
        float32. Default is None.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Default is
        None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to. Default is None.

    Returns
    -------
    ret
        Tensor of evenly-spaced values in log space.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With float input:

    >>> print(ivy.logspace(1, 2, 4))
    ivy.array([ 10., 21.5443469, 46.41588834, 100.])

    >>> print(ivy.logspace(1, 2, 4, endpoint=False))
    ivy.array([10., 17.7827941, 31.6227766, 56.23413252])

    >>> print(ivy.logspace(1, 2, 4, dtype= int))
    ivy.array([ 10.,  10.,  10., 100.])

    >>> out = ivy.array([0,0,0,0])
    >>> ivy.logspace(1, 2, 4, out = out)
    >>> print(out)
    ivy.array([ 10,  21,  46, 100])

    With :class:`ivy.Array` input:
    >>> x = ivy.array([1, 2])
    >>> y = ivy.array([4, 5])
    >>> print(ivy.logspace(x, y, 4))
    ivy.array([[1.e+01, 1.e+02],
               [1.e+02, 1.e+03],
               [1.e+03, 1.e+04],
               [1.e+04, 1.e+05])

    >>> x = ivy.array([1, 2])
    >>> y = ivy.array([4, 5])
    >>> print(ivy.logspace(x, y, 4, axis = 1))
    ivy.array([[[1.e+01, 1.e+02, 1.e+03, 1.e+04],
               [1.e+02, 1.e+03, 1.e+04, 1.e+05]]])

    >>> x = ivy.array([1, 2])
    >>> y = ivy.array([4])
    >>> print(ivy.logspace(x, y, 4))
    ivy.array([[   10.,   100.],
           [  100.,   100.],
           [ 1000.,  1000.],
           [10000., 10000.]])
    """
    result = base ** linspace(
        start,
        stop,
        num,
        endpoint=endpoint,
        axis=axis,
        dtype=dtype,
        device=device,
    )
    if ivy.exists(out):
        return ivy.inplace_update(out, result)
    return result


@handle_nestable
@outputs_to_ivy_arrays
def frombuffer(
    buffer: bytes,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> ivy.Array:
    r"""
    Interpret a buffer as a 1-dimensional array.

    .. note::
        Note that either of the following must be true:
        1. count is a positive non-zero number, and the total number of bytes
        in the buffer is equal or greater than offset plus count times the size
        (in bytes) of dtype.
        2. count is negative, and the length (number of bytes) of the buffer
        subtracted by the offset is a multiple of the size (in bytes) of dtype.

    Parameters
    ----------
    buffer
        An object that exposes the buffer interface.
    dtype
        Data-type of the returned array; default: float.
    count
        Number of items to read. -1 means all data in the buffer.
    offset
        Start reading the buffer from this offset (in bytes); default: 0.

    Returns
    -------
    out
        1-dimensional array.

    Examples
    --------
    With :class:`bytes` inputs:

    >>> x = b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@'
    >>> y = ivy.frombuffer(x, dtype=ivy.float64)
    >>> print(y)
    ivy.array([1., 2.])

    >>> x = b'\x01\x02\x03\x04'
    >>> y = ivy.frombuffer(x, dtype='int8', count=-2, offset=1)
    >>> print(y)
    ivy.array([2, 3, 4])

    >>> x = b'\x00<\x00@\x00B\x00D\x00E'
    >>> y = ivy.frombuffer(x, dtype='float16', count=4, offset=2)
    >>> print(y)
    ivy.array([2., 3., 4., 5.])
    """
    return current_backend().frombuffer(
        buffer,
        dtype=dtype,
        count=count,
        offset=offset,
    )


@handle_exceptions
@handle_nestable
@outputs_to_ivy_arrays
@infer_device
def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> Tuple[ivy.Array]:
    """
    Return the indices of the upper triangular part of a row by col matrix in a 2-by-N
    shape (tuple of two N dimensional arrays), where the first row contains row
    coordinates of all indices and the second row contains column coordinates. Indices
    are ordered based on rows and then columns.  The upper triangular part of the matrix
    is defined as the elements on and above the diagonal.  The argument k controls which
    diagonal to consider. If k = 0, all elements on and above the main diagonal are
    retained. A positive value excludes just as many diagonals above the main diagonal,
    and similarly a negative value includes just as many diagonals below the main
    diagonal. The main diagonal are the set of indices {(i,i)} for i∈[0,min{n_rows,
    n_cols}−1].

    Notes
    -----
    Primary purpose of this function is to slice an array of shape (n,m). See
    https://numpy.org/doc/stable/reference/generated/numpy.triu_indices.html
    for examples

    Tensorflow does not support slicing 2-D tensor with tuple of tensor of indices

    Parameters
    ----------
    n_rows
       number of rows in the 2-d matrix.
    n_cols
       number of columns in the 2-d matrix. If None n_cols will be the same as n_rows
    k
       number of shifts from the main diagonal. k = 0 includes main diagonal,
       k > 0 moves upwards and k < 0 moves downwards
    device
       device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        an 2xN shape, tuple of two N dimensional, where first subarray (i.e. ret[0])
        contains row coordinates of all indices and the second subarray (i.e ret[1])
        contains columns indices.

    Function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> x = ivy.triu_indices(4,4,0)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
    ivy.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    >>> x = ivy.triu_indices(4,4,1)
    >>> print(x)
    (ivy.array([0, 0, 0, 1, 1, 2]),
    ivy.array([1, 2, 3, 2, 3, 3]))

    >>> x = ivy.triu_indices(4,4,-2)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]),
    ivy.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3]))

    >>> x = ivy.triu_indices(4,2,0)
    >>> print(x)
    (ivy.array([0, 0, 1]),
    ivy.array([0, 1, 1]))

    >>> x = ivy.triu_indices(2,4,0)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1]),
    ivy.array([0, 1, 2, 3, 1, 2, 3]))

    >>> x = ivy.triu_indices(4,-4,0)
    >>> print(x)
    (ivy.array([]), ivy.array([]))

    >>> x = ivy.triu_indices(4,4,100)
    >>> print(x)
    (ivy.array([]), ivy.array([]))

    >>> x = ivy.triu_indices(2,4,-100)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 1]), ivy.array([0, 1, 2, 3, 0, 1, 2, 3]))
    """
    return current_backend().triu_indices(n_rows, n_cols, k, device=device)


@handle_nestable
@outputs_to_ivy_arrays
def loadtxt(
    fname: str,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    comments: str = "#",
    delimiter: Optional[str] = None,
    converters: Optional[Dict[int, Callable]] = None,
    skiprows: int = 0,
    usecols: Optional[Union[int, Sequence[int]]] = None,
    unpack: bool = False,
    ndmin: int = 0,
    encoding: Optional[str] = "bytes",
    max_rows: Optional[int] = None,
    quotechar = None,

) -> ivy.Array:
    r"""
    Load data from a text file.

    Parameters
    ----------
    fname : file, str, pathlib.Path, list of str, generator
        File, filename, list, or generator to read. If the filename
        extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note
        that generators must return bytes or strings. The strings
        in a list or produced by a generator are treated as lines.
    dtype : data-type, optional
        Data-type of the resulting array; default: float. If this is a
        structured data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array. In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str or sequence of str or None, optional
        The characters or list of characters used to indicate the start of a
        comment. None implies no comments. For backward compatibility, byte
        strings will be decoded as 'latin1'. The default is '#'.
    delimiter : str, optional
        The character used to separate the values. For backward compatibility,
        byte strings will be decoded as 'latin1'. The default is whitespace.

        Only single character delimiters are supported. Newline characters
        cannot be used as the delimiter.

    converters : dict or callable, optional
        Converter functions to customize value parsing. If `converters` is
        callable, the function is applied to all columns, else it must be a
        dict that maps column number to a parser function.
        See examples for further details.
        Default: None.

        The ability to pass a single callable to be applied to all columns
        was added.

    skiprows : int, optional
        Skip the first `skiprows` lines, including comments; default: 0.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The default, None, results in all columns being read.

        When a single column has to be read it is possible to use
        an integer instead of a tuple. E.g ``usecols = 3`` reads the
        fourth column the same way as ``usecols = (3,)`` would.

    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``.  When used with a
        structured data-type, arrays are returned for each field.
        Default is False.
    ndmin : int, optional
        The returned array will have at least `ndmin` dimensions.
        Otherwise mono-dimensional axes will be squeezed.
        Legal values: 0 (default), 1 or 2.

    encoding : str, optional
        Encoding used to decode the input file. Does not apply to input streams.
        The special value 'bytes' enables backward compatibility workarounds
        that ensure you receive byte arrays as results if possible and pass
        'latin1' encoded strings to converters. Override this value to receive
        unicode arrays and pass strings as input to converters.  If set to None
        the system default is used. The default value is 'bytes'.


    max_rows : int, optional
        Read `max_rows` rows of content after `skiprows` lines. The default is
        to read all the rows. Note that empty rows containing no data such as
        empty lines and comment lines are not counted towards `max_rows`,
        while such lines are counted in `skiprows`.

        Lines containing no data, including comment lines (e.g., lines
        starting with '#' or as specified via `comments`) are not counted
        towards `max_rows`.

    quotechar : unicode character or None, optional
        The character used to denote the start and end of a quoted item.
        Occurrences of the delimiter or comment characters are ignored within
        a quoted item. The default value is ``quotechar=None``, which means
        quoting support is disabled.

        If two consecutive instances of `quotechar` are found within a quoted
        field, the first is treated as an escape character. See examples.

    Returns
    -------
    out : ndarray
        Data read from the text file.


    Notes
    -----
    This function aims to be a fast reader for simply formatted files.  The
    `genfromtxt` function provides more sophisticated handling of, e.g.,
    lines with missing values.

    Each row in the input text file must have the same number of values to be
    able to read all values. If all rows do not have the same number of values, a
    subset of up to n columns (where n is the least number of values present
    in all rows) can be read by specifying the columns via `usecols`.


    The strings produced by the Python float.hex method can be used as
    input for floats.

    Examples
    --------
    With :class:`str` inputs:

    >>> x = '1 2\n3 4'
    >>> y = ivy.loadtxt(x, dtype=ivy.float32)
    >>> print(y)
    ivy.array([[1., 2.],
            [3., 4.]])

    >>> x = '1,2,3\n4,5,6'
    >>> y = ivy.loadtxt(x, dtype='int', delimiter=',')
    >>> print(y)
    ivy.array([[1, 2, 3],
            [4, 5, 6]])

    With :class:`StringIO` inputs:

    >>> from io import StringIO
    >>> c = StringIO("0 1\n2 3")
    >>> ivy.loadtxt(c)
    ivy.array([[0., 1.],
            [2., 3.]])

    >>> d = StringIO("M 21 72\nF 35 58")
    >>> ivy.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    ivy.array([(b'M', 21, 72.), (b'F', 35, 58.)],
            dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])

    >>> c = StringIO("1,0,2\n3,0,4")
    >>> x, y = ivy.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
    >>> x
    ivy.array([1., 3.])
    >>> y
    ivy.array([2., 4.])

    Using converters:

    >>> s = StringIO("1.618, 2.296\n3.141, 4.669\n")
    >>> conv = {
    ...     0: lambda x: ivy.floor(float(x)),  # conversion fn for column 0
    ...     1: lambda x: ivy.ceil(float(x)),   # conversion fn for column 1
    ... }
    >>> ivy.loadtxt(s, delimiter=",", converters=conv)
    ivy.array([[1., 3.],
            [3., 5.]])

    Using a callable converter for all columns:

    >>> s = StringIO("0xDE 0xAD\n0xC0 0xDE")
    >>> import functools
    >>> conv = functools.partial(int, base=16)
    >>> ivy.loadtxt(s, converters=conv)
    ivy.array([[222., 173.],
            [192., 222.]])

    Handling values with different formatting:

    >>> s = StringIO('10.01 31.25-\n19.22 64.31\n17.57- 63.94')
    >>> def conv(fld):
    ...     return -float(fld[:-1]) if fld.endswith(b'-') else float(fld)
    ...
    >>> ivy.loadtxt(s, converters=conv)
    ivy.array([[ 10.01, -31.25],
            [ 19.22,  64.31],
            [-17.57,  63.94]])

    Handling values with different formatting and disabling encoding:

    >>> s = StringIO('10.01 31.25-\n19.22 64.31\n17.57- 63.94')
    >>> conv = lambda x: -float(x[:-1]) if x.endswith('-') else float(x)
    >>> ivy.loadtxt(s, converters=conv, encoding=None)
    ivy.array([[ 10.01, -31.25],
            [ 19.22,  64.31],
            [-17.57,  63.94]])

    Support for quoted fields is enabled with the `quotechar` parameter.
    Comment and delimiter characters are ignored when they appear within
    a quoted item delineated by `quotechar`:

    >>> s = StringIO('"alpha, #42", 10.0\n"beta, #64", 2.0\n')
    >>> dtype = ivy.dtype([("label", "U12"), ("value", float)])
    >>> ivy.loadtxt(s, dtype=dtype, delimiter=",", quotechar='"')
    ivy.array([('alpha, #42', 10.), ('beta, #64',  2.)],
            dtype=[('label', '<U12'), ('value', '<f8')])

    Quoted fields can be separated by multiple whitespace characters:

    >>> s = StringIO('"alpha, #42"       10.0\n"beta, #64" 2.0\n')
    >>> dtype = ivy.dtype([("label", "U12"), ("value", float)])
    >>> ivy.loadtxt(s, dtype=dtype, delimiter=None, quotechar='"')
    ivy.array([('alpha, #42', 10.), ('beta, #64',  2.)],
            dtype=[('label', '<U12'), ('value', '<f8')])

    Two consecutive quote characters within a quoted field are treated as a
    single escaped character:

    >>> s = StringIO('"Hello, my name is ""Monty""!"')
    >>> ivy.loadtxt(s, dtype="U", delimiter=",", quotechar='"')
    ivy.array('Hello, my name is "Monty"!', dtype='<U26')

    Read subset of columns when all rows do not contain equal number of values:

    >>> d = StringIO("1 2\n2 4\n3 9 12\n4 16 20")
    >>> ivy.loadtxt(d, usecols=(0, 1))
    ivy.array([[ 1.,  2.],
            [ 2.,  4.],
            [ 3.,  9.],
            [ 4., 16.]])
    """
    return current_backend().loadtxt(
        fname,
        dtype=dtype,
        comments=comments,
        delimiter=delimiter,
        converters=converters,
        skiprows=skiprows,
        usecols=usecols,
        unpack=unpack,
        ndmin=ndmin,
        max_rows=max_rows,
        encoding=encoding,
        quotechar=quotechar
    )
