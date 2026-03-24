# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These methods are used to collate samples fetched from dataset into Tensor(s).
These **needs** to be in global scope since Py2 doesn't support serializing
static methods.

`default_collate` and `default_convert` are exposed to users via 'dataloader.py'.
"""

import collections
import contextlib
import copy
import re
from typing import Callable, Dict, Optional, Tuple, Type, Union

import torch
from types import NoneType


np_str_obj_array_pattern = re.compile(r"[SaUO]")

# --- NEW: Helper class to handle None inputs in tensor-like loops ---
class SliceableNone:
    """
    A helper class that acts like None but allows for tensor-style slicing 
    and device movement without crashing.
    """
    def __getitem__(self, index):
        # Swallows any slicing (e.g., [:, 0]) and returns itself
        return self
    
    def to(self, *args, **kwargs):
        # Swallows .to(device) calls
        return self
    
    def detach(self):
        # Swallows .detach() calls
        return self
    
    def __bool__(self):
        # Evaluates to False, just like None
        return False
        
    def __repr__(self):
        return "SliceableNone"


def default_convert(data):
    r"""
    Convert each NumPy array element into a :class:`torch.Tensor`.
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        # array of string classes and object
        if (
            elem_type.__name__ == "ndarray"
            and np_str_obj_array_pattern.search(data.dtype.str) is not None
        ):
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                clone = copy.copy(data)
                clone.update({key: default_convert(data[key]) for key in data})
                return clone
            else:
                return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes)
    ):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, d in enumerate(data):
                    clone[i] = default_convert(d)
                return clone
            else:
                return elem_type([default_convert(d) for d in data])
        except TypeError:
            return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def collate(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    r"""
    General collate function that handles collection type of element within each batch.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](
                    batch, collate_fn_map=collate_fn_map
                )

    if isinstance(elem, collections.abc.Mapping):
        try:
            if isinstance(elem, collections.abc.MutableMapping):
                clone = copy.copy(elem)
                clone.update(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
                return clone
            else:
                return elem_type(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
        except TypeError:
            return {
                key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in zip(*batch)
            )
        )
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            # Relaxed check for None/SliceableNone mixed batches potentially
            pass 
            # raise RuntimeError("each element in list of batch should be of equal size")
        
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            # Check if this is a tuple of Nones (common when Dataset returns (None,)*steps)
            # If so, return a single SliceableNone instead of a list of SliceableNones
            result = [
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]
            
            # --- PATCH for (None,) tuples ---
            # If the result is a list containing only SliceableNone, return the SliceableNone directly.
            # This fixes the "list indices..." error by removing the list wrapper.
            if all(isinstance(x, SliceableNone) for x in result):
                return SliceableNone()
            
            return result 

        else:
            try:
                if isinstance(elem, collections.abc.MutableSequence):
                    clone = copy.copy(elem)  # type: ignore[arg-type]
                    for i, samples in enumerate(transposed):
                        clone[i] = collate(samples, collate_fn_map=collate_fn_map)
                    return clone
                else:
                    return elem_type(
                        [
                            collate(samples, collate_fn_map=collate_fn_map)
                            for samples in transposed
                        ]
                    )
            except TypeError:
                return [
                    collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    elem = batch[0]
    out = None
    if elem.is_nested:
        raise RuntimeError("Batches of nested tensors are not currently supported...")
    if elem.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }:
        raise RuntimeError("Batches of sparse tensors are not currently supported...")
        
    if torch.utils.data.get_worker_info() is not None:
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)


def collate_numpy_array_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    elem = batch[0]
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))
    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


def collate_numpy_scalar_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return torch.as_tensor(batch)


def collate_float_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return torch.tensor(batch, dtype=torch.float64)


def collate_int_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return torch.tensor(batch)


def collate_str_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return batch

def collate_none_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    # UPDATED: Return SliceableNone instead of None
    return SliceableNone()

default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {
    torch.Tensor: collate_tensor_fn
}
with contextlib.suppress(ImportError):
    import numpy as np
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn
    
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[str] = collate_str_fn
default_collate_fn_map[bytes] = collate_str_fn
default_collate_fn_map[NoneType] = collate_none_fn
# Add SliceableNone to map so it handles recursion correctly
default_collate_fn_map[SliceableNone] = lambda batch, **kwargs: SliceableNone()


def default_collate(batch):
    r"""
    Take in a batch of data and put the elements within the batch into a tensor with an additional outer dimension - batch size.
    """
    return collate(batch, collate_fn_map=default_collate_fn_map)