import torch 
from flwr.common import Parameters, ArrayRecord

EMPTY_TENSOR_KEY = "_empty"

def state_dict_to_vec(state_dict):
    """Converts a Pytorch state_dict into a 1-dimensional tensor"""
    return torch.cat(tuple(tensor.flatten() for tensor in state_dict.values()))

def vec_to_state_dict(state_dict, weights):
    """Converts a 1-dimensional tensor into a state dict of the form given by state_dict"""
    weights_used = 0
    for name, tensor in state_dict.items():
        state_dict[name] = weights[weights_used:weights_used + tensor.numel()].reshape(tensor.shape)
        weights_used += tensor.numel()

    return state_dict

#below taken from https://github.com/flwrlabs/flower/blob/main/framework/py/flwr/compat/common/recorddict_compat.py
def arrayrecord_to_parameters(record: ArrayRecord, keep_input: bool) -> Parameters:
    """Convert ParameterRecord to legacy Parameters.

    Warnings
    --------
    Because `Array`s in `ArrayRecord` encode more information of the
    array-like or tensor-like data (e.g their datatype, shape) than `Parameters` it
    might not be possible to reconstruct such data structures from `Parameters` objects
    alone. Additional information or metadata must be provided from elsewhere.

    Parameters
    ----------
    record : ArrayRecord
        The record to be conveted into Parameters.
    keep_input : bool
        A boolean indicating whether entries in the record should be deleted from the
        input dictionary immediately after adding them to the record.

    Returns
    -------
    parameters : Parameters
        The parameters in the legacy format Parameters.
    """
    parameters = Parameters(tensors=[], tensor_type="")

    for key in list(record.keys()):
        if key != EMPTY_TENSOR_KEY:
            parameters.tensors.append(record[key].data)

        if not parameters.tensor_type:
            # Setting from first array in record. Recall the warning in the docstrings
            # of this function.
            parameters.tensor_type = record[key].stype

        if not keep_input:
            del record[key]

    return parameters