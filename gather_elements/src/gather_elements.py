import numpy as np

def gather_elements_scalar(data, indices, axis=0):
    """
    Scalar implementation of GatherElements operation.
    
    Args:
        data: Input tensor to gather from
        indices: Indices tensor specifying where to gather from
        axis: Axis along which to gather
    
    Returns:
        Output tensor with gathered values
    """
    # Output has the same shape as indices
    output = np.zeros_like(indices, dtype=data.dtype)
    
    # Normalize axis
    if axis < 0:
        axis = len(data.shape) + axis
    
    # Get iterator for all indices
    it = np.nditer(indices, flags=['multi_index'])
    
    for idx in it:
        # Build the index tuple for data
        multi_idx = list(it.multi_index)
        # Replace the axis dimension with the value from indices
        multi_idx[axis] = int(indices[it.multi_index])
        data_idx = tuple(multi_idx)
        
        # Gather the value from data
        output[it.multi_index] = data[data_idx]
    
    return output
