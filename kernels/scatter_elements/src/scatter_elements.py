import numpy as np

def scatter_elements_scalar(data, indices, updates, axis=0, reduction='none'):
    """
    Scalar implementation of ScatterElements operation.
    
    Args:
        data: Input tensor to scatter into
        indices: Indices tensor specifying where to scatter
        updates: Values to scatter
        axis: Axis along which to scatter
        reduction: 'none', 'add', 'mul', 'max', 'min'
    
    Returns:
        Output tensor with scattered values
    """
    output = data.copy()
    
    # Normalize axis
    if axis < 0:
        axis = len(data.shape) + axis
    
    # Get iterator for all indices
    it = np.nditer(indices, flags=['multi_index'])
    
    for idx in it:
        # Build the index tuple for output
        multi_idx = list(it.multi_index)
        # Replace the axis dimension with the value from indices
        multi_idx[axis] = int(indices[it.multi_index])
        output_idx = tuple(multi_idx)
        update_val = updates[it.multi_index]
        
        if reduction == 'none':
            output[output_idx] = update_val
        elif reduction == 'add':
            output[output_idx] += update_val
        elif reduction == 'mul':
            output[output_idx] *= update_val
        elif reduction == 'max':
            output[output_idx] = max(output[output_idx], update_val)
        elif reduction == 'min':
            output[output_idx] = min(output[output_idx], update_val)
    
    return output
