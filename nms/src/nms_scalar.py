import numpy as np

def nms_py_scalar(input_array):
    size = len(input_array)
    output_array = np.zeros_like(input_array)

    if size == 0:
        return output_array

    if size == 1:
        return input_array

    # First element
    if input_array[0] >= input_array[1]:
        output_array[0] = input_array[0]
    else:
        output_array[0] = 0.0

    # Middle elements
    for i in range(1, size - 1):
        center = input_array[i]
        if center >= input_array[i-1] and center >= input_array[i+1]:
            output_array[i] = center
        else:
            output_array[i] = 0.0

    # Last element
    if input_array[size - 1] >= input_array[size - 2]:
        output_array[size - 1] = input_array[size - 1]
    else:
        output_array[size - 1] = 0.0
        
    return output_array