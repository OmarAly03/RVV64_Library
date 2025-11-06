#ifndef CONFIG_HPP
#define CONFIG_HPP

/**
 * @brief Master switch for kernel selection.
 * Set to 1 to use vectorized (RVV) kernels where available.
 * Set to 0 to use scalar C++ kernels.
 */
#define USE_VECTOR_KERNELS 0

// --- Kernel Selection ---
#if USE_VECTOR_KERNELS
    // --- Use Vector Implementations ---
    
    #define conv2d      conv2d_e32m8
    #define maxpool     maxpool_e32m8_tiled
    
    #define relu        relu_e32m8
    #define bias_add    bias_add_e32m8
    #define dense       dense_e32m8
    #define tensor_add  tensor_add_e32m8
    
    // Softmax must be handled separately in the .cpp
    // because its function signature is different.
    #define USE_VECTOR_SOFTMAX 1 

#else
    // --- Use Scalar Implementations ---
    #define conv2d      conv2d_scalar
    #define maxpool     maxpool_scalar_tile
    #define relu        relu_scalar
    #define bias_add    bias_add_scalar
    #define dense       dense_scalar
    #define tensor_add  tensor_add_scalar

    #define USE_VECTOR_SOFTMAX 0
#endif


#endif // CONFIG_HPP