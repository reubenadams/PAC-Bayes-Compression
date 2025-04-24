import torch
import numpy as np


def truncate_exponent(x: torch.Tensor, num_msbs_to_trunc: int) -> torch.Tensor:
    """
    Truncates the most significant bits of the exponent in IEEE 754 single-precision floating-point numbers.
    
    Args:
        x: Input tensor of type float32
        num_msbs_to_trunc: Number of most significant bits to truncate from the exponent (0-7)
        
    Returns:
        A new tensor with truncated exponents
        
    Note:
        This function preserves special values (NaN, infinity, zero) and only modifies normal values.
        The exponent truncation reduces the dynamic range of the floating point representation.
    """

    if x.dtype != torch.float32:
        raise ValueError("Input tensor must be of type float32.")
    
    if num_msbs_to_trunc < 0 or num_msbs_to_trunc > 7:
        raise ValueError("num_msbs_to_trunc must be between 0 and 7.")
    
    # No truncation needed
    if num_msbs_to_trunc == 0:
        return x.clone()

    x_np = x.clone().detach().cpu().numpy()
    
    # Get the raw bytes and convert to uint32 for bit manipulation
    raw_bytes = x_np.tobytes()
    int_representation = np.frombuffer(raw_bytes, dtype=np.uint32).reshape(x_np.shape)
    
    # IEEE 754 single precision masks
    SIGN_MASK = 0x80000000  # Bit 31
    EXPONENT_MASK = 0x7F800000  # Bits 23-30
    MANTISSA_MASK = 0x007FFFFF  # Bits 0-22
    
    # Extract components
    signs = int_representation & SIGN_MASK
    exponents = (int_representation & EXPONENT_MASK) >> 23
    mantissas = int_representation & MANTISSA_MASK
    
    # Create a mask for special values (NaN and Infinity have all exponent bits set)
    special_values_mask = (exponents == 0) | (exponents == 0xFF)
    
    # Constants for special cases
    ZERO_EXPONENT = 0
    SPECIAL_EXPONENT = 0xFF  # All 1's for NaN/Infinity
    
    # Create masks for different value categories
    zero_mask = (exponents == ZERO_EXPONENT)               # Zeros and subnormals
    special_mask = (exponents == SPECIAL_EXPONENT)         # NaN and Infinity
    normal_mask = ~(zero_mask | special_mask)              # Normal values
    
    # Only process normal values
    if np.any(normal_mask):
        # Get normal exponents
        normal_exponents = exponents[normal_mask]
        
        # Unbias (convert to signed exponent)
        unbiased = normal_exponents.astype(np.int32) - 127
        
        # Get absolute values and signs separately
        abs_exp = np.abs(unbiased)
        signs_exp = np.where(unbiased < 0, -1, 1)
        
        # Create a mask to remove the MSBs (keep only lower bits)
        max_exp_bits = 8  # IEEE 754 single precision has 8 exponent bits
        keep_bits_mask = (1 << (max_exp_bits - num_msbs_to_trunc)) - 1
        
        # Apply mask to preserve only lower bits, removing MSBs
        truncated_abs = abs_exp & keep_bits_mask
        
        # Ensure at least 1 for any non-zero exponent (prevent becoming subnormal)
        truncated_abs = np.maximum(truncated_abs, np.ones_like(truncated_abs) * (abs_exp > 0))
        
        # Reapply sign
        truncated_unbiased = truncated_abs * signs_exp
        
        # Rebias the exponent
        truncated_biased = truncated_unbiased + 127
        
        # Clamp exponent to valid range (1 to 254 for normal values)
        truncated_biased = np.clip(truncated_biased, 1, 254).astype(np.uint32)
        
        # Update exponents in the original array
        exponents[normal_mask] = truncated_biased
        
    # Leave special values unchanged (zeros, subnormals, NaN, infinity)
    
    # Reconstruct IEEE 754 representation
    result_int = signs | (exponents << 23) | mantissas
    
    # Convert back to float32
    result_bytes = result_int.tobytes()
    result_np = np.frombuffer(result_bytes, dtype=np.float32).reshape(x_np.shape)
    
    # Convert back to PyTorch tensor
    return torch.from_numpy(result_np.copy())