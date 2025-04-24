import torch
import numpy as np


# IEEE 754 single precision masks
SIGN_MASK = 0x80000000  # Bit 31
EXPONENT_MASK = 0x7F800000  # Bits 23-30
MANTISSA_MASK = 0x007FFFFF  # Bits 0-22


def truncate_mantissa(x: torch.Tensor, msbs_to_keep: int) -> torch.Tensor:
    """Keeps only the num_msbs_to_keep most significant bits of the mantissa in IEEE 754 single-precision floating-point numbers."""

    if x.dtype != torch.float32:
        raise ValueError("Input tensor must be of type float32.")
    
    if msbs_to_keep < 0 or msbs_to_keep > 23:
        raise ValueError("num_bits_to_keep must be between 0 and 23.")

    if x.isnan().any():
        raise ValueError("Input tensor contains NaN values.")

    if x.isinf().any():
        raise ValueError("Input tensor contains infinity values.")

    if msbs_to_keep == 23:
        return x.clone()

    x_np = x.clone().detach().cpu().numpy()
    raw_bytes = x_np.tobytes()
    int_representation = np.frombuffer(raw_bytes, dtype=np.uint32).reshape(x_np.shape)

    # Extract components
    signs = int_representation & SIGN_MASK
    exponents = int_representation & EXPONENT_MASK
    mantissas = int_representation & MANTISSA_MASK
    
    # Calculate how many bits to zero out
    bits_to_truncate = 23 - msbs_to_keep
    
    # Mask out the least significant bits
    truncation_mask = ~((1 << bits_to_truncate) - 1) & MANTISSA_MASK
    truncated_mantissas = mantissas & truncation_mask
    
    # Combine components and convert back to pytorch tensor
    result_int = (signs | exponents | truncated_mantissas).astype(np.uint32)
    result_bytes = result_int.tobytes()
    result_np = np.frombuffer(result_bytes, dtype=np.float32).reshape(x_np.shape)
    result_pt = torch.from_numpy(result_np.copy()).to(x.device).to(x.dtype)
    return result_pt


def compress_exponent(x: torch.Tensor, lsbs_to_keep: int) -> torch.Tensor:
    """Clips the unbiased exponents of the input tensor to lie near zero. Zeros and subnormals are preserved."""

    if x.dtype != torch.float32:
        raise ValueError("Input tensor must be of type float32.")
    
    if lsbs_to_keep < 0 or lsbs_to_keep > 8:
        raise ValueError("num_remaining_bits must be between 0 and 8.")

    if x.isnan().any():
        raise ValueError("Input tensor contains NaN values.")

    if x.isinf().any():
        raise ValueError("Input tensor contains infinity values.")

    if lsbs_to_keep == 8:
        return x.clone()

    x_np = x.clone().detach().cpu().numpy()
    raw_bytes = x_np.tobytes()
    int_representation = np.frombuffer(raw_bytes, dtype=np.uint32).reshape(x_np.shape)

    # Extract components
    signs = int_representation & SIGN_MASK
    exponents = (int_representation & EXPONENT_MASK) >> 23
    mantissas = int_representation & MANTISSA_MASK
    zero_locs = (exponents == 0)  # Zeros and subnormals
    assert (0 <= exponents).all() and (exponents <= 254).all(), "Exponents out of range [0, 254]"  # 0-254 for normal and subnormal numbers. Cannot be 255 (NaN or Inf).

    unbiased_exps = exponents.astype(np.int32) - 127
    assert (unbiased_exps >= -127).all() and (unbiased_exps <= 127).all(), "Unbiased exponents out of range [-127, 127]"  # This is 255 possible values. Note that -127 corresponds

    # Truncate the exponent
    clip_val = 2 ** (lsbs_to_keep - 1) - 1  # 0: 127, 1: 63, 2: 31, 3: 15, 4: 7, 5: 3, 6: 1, 7: 0
    truncated_unbiased_exps = np.clip(unbiased_exps, -clip_val, clip_val)

    truncated_exps = truncated_unbiased_exps + 127
    truncated_exps[zero_locs] = 0  # Set the exponent to 0 for zeros and subnormals
    assert len(set(truncated_exps)) <= 2 ** lsbs_to_keep

    # Combine components and convert back to pytorch tensor
    result_int = (signs | (truncated_exps << 23) | mantissas).astype(np.uint32)
    result_bytes = result_int.tobytes()
    result_np = np.frombuffer(result_bytes, dtype=np.float32).reshape(x_np.shape)
    result_pt = torch.from_numpy(result_np.copy()).to(x.device).to(x.dtype)
    return result_pt


if __name__ == "__main__":
    x = torch.randn(10000, dtype=torch.float32)
    for b_mantissa in [23, 19, 15, 11, 7, 3]:
        for b_exponent in [8, 6, 4, 2]:
            print(f"Truncating mantissa to {b_mantissa} bits and exponent to {b_exponent} bits")
            y = truncate_mantissa(x, b_mantissa)
            z = compress_exponent(y, b_exponent)
            error = torch.abs(x - z).max()
            print(f"max error: {error}")
            print()