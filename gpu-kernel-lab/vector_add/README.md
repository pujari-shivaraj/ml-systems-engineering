
# Triton Vector Addition

## Objective
Implement vector addition using a custom Triton GPU kernel
and compare it against PyTorch.

## Hardware
- GPU: NVIDIA RTX 3050 6GB Laptop GPU
- Python: 3.12.3
- PyTorch: CUDA-enabled
- Triton: Installed in Python virtual environment

## Correctness
Validated against PyTorch using torch.testing.assert_close.

Tested vector sizes:
- 1,000
- 1,025
- 4,097
- 100,000
- 1,048,576

All correctness tests passed.

## Block-size experiment

Vector size: 1,048,576

| Block size | Median latency |
|---|---:|
| 128 | 108.30 us |
| 256 | 108.24 us |
| 512 | 106.89 us |
| 1024 | 107.44 us |

## Observation

Block size 512 achieved the lowest measured latency
in this experiment.

The improvement over block size 128 was approximately 1.3%.

The operation is memory-bound, so block-size tuning
produced only modest improvements.

## Conclusion

Implemented and validated a custom Triton vector
addition kernel, established a PyTorch baseline,
and investigated the effect of block size on performance.
