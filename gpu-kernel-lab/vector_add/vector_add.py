
#c[i] = A[i] + B[i]
 

import torch 

import triton
import triton.language as tl


#1. GPU KERNEL

@triton.jit
def vector_add_kernel(A, B, C, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    #which block is this program processing

    pid = tl.program_id(0)

    #element indices handled by this block

    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)

    #ignore indices outside the vector
    mask = offsets < N

    #LOAD VALUES FROM GPU MEMORY
    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)

    #perform addition
    c = a + b

    #store results
    tl.store(C + offsets, c)

#python function that launches the kernel

def vector_add(A, B):

    assert A.is_cuda and B.is_cuda
    assert A.shape == B.shape

    N = A.numel()
    C = torch.empty_like(A)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    vector_add_kernel[grid](
        A, B, C, N, BLOCK_SIZE
    )

    return C



# 3. Test the kernel
if __name__ == "__main__":

    N = 1000

    A = torch.rand(N, device="cuda")
    B = torch.rand(N, device="cuda")

    # Our Triton implementation
    C_triton = vector_add(A, B)

    # PyTorch reference implementation
    C_torch = A + B

    # Check numerical correctness
    torch.testing.assert_close(C_triton, C_torch)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Vector size:", N)
    print("First 10 results:", C_triton[:10])
    print("Correctness test passed!")

    # Test 2: check nothing is written past the end
    C = torch.zeros(1024, device="cuda")     # extra room on purpose
    vector_add_kernel[(4,)](A, B, C, N, BLOCK_SIZE=256)
    assert torch.all(C[N:] == 0), "Kernel wrote out of bounds!"
    print("Out-of-bounds test passed!")