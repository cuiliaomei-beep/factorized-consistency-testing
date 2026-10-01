# Template: one reproducer file = one program under test.
# Rules:
#   1. Define a function `f` (or mark the compilation target with the @torch.compile decorator / torch.compile(some_function);
#      if neither is present, the file may contain only one function);
#   2. Define `args`: a tuple of arguments for one valid call (tensors are re-cloned before each execution, so in-place is safe);
#   3. Do not use subprocess / open( / eval( / network access etc. (the safety gate rejects the whole file);
#   4. Code that needs a GPU ("cuda") is skipped on machines without a GPU;
#   5. The file name becomes the program name; issue number + topic is recommended.
#
# Optional: write the source issue in a module-level comment; it goes verbatim into the notes of the report.
# Source: https://github.com/pytorch/pytorch/issues/144699 (inductor full_like stride for non-contiguous input)
import torch


def f(x):
    return torch.full_like(x, 3)


args = (torch.randn(4, 5, 6).transpose(1, -1),)
