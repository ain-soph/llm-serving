GEMM kernel files from ain-soph/mflux `serving` (src/mflux/models/common/quantization/w8a8_kernels, commit 4c740d3),
which took them from mlx-serve (David Dalcu, MIT, see LICENSE) as used by h3-image-mlx. 4c740d3 adds the edge-tile
bounds fix: the last row tile reads padded activation rows and the last column tile is moved inside the weight, so no
tile reads past either tensor. Outputs are byte-identical to the original kernel.
