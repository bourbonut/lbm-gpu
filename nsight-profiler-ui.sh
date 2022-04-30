PATH_CUDA=$(find /usr/local/cuda-*/bin | sort -V | sed 1q)
$PATH_CUDA/ncu-ui $1
