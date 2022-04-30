if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
  PATH_CUDA=$(find /usr/local/cuda-*/bin | sort -V | sed 1q)
  PATH_PYTHON=$(which python)
  sudo $PATH_CUDA/ncu -o profile $PATH_PYTHON $1
fi
