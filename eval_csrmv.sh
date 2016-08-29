#!/bin/bash

if (( $# != 2 )); then
  echo "$0 <mtx dataset dir> <cpu_spmv | gpu_spmv [--device=...]>"
  exit 0
fi

echo "file, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, method_name, setup_ms, avg_spmv_ms, gflops, effective_GBs"

MTX_DIR=$1

shift

for i in `find $MTX_DIR -name *.mtx`
do
    ./$@ --quiet --mtx=$i
done
