# Merge-based Parallel Sparse Matrix-Vector Multiplication

<h4>Abstract</h4>

We present a strictly balanced method for the parallel computation of sparse matrix-vector products (SpMV). Our algorithm operates directly upon the Compressed Sparse Row (CSR) sparse matrix format without preprocessing, inspection, reformatting, or supplemental encoding. Regardless of nonzero structure, our equitable 2D merge-based decomposition tightly bounds the workload assigned to each processing element. Furthermore, our technique is suitable for recursively partitioning CSR datasets themselves into multi-scale, distributed, NUMA, and GPU environments that are constrained by fixed-size local memories.

We evaluate our method on both CPU and GPU microarchitectures across a very large corpus of diverse sparse matrix datasets. We show that traditional CsrMV methods are inconsistent performers, often subject to order-of-magnitude performance variation across similarly-sized datasets. In comparison, our method provides predictable performance that is substantially uncorrelated to the distribution of nonzeros among rows and broadly improves upon that of current CsrMV methods.

<h4>Preprint</h4>

Duane Merrill and Michael Garland.  2016.  Merge-based Parallel Sparse Matrix-Vector Multiplication.  In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '16). IEEE Press, *to appear*.

![](http://wwwimages.adobe.com/content/dam/acom/en/legal/images/badges/Adobe_PDF_file_icon_32x32.png) [merge-based-spmv-sc16-preprint.pdf](https://github.com/dumerrill/merge-spmv/raw/master/merge-based-spmv-sc16-preprint.pdf)

<hr>
<h3>The Algorithm</h3>

The central idea is to frame the parallel CsrMV decomposition as a logical merger of the CSR row-offsets and CSR non-zero data.  This equitable multi-partitioning ensures that no single processing element can be overwhelmed by assignment to (a) arbitrarily-long rows or (b) an arbitrarily-large number of zero-length rows.

![Merge-based parallel decomposition](https://github.com/dumerrill/merge-spmv/raw/master/merge_decomposition.png)

More specifically, we are logically merging two sequences: (*A*) the CSR row-offsets, and (*B*) the natural numbers ℕ that index the CSR nonzero values.  Individual processing elements are assigned equal-sized shares of this logical merger, with each processing element perform-ing a two-dimensional search to isolate the corresponding region within each list that would comprise its share.  The regions can then be treated as independent CsrMV subproblems and consumed directly from the CSR data structures using the sequential method.

In general, a merge computation can be viewed as a decision path of length |*A*|+|*B*| in which progressively larger elements are consumed from *A* and *B*.  Fig. 7 illustrates this as a two-dimensional grid in which the elements of *A* are arranged along the x-axis and the elements of *B* are arranged along the y-axis.  The decision path begins in the top-left corner and ends in the bottom-right. When traced sequentially, the merge path moves right when consuming the elements from *A* and down when consuming from *B*.  As a consequence, the path coordinates describe a complete schedule of element comparison and consumption across both input sequences.  Furthermore, each path coordinate can be linearly indexed by its grid diagonal, where diagonals are enumerated from top-left to bottom-right.  Per convention, the semantics of merge always prefer items from *A* over those from *B* when comparing same-valued items. This results in exactly one decision path.

![CsrMV partitioning using the 2D merge path](https://github.com/dumerrill/merge-spmv/raw/master/merge_spmv.png)

To parallelize across *p* threads, the grid is sliced diagonally into p swaths of equal width, and it is the job of each thread is to establish the route of the decision-path through its swath.  The fundamental insight is that any given path coordinate (*i*,*j*) can be found independently using a two-dimensional search procedure.  More specifically, the two elements *Ai* and *Bj* scheduled to be compared at diagonal *k* can be found via constrained binary search along that diagonal: find the first (*i*,*j*) where *Ai* is greater than all of the items consumed before *Bj*, given that *i*+*j*=*k*.  Each thread need only search the first diagonal in its swath; the remainder of its path segment can be trivially established via sequential comparisons seeded from that starting coordinate.

We can compute CsrMV using the merge-path decomposition by logically merging the row-offsets vector with the sequence of natural numbers ℕ used to index the values and column-indices vectors.  We emphasize that this merger is never physically realized, but rather serves to guide the equitable consumption of the CSR matrix.  By design, each contiguous vertical section of the decision path corresponds to a row of nonzeros in the CSR sparse matrix.  As threads follow the merge-path, they accumulate matrix-vector dot-products when moving downwards.  When moving rightwards, threads then flush these accumulated values to the corresponding row output in y and reset their accumulator. The partial sums from rows that span multiple threads can be aggregated in a subsequent reduce-value-by-key “fix-up” pass.

<hr>
<h3>Prerequisites</h3>

<h4>CPU-based CsrMV</h4>

 * Intel CPU with AVX or wider vector extensions Intel CPU (two sockets of Xeon CPU E5-2695 v2 @ 2.40GHz as tested)
 * Intel C++ compiler and Math Kernel Library, both of which are included with Intel Parallel Studio (v2016.0.109 as tested)

<h4>GPU-based CsrMV</h4>
 * NVIDIA GPU with compute capability at least 3.5 (NVIDIA Tesla K40 as tested)
 * NVIDIA nvcc CUDA compiler and cuSPARSE library, both of which are included with CUDA Toolkit  (CUDA v7.5 as tested)
 * GNU GCC (v4.4.7 as tested)

Both CPU and GPU test programs have been tested on CentOs 6.4 and Ubuntu 12.04/14.04, and are expected to run correctly under other Linux distributions.

<hr>
<h3>Datasets</h3>

Our test programs currently support input files encoded using the [matrix market format](http://math.nist.gov/MatrixMarket/formats.html).  All matrix market datasets used in this evaluation are publicly available from the Florida Sparse Matrix Repository.  Datasets can be downloaded individually from the [UF website](https://www.cise.ufl.edu/research/sparse/matrices/).

Additionally, the merge-spmv project provides users with the script `get_uf_datasets.sh` that will download and unpack the entire corpus used in this evaluation.  For example:

```sh
[dumerrill@bistromath merge-spmv]$ ./get_uf_datasets.sh ufl_mtx

[dumerrill@bistromath merge-spmv]$ ls ufl_mtx/
08blocks.mtx                         extr1b_Zeros_11.mtx                 nemsemm1_lo.mtx
1138_bus.mtx                         extr1b_Zeros_12.mtx                 nemsemm1.mtx
12month1.mtx                         extr1b_Zeros_13.mtx                 nemsemm1_z0.mtx
...                                  ...                                 ...
extr1b_Zeros_08.mtx                  nemsemm1_b.mtx                      Zewail_pubyear.mtx
extr1b_Zeros_09.mtx                  nemsemm1_c.mtx                      Zhao1.mtx
extr1b_Zeros_10.mtx                  nemsemm1_hi.mtx                     Zhao2.mtx
```

*Warning: at present time this corpus requires 243GB of free storage.*

<hr>
<h3>Building and Evaluating</h3>

We use GNU make to build CPU and GPU test programs, for example:

```sh
[dumerrill@bistromath merge-spmv]$ make cpu_spmv
[dumerrill@bistromath merge-spmv]$ make gpu_spmv sm=350
```

The GPU compilation takes an optional `sm=<cuda-arch, e.g., 350>` parameter that specifies the compute capability to compile for.  (For best results, compile the program for the maximal capability supported by the desired device.)

You can use the `cpu_spmv` and `gpu_spmv` test programs to compare CsrMV performance between MKL, cuSPARSE, and our merge-based implemenations.  For full usage information, specify the `--help` commandline option.  For example, on the CPU:

```sh
[dumerrill@bistromath merge-spmv]$ ./cpu_spmv --mtx=ufl_mtx/circuit5M.mtx
Reading... Parsing... (symmetric: 0, skew: 0, array: 0) done. ufl_mtx/circuit5M.mtx,
         num_rows: 5558326
         num_cols: 5558326
         num_nonzeros: 59524291
         row_length_mean: 10.70903
         row_length_std_dev: 1356.61627
         row_length_variation: 126.67964
         row_length_skewness: 868.02862

CSR matrix (5558326 rows, 5558326 columns, 59524291 non-zeros, max-length 1290501):
        Degree 1e-1:    0 (0.00%)
        Degree 1e0:     5205090 (93.64%)
        Degree 1e1:     325350 (5.85%)
        Degree 1e2:     20298 (0.37%)
        Degree 1e3:     7568 (0.14%)
        Degree 1e4:     0 (0.00%)
        Degree 1e5:     14 (0.00%)
        Degree 1e6:     6 (0.00%)

MKL CsrMV PASS
fp64: 0.0000 setup ms, 53.6234 avg ms, 2.22009 gflops, 23.445 effective GB/s

Merge CsrMV (Using 48 threads on 48 procs) PASS
fp64: 0.0000 setup ms, 12.6356 avg ms, 9.42165 gflops, 99.495 effective GB/s
```

For example, on the GPU:
```sh
[dumerrill@bistromath merge-spmv]$ ./gpu_spmv --mtx=ufl_mtx/circuit5M.mtx
Using device 0: Tesla K40m (PTX version 350, SM350, 15 SMs, 12131 free / 12204 total MB physmem, 288.384 GB/s @ 3004000 kHz mem clock, ECC off)
Reading... Parsing... (symmetric: 0, skew: 0, array: 0) done. ufl_mtx/circuit5M.mtx,

         num_rows: 5558326
         num_cols: 5558326
         num_nonzeros: 59524291
         row_length_mean: 10.70903
         row_length_std_dev: 1356.61627
         row_length_variation: 126.67964
         row_length_skewness: 868.02862

CSR matrix (5558326 rows, 5558326 columns, 59524291 non-zeros, max-length 1290501):
        Degree 1e-1:    0 (0.00%)
        Degree 1e0:     5205090 (93.64%)
        Degree 1e1:     325350 (5.85%)
        Degree 1e2:     20298 (0.37%)
        Degree 1e3:     7568 (0.14%)
        Degree 1e4:     0 (0.00%)
        Degree 1e5:     14 (0.00%)
        Degree 1e6:     6 (0.00%)

Merge-based CsrMV,      PASS
fp64: 0.0000 setup ms, 6.9239 avg ms, 17.19377 gflops, 181.571 effective GB/s (62.96% peak)

cuSPARSE CsrMV,         PASS
fp64: 0.0000 setup ms, 316.7821 avg ms, 0.37581 gflops, 3.969 effective GB/s (1.38% peak)

cuSPARSE HybMV,         PASS
fp64: 1059.8390 setup ms, 8.6398 avg ms, 13.77906 gflops, 145.511 effective GB/s (50.46% peak)
```

To evaluate performance across an entire directory of .mtx datasets, simply use the `eval_csrmv.sh` script as follows:

```sh
[dumerrill@bistromath merge-spmv]$ ./eval_csrmv.sh ufl_mtx cpu_spmv
file, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, method_name, setup_ms, avg_spmv_ms, gflops, effective_GBs
ufl_mtx/TSOPF_RS_b39_c7_b.mtx, 14098, 19, 7053, 0.50028, 0.50198, 1.00339, 0.02242, MKL CsrMV, 0.00000, 0.00858, 1.644377, 36.165, Merge CsrMV, 0.00000, 0.00894, 1.577218, 34.688,
ufl_mtx/hydr1c_A_20.mtx, 5308, 5308, 19503, 3.67427, 2.85421, 0.77681, 1.09421, MKL CsrMV, 0.00000, 0.00695, 5.614087, 65.309, Merge CsrMV, 0.00000, 0.00835, 4.673124, 54.362,
ufl_mtx/ford2.mtx, 100196, 100196, 544688, 5.43622, 1.55823, 0.28664, 4.60424, MKL CsrMV, 0.00000, 0.07418, 14.686324, 163.073, Merge CsrMV, 0.00000, 0.04599, 23.684943, 262.991,
ufl_mtx/jan99jac100.mtx, 34454, 34454, 215862, 6.26522, 6.61269, 1.05546, 5.07328, MKL CsrMV, 0.00000, 0.02180, 19.803599, 217.001, Merge CsrMV, 0.00000, 0.02023, 21.344958, 233.891,
ufl_mtx/fs_541_3.mtx, 541, 541, 4285, 7.92052, 1.85898, 0.23470, -0.16370, MKL CsrMV, 0.00000, 0.00587, 1.460053, 15.707, Merge CsrMV, 0.00000, 0.00602, 1.423420, 15.312,
ufl_mtx/adder_dcop_05.mtx, 1813, 1813, 11097, 6.12079, 30.77725, 5.02831, 41.95553, MKL CsrMV, 0.00000, 0.00710, 3.124376, 34.306, Merge CsrMV, 0.00000, 0.00685, 3.240352, 35.580,
ufl_mtx/as-735_G_478.mtx, 7716, 7716, 18836, 2.44116, 17.51463, 7.17471, 43.18932, MKL CsrMV, 0.00000, 0.00872, 4.321554, 53.837, Merge CsrMV, 0.00000, 0.00856, 4.401287, 54.831,
ufl_mtx/ct20stif.mtx, 52329, 52329, 2698463, 51.56726, 16.97858, 0.32925, 0.53036, MKL CsrMV, 0.00000, 0.19594, 27.544300, 278.648, Merge CsrMV, 0.00000, 0.12676, 42.576217, 430.716,
ufl_mtx/Wordnet3.mtx, 82670, 82670, 132964, 1.60837, 1.74744, 1.08647, 7.85034, MKL CsrMV, 0.00000, 0.04084, 6.512241, 89.416, Merge CsrMV, 0.00000, 0.03047, 8.727355, 119.831,
...
```

This will print matrix details and performance output in comma-separated format, one entry per line.  (Furthermore, all trivial datasets are skipped -- those having only a single row or column.)

<hr>
<h3>Open Source License</h3>

`merge-spmv` is available under the "New BSD" open-source license:

```
Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   *  Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   *  Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   *  Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
