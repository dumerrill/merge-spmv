# Merge-based Parallel Sparse Matrix-Vector Multiplication

<h4>Abstract</h4>

We present a strictly balanced method for the parallel computation of sparse matrix-vector products (SpMV). Our algorithm operates directly upon the Compressed Sparse Row (CSR) sparse matrix format without preprocessing, inspection, reformatting, or supplemental encoding. Regardless of nonzero structure, our equitable 2D merge-based decomposition tightly bounds the workload assigned to each processing element. Furthermore, our technique is suitable for recursively partitioning CSR datasets themselves into multi-scale, distributed, NUMA, and GPU environments that are constrained by fixed-size local memories.

We evaluate our method on both CPU and GPU microarchitectures across a very large corpus of diverse sparse matrix datasets. We show that traditional CsrMV methods are inconsistent performers, often subject to order-of-magnitude performance variation across similarly-sized datasets. In comparison, our method provides predictable performance that is substantially uncorrelated to the distribution of nonzeros among rows and broadly improves upon that of current CsrMV methods.

<h4>Preprint</h4>

Duane Merrill and Michael Garland.  2016.  Merge-based Parallel Sparse Matrix-Vector Multiplication.  In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '16). IEEE Press, *to appear*.

![](http://wwwimages.adobe.com/content/dam/acom/en/legal/images/badges/Adobe_PDF_file_icon_32x32.png) [merge-based-spmv-sc16-preprint.pdf](https://github.com/dumerrill/merge-spmv/raw/master/merge-based-spmv-sc16-preprint.pdf)

<hr>
<h3>Prerequisites</h3>

<h4>CPU-based CsrMV</h4>

 * Intel CPU with AVX or wider vector extensions Intel CPU (two sockets of Xeon CPU E5-2695 v2 @ 2.40GHz as tested)
 * Intel C++ compiler and Math Kernel Library, both of which are included with Intel Parallel Studio (v2016.0.109 as tested)

<h4>GPU-based CsrMV</h4>
 * NVIDIA GPU with compute capability at least 3.5 (NVIDIA Tesla K40 as tested)
 * NVIDIA nvcc CUDA compiler and cuSPARSE library, both of which are included with CUDA Toolkit  (CUDA v7.5 as tested)
 * GNU GCC (v4.4.7 as tested)

Both CPU and GPU driver programs have been tested on CentOs 6.4 and Ubuntu 12.04/14.04, and are expected to run correctly under other Linux distributions.

<hr>
<h3>Datasets</h3>

Our test driver programs currently support input files encoded using the [matrix market format](http://math.nist.gov/MatrixMarket/formats.html).  All matrix market datasets used in this evaluation are publicly available from the Florida Sparse Matrix Repository.  Datasets can be downloaded individually from the [UF website](https://www.cise.ufl.edu/research/sparse/matrices/).

Additionally, the merge-spmv project provides users with the script `get_uf_datasets.sh` that will download and unpack the entire corpus used in this evaluation.  For example:

```sh
[dumerrill@dtlogin merge-spmv]$ ./get_uf_datasets.sh ufl_mtx

[dumerrill@dtlogin merge-spmv]$ ls ufl_mtx/
08blocks.mtx                         extr1b_Zeros_11.mtx                 nemsemm1_lo.mtx
1138_bus.mtx                         extr1b_Zeros_12.mtx                 nemsemm1.mtx
12month1.mtx                         extr1b_Zeros_13.mtx                 nemsemm1_z0.mtx
...
extr1b_Zeros_08.mtx                  nemsemm1_b.mtx                      Zewail_pubyear.mtx
extr1b_Zeros_09.mtx                  nemsemm1_c.mtx                      Zhao1.mtx
extr1b_Zeros_10.mtx                  nemsemm1_hi.mtx                     Zhao2.mtx
```

*Warning: at present time this corpus requires 243GB of free storage.*

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
