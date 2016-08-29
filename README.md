# Merge-based Parallel Sparse Matrix-Vector Multiplication

<h4>Abstract</h4>

We present a strictly balanced method for the parallel computation of sparse matrix-vector products (SpMV). Our algorithm operates directly upon the Compressed Sparse Row (CSR) sparse matrix format without preprocessing, inspection, reformatting, or supplemental encoding. Regardless of nonzero structure, our equitable 2D merge-based decomposition tightly bounds the workload assigned to each processing element. Furthermore, our technique is suitable for recursively partitioning CSR datasets themselves into multi-scale, distributed, NUMA, and GPU environments that are constrained by fixed-size local memories.

We evaluate our method on both CPU and GPU microarchitectures across a very large corpus of diverse sparse matrix datasets. We show that traditional CsrMV methods are inconsistent performers, often subject to order-of-magnitude performance variation across similarly-sized datasets. In comparison, our method provides predictable performance that is substantially uncorrelated to the distribution of nonzeros among rows and broadly improves upon that of current CsrMV methods.

<h4>Preprint</h4>

Duane Merrill and Michael Garland.  2016.  Merge-based Parallel Sparse Matrix-Vector Multiplication.  In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '16). IEEE Press, *to appear*.

![](http://wwwimages.adobe.com/content/dam/acom/en/legal/images/badges/Adobe_PDF_file_icon_32x32.png) [merge-based-spmv-sc16-preprint.pdf](https://github.com/dumerrill/merge-spmv/raw/master/merge-based-spmv-sc16-preprint.pdf)

<br><hr>
<h3>Open Source License</h3>

`merge-spmv` is available under the "New BSD" open-source license:

```
Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
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
