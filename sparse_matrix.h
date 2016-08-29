/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Matrix data structures and parsing logic
 ******************************************************************************/

#pragma once

#include <cmath>
#include <cstring>

#include <iterator>
#include <string>
#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <list>
#include <fstream>
#include <stdio.h>

#ifdef CUB_MKL
    #include <numa.h>
    #include <mkl.h>
#endif

using namespace std;

/******************************************************************************
 * Graph stats
 ******************************************************************************/

struct GraphStats
{
    int         num_rows;
    int         num_cols;
    int         num_nonzeros;

    double      pearson_r;              // coefficient of variation x vs y (how linear the sparsity plot is)

    double      row_length_mean;        // mean
    double      row_length_std_dev;     // sample std_dev
    double      row_length_variation;   // coefficient of variation
    double      row_length_skewness;    // skewness

    void Display(bool show_labels = true)
    {
        if (show_labels)
            printf("\n"
                "\t num_rows: %d\n"
                "\t num_cols: %d\n"
                "\t num_nonzeros: %d\n"
                "\t row_length_mean: %.5f\n"
                "\t row_length_std_dev: %.5f\n"
                "\t row_length_variation: %.5f\n"
                "\t row_length_skewness: %.5f\n",
                    num_rows,
                    num_cols,
                    num_nonzeros,
                    row_length_mean,
                    row_length_std_dev,
                    row_length_variation,
                    row_length_skewness);
        else
            printf(
                "%d, "
                "%d, "
                "%d, "
                "%.5f, "
                "%.5f, "
                "%.5f, "
                "%.5f, ",
                    num_rows,
                    num_cols,
                    num_nonzeros,
                    row_length_mean,
                    row_length_std_dev,
                    row_length_variation,
                    row_length_skewness);
    }
};


/******************************************************************************
 * COO matrix type
 ******************************************************************************/

/**
 * COO matrix type.  A COO matrix is just a vector of edge tuples.  Tuples are sorted
 * first by row, then by column.
 */
template<typename ValueT, typename OffsetT>
struct CooMatrix
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    // COO edge tuple
    struct CooTuple
    {
        OffsetT            row;
        OffsetT            col;
        ValueT             val;

        CooTuple() {}
        CooTuple(OffsetT row, OffsetT col) : row(row), col(col) {}
        CooTuple(OffsetT row, OffsetT col, ValueT val) : row(row), col(col), val(val) {}
    };


    //---------------------------------------------------------------------
    // Data members
    //---------------------------------------------------------------------

    // Fields
    OffsetT             num_rows;
    OffsetT             num_cols;
    OffsetT             num_nonzeros;
    CooTuple*           coo_tuples;

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    // Constructor
    CooMatrix() : num_rows(0), num_cols(0), num_nonzeros(0), coo_tuples(NULL) {}


    /**
     * Clear
     */
    void Clear()
    {
        if (coo_tuples) delete[] coo_tuples;
        coo_tuples = NULL;
    }


    // Destructor
    ~CooMatrix()
    {
        Clear();
    }


    // Display matrix to stdout
    void Display()
    {
        cout << "COO Matrix (" << num_rows << " rows, " << num_cols << " columns, " << num_nonzeros << " non-zeros):\n";
        cout << "Ordinal, Row, Column, Value\n";
        for (OffsetT i = 0; i < num_nonzeros; i++)
        {
            cout << '\t' << i << ',' << coo_tuples[i].row << ',' << coo_tuples[i].col << ',' << coo_tuples[i].val << "\n";
        }
    }


    /**
     * Builds a COO sparse from a relabeled CSR matrix.
     */
    template <typename CsrMatrixT>
    void InitCsrRelabel(CsrMatrixT &csr_matrix, OffsetT* relabel_indices)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        num_rows        = csr_matrix.num_rows;
        num_cols        = csr_matrix.num_cols;
        num_nonzeros    = csr_matrix.num_nonzeros;
        coo_tuples      = new CooTuple[num_nonzeros];

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            for (OffsetT nonzero = csr_matrix.row_offsets[row]; nonzero < csr_matrix.row_offsets[row + 1]; ++nonzero)
            {
                coo_tuples[nonzero].row = relabel_indices[row];
                coo_tuples[nonzero].col = relabel_indices[csr_matrix.column_indices[nonzero]];
                coo_tuples[nonzero].val = csr_matrix.values[nonzero];
            }
        }
    }


    /**
     * Builds a MARKET COO sparse from the given file.
     */
    void InitMarket(
        const string&   market_filename,
        ValueT          default_value       = 1.0,
        bool            verbose             = false)
    {
        if (verbose) {
            printf("Reading... "); fflush(stdout);
        }

        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        std::ifstream ifs;
        ifs.open(market_filename.c_str(), std::ifstream::in);
        if (!ifs.good())
        {
            fprintf(stderr, "Error opening file\n");
            exit(1);
        }

        bool    array = false;
        bool    symmetric = false;
        bool    skew = false;
        OffsetT     current_nz = -1;
        char    line[1024];

        if (verbose) {
            printf("Parsing... "); fflush(stdout);
        }

        while (true)
        {
            ifs.getline(line, 1024);
            if (!ifs.good())
            {
                // Done
                break;
            }

            if (line[0] == '%')
            {
                // Comment
                if (line[1] == '%')
                {
                    // Banner
                    symmetric   = (strstr(line, "symmetric") != NULL);
                    skew        = (strstr(line, "skew") != NULL);
                    array       = (strstr(line, "array") != NULL);

                    if (verbose) {
                        printf("(symmetric: %d, skew: %d, array: %d) ", symmetric, skew, array); fflush(stdout);
                    }
                }
            }
            else if (current_nz == -1)
            {
                // Problem description
                OffsetT nparsed = sscanf(line, "%d %d %d", &num_rows, &num_cols, &num_nonzeros);
                if ((!array) && (nparsed == 3))
                {
                    if (symmetric)
                        num_nonzeros *= 2;

                    // Allocate coo matrix
                    coo_tuples = new CooTuple[num_nonzeros];
                    current_nz = 0;

                }
                else if (array && (nparsed == 2))
                {
                    // Allocate coo matrix
                    num_nonzeros = num_rows * num_cols;
                    coo_tuples = new CooTuple[num_nonzeros];
                    current_nz = 0;
                }
                else
                {
                    fprintf(stderr, "Error parsing MARKET matrix: invalid problem description: %s\n", line);
                    exit(1);
                }

            }
            else
            {
                // Edge
                if (current_nz >= num_nonzeros)
                {
                    fprintf(stderr, "Error parsing MARKET matrix: encountered more than %d num_nonzeros\n", num_nonzeros);
                    exit(1);
                }

                OffsetT row, col;
                double val;

                if (array)
                {
                    if (sscanf(line, "%lf", &val) != 1)
                    {
                        fprintf(stderr, "Error parsing MARKET matrix: badly formed current_nz: '%s' at edge %d\n", line, current_nz);
                        exit(1);
                    }
                    col = (current_nz / num_rows);
                    row = (current_nz - (num_rows * col));

                    coo_tuples[current_nz] = CooTuple(row, col, val);    // Convert indices to zero-based
                }
                else
                {
                    // Parse nonzero (note: using strtol and strtod is 2x faster than sscanf or istream parsing)
                    char *l = line;
                    char *t = NULL;

                    // parse row
                    row = strtol(l, &t, 0);
                    if (t == l)
                    {
                        fprintf(stderr, "Error parsing MARKET matrix: badly formed row at edge %d\n", current_nz);
                        exit(1);
                    }
                    l = t;

                    // parse col
                    col = strtol(l, &t, 0);
                    if (t == l)
                    {
                        fprintf(stderr, "Error parsing MARKET matrix: badly formed col at edge %d\n", current_nz);
                        exit(1);
                    }
                    l = t;

                    // parse val
                    val = strtod(l, &t);
                    if (t == l)
                    {
                        val = default_value;
                    }

                    coo_tuples[current_nz] = CooTuple(row - 1, col - 1, val);    // Convert indices to zero-based
                }

                current_nz++;

                if (symmetric && (row != col))
                {
                    coo_tuples[current_nz].row = coo_tuples[current_nz - 1].col;
                    coo_tuples[current_nz].col = coo_tuples[current_nz - 1].row;
                    coo_tuples[current_nz].val = coo_tuples[current_nz - 1].val * (skew ? -1 : 1);
                    current_nz++;
                }
            }
        }

        // Adjust nonzero count (nonzeros along the diagonal aren't reversed)
        num_nonzeros = current_nz;

        if (verbose) {
            printf("done. "); fflush(stdout);
        }

        ifs.close();
    }


    /**
     * Builds a dense matrix
     */
    int InitDense(
        OffsetT     num_rows,
        OffsetT     num_cols,
        ValueT      default_value   = 1.0,
        bool        verbose         = false)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        this->num_rows  = num_rows;
        this->num_cols  = num_cols;

        num_nonzeros    = num_rows * num_cols;
        coo_tuples      = new CooTuple[num_nonzeros];

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            for (OffsetT col = 0; col < num_cols; ++col)
            {
                coo_tuples[(row * num_cols) + col] = CooTuple(row, col, default_value);
            }
        }

        return 0;
    }


    /**
     * Builds a wheel COO sparse matrix having spokes spokes.
     */
    int InitWheel(
        OffsetT     spokes,
        ValueT      default_value   = 1.0,
        bool        verbose         = false)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        num_rows        = spokes + 1;
        num_cols        = num_rows;
        num_nonzeros    = spokes * 2;
        coo_tuples      = new CooTuple[num_nonzeros];

        // Add spoke num_nonzeros
        OffsetT current_nz = 0;
        for (OffsetT i = 0; i < spokes; i++)
        {
            coo_tuples[current_nz] = CooTuple(0, i + 1, default_value);
            current_nz++;
        }

        // Add rim
        for (OffsetT i = 0; i < spokes; i++)
        {
            OffsetT dest = (i + 1) % spokes;
            coo_tuples[current_nz] = CooTuple(i + 1, dest + 1, default_value);
            current_nz++;
        }

        return 0;
    }


    /**
     * Builds a square 2D grid CSR matrix.  Interior num_vertices have degree 5 when including
     * a self-loop.
     *
     * Returns 0 on success, 1 on failure.
     */
    int InitGrid2d(OffsetT width, bool self_loop, ValueT default_value = 1.0)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        OffsetT     interior_nodes  = (width - 2) * (width - 2);
        OffsetT     edge_nodes      = (width - 2) * 4;
        OffsetT     corner_nodes    = 4;
        num_rows                       = width * width;
        num_cols                       = num_rows;
        num_nonzeros                   = (interior_nodes * 4) + (edge_nodes * 3) + (corner_nodes * 2);

        if (self_loop)
            num_nonzeros += num_rows;

        coo_tuples          = new CooTuple[num_nonzeros];
        OffsetT current_nz    = 0;

        for (OffsetT j = 0; j < width; j++)
        {
            for (OffsetT k = 0; k < width; k++)
            {
                OffsetT me = (j * width) + k;

                // West
                OffsetT neighbor = (j * width) + (k - 1);
                if (k - 1 >= 0) {
                    coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                    current_nz++;
                }

                // East
                neighbor = (j * width) + (k + 1);
                if (k + 1 < width) {
                    coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                    current_nz++;
                }

                // North
                neighbor = ((j - 1) * width) + k;
                if (j - 1 >= 0) {
                    coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                    current_nz++;
                }

                // South
                neighbor = ((j + 1) * width) + k;
                if (j + 1 < width) {
                    coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                    current_nz++;
                }

                if (self_loop)
                {
                    neighbor = me;
                    coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                    current_nz++;
                }
            }
        }

        return 0;
    }


    /**
     * Builds a square 3D grid COO sparse matrix.  Interior num_vertices have degree 7 when including
     * a self-loop.  Values are unintialized, coo_tuples are sorted.
     */
    int InitGrid3d(OffsetT width, bool self_loop, ValueT default_value = 1.0)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            return -1;
        }

        OffsetT interior_nodes  = (width - 2) * (width - 2) * (width - 2);
        OffsetT face_nodes      = (width - 2) * (width - 2) * 6;
        OffsetT edge_nodes      = (width - 2) * 12;
        OffsetT corner_nodes    = 8;
        num_cols                       = width * width * width;
        num_rows                       = num_cols;
        num_nonzeros                     = (interior_nodes * 6) + (face_nodes * 5) + (edge_nodes * 4) + (corner_nodes * 3);

        if (self_loop)
            num_nonzeros += num_rows;

        coo_tuples              = new CooTuple[num_nonzeros];
        OffsetT current_nz    = 0;

        for (OffsetT i = 0; i < width; i++)
        {
            for (OffsetT j = 0; j < width; j++)
            {
                for (OffsetT k = 0; k < width; k++)
                {

                    OffsetT me = (i * width * width) + (j * width) + k;

                    // Up
                    OffsetT neighbor = (i * width * width) + (j * width) + (k - 1);
                    if (k - 1 >= 0) {
                        coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                        current_nz++;
                    }

                    // Down
                    neighbor = (i * width * width) + (j * width) + (k + 1);
                    if (k + 1 < width) {
                        coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                        current_nz++;
                    }

                    // West
                    neighbor = (i * width * width) + ((j - 1) * width) + k;
                    if (j - 1 >= 0) {
                        coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                        current_nz++;
                    }

                    // East
                    neighbor = (i * width * width) + ((j + 1) * width) + k;
                    if (j + 1 < width) {
                        coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                        current_nz++;
                    }

                    // North
                    neighbor = ((i - 1) * width * width) + (j * width) + k;
                    if (i - 1 >= 0) {
                        coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                        current_nz++;
                    }

                    // South
                    neighbor = ((i + 1) * width * width) + (j * width) + k;
                    if (i + 1 < width) {
                        coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                        current_nz++;
                    }

                    if (self_loop)
                    {
                        neighbor = me;
                        coo_tuples[current_nz] = CooTuple(me, neighbor, default_value);
                        current_nz++;
                    }
                }
            }
        }

        return 0;
    }
};




/******************************************************************************
 * CSR matrix type
 ******************************************************************************/

/**
 * CSR sparse format matrix
 */
template<
    typename ValueT,
    typename OffsetT>
struct CsrMatrix
{
    // Sort by rows, then columns
    struct CooComparator
    {
        template <typename CooTuple>
        bool operator()(const CooTuple &a, const CooTuple &b) const
        {
            return ((a.row < b.row) || ((a.row == b.row) && (a.col < b.col)));
        }
    };

    OffsetT     num_rows;
    OffsetT     num_cols;
    OffsetT     num_nonzeros;
    OffsetT*    row_offsets;
    OffsetT*    column_indices;
    ValueT*     values;


    // Whether to use NUMA malloc to always put storage on the same sockets (for perf repeatability)
    bool IsNumaMalloc()
    {
#ifdef CUB_MKL
        return (numa_available() >= 0);
#else
        return false;
#endif
    }

    /**
     * Initializer
     */
    void Init(
        CooMatrix<ValueT, OffsetT>  &coo_matrix,
        bool                        verbose = false)
    {
        num_rows        = coo_matrix.num_rows;
        num_cols        = coo_matrix.num_cols;
        num_nonzeros    = coo_matrix.num_nonzeros;

        // Sort by rows, then columns
        if (verbose) printf("Ordering..."); fflush(stdout);
        std::stable_sort(coo_matrix.coo_tuples, coo_matrix.coo_tuples + num_nonzeros, CooComparator());
        if (verbose) printf("done."); fflush(stdout);

#ifdef CUB_MKL

        if (IsNumaMalloc())
        {
            numa_set_strict(1);

            row_offsets     = (OffsetT*) numa_alloc_onnode(sizeof(OffsetT) * (num_rows + 1), 0);
            column_indices  = (OffsetT*) numa_alloc_onnode(sizeof(OffsetT) * num_nonzeros, 0);

            if (numa_num_task_nodes() > 1)
                values          = (ValueT*) numa_alloc_onnode(sizeof(ValueT) * num_nonzeros, 1);    // put on different socket than column_indices
            else
                values          = (ValueT*) numa_alloc_onnode(sizeof(ValueT) * num_nonzeros, 0);
        }
        else
        {
            values          = (ValueT*) mkl_malloc(sizeof(ValueT) * num_nonzeros, 4096);
            row_offsets     = (OffsetT*) mkl_malloc(sizeof(OffsetT) * (num_rows + 1), 4096);
            column_indices  = (OffsetT*) mkl_malloc(sizeof(OffsetT) * num_nonzeros, 4096);

        }

#else
        row_offsets     = new OffsetT[num_rows + 1];
        column_indices  = new OffsetT[num_nonzeros];
        values          = new ValueT[num_nonzeros];
#endif

        OffsetT prev_row = -1;
        for (OffsetT current_nz = 0; current_nz < num_nonzeros; current_nz++)
        {
            OffsetT current_row = coo_matrix.coo_tuples[current_nz].row;

            // Fill in rows up to and including the current row
            for (OffsetT row = prev_row + 1; row <= current_row; row++)
            {
                row_offsets[row] = current_nz;
            }
            prev_row = current_row;

            column_indices[current_nz]    = coo_matrix.coo_tuples[current_nz].col;
            values[current_nz]            = coo_matrix.coo_tuples[current_nz].val;
        }

        // Fill out any trailing edgeless vertices (and the end-of-list element)
        for (OffsetT row = prev_row + 1; row <= num_rows; row++)
        {
            row_offsets[row] = num_nonzeros;
        }
    }


    /**
     * Clear
     */
    void Clear()
    {
#ifdef CUB_MKL
        if (IsNumaMalloc())
        {
            numa_free(row_offsets, sizeof(OffsetT) * (num_rows + 1));
            numa_free(values, sizeof(ValueT) * num_nonzeros);
            numa_free(column_indices, sizeof(OffsetT) * num_nonzeros);
        }
        else
        {
            if (row_offsets)    mkl_free(row_offsets);
            if (column_indices) mkl_free(column_indices);
            if (values)         mkl_free(values);
        }

#else
        if (row_offsets)    delete[] row_offsets;
        if (column_indices) delete[] column_indices;
        if (values)         delete[] values;
#endif

        row_offsets = NULL;
        column_indices = NULL;
        values = NULL;

    }


    /**
     * Constructor
     */
    CsrMatrix(
        CooMatrix<ValueT, OffsetT>  &coo_matrix,
        bool                        verbose = false)
    {
        Init(coo_matrix, verbose);
    }


    /**
     * Destructor
     */
    ~CsrMatrix()
    {
        Clear();
    }


    /**
     * Get graph statistics
     */
    GraphStats Stats()
    {
        GraphStats stats;
        stats.num_rows = num_rows;
        stats.num_cols = num_cols;
        stats.num_nonzeros = num_nonzeros;

        //
        // Compute diag-distance statistics
        //

        OffsetT samples     = 0;
        double  mean        = 0.0;
        double  ss_tot      = 0.0;

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            OffsetT nz_idx_start    = row_offsets[row];
            OffsetT nz_idx_end      = row_offsets[row + 1];

            for (int nz_idx = nz_idx_start; nz_idx < nz_idx_end; ++nz_idx)
            {
                OffsetT col             = column_indices[nz_idx];
                double x                = (col > row) ? col - row : row - col;

                samples++;
                double delta            = x - mean;
                mean                    = mean + (delta / samples);
                ss_tot                  += delta * (x - mean);
            }
        }

        //
        // Compute deming statistics
        //

        samples         = 0;
        double mean_x   = 0.0;
        double mean_y   = 0.0;
        double ss_x     = 0.0;
        double ss_y     = 0.0;

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            OffsetT nz_idx_start    = row_offsets[row];
            OffsetT nz_idx_end      = row_offsets[row + 1];

            for (int nz_idx = nz_idx_start; nz_idx < nz_idx_end; ++nz_idx)
            {
                OffsetT col             = column_indices[nz_idx];

                samples++;
                double x                = col;
                double y                = row;
                double delta;

                delta                   = x - mean_x;
                mean_x                  = mean_x + (delta / samples);
                ss_x                    += delta * (x - mean_x);

                delta                   = y - mean_y;
                mean_y                  = mean_y + (delta / samples);
                ss_y                    += delta * (y - mean_y);
            }
        }

        samples         = 0;
        double s_xy     = 0.0;
        double s_xxy    = 0.0;
        double s_xyy    = 0.0;
        for (OffsetT row = 0; row < num_rows; ++row)
        {
            OffsetT nz_idx_start    = row_offsets[row];
            OffsetT nz_idx_end      = row_offsets[row + 1];

            for (int nz_idx = nz_idx_start; nz_idx < nz_idx_end; ++nz_idx)
            {
                OffsetT col             = column_indices[nz_idx];

                samples++;
                double x                = col;
                double y                = row;

                double xy =             (x - mean_x) * (y - mean_y);
                double xxy =            (x - mean_x) * (x - mean_x) * (y - mean_y);
                double xyy =            (x - mean_x) * (y - mean_y) * (y - mean_y);
                double delta;

                delta                   = xy - s_xy;
                s_xy                    = s_xy + (delta / samples);

                delta                   = xxy - s_xxy;
                s_xxy                   = s_xxy + (delta / samples);

                delta                   = xyy - s_xyy;
                s_xyy                   = s_xyy + (delta / samples);
            }
        }

        double s_xx     = ss_x / num_nonzeros;
        double s_yy     = ss_y / num_nonzeros;

        double deming_slope = (s_yy - s_xx + sqrt(((s_yy - s_xx) * (s_yy - s_xx)) + (4 * s_xy * s_xy))) / (2 * s_xy);

        stats.pearson_r = (num_nonzeros * s_xy) / (sqrt(ss_x) * sqrt(ss_y));

        //
        // Compute row-length statistics
        //

        // Sample mean
        stats.row_length_mean       = double(num_nonzeros) / num_rows;
        double variance             = 0.0;
        stats.row_length_skewness   = 0.0;
        for (OffsetT row = 0; row < num_rows; ++row)
        {
            OffsetT length              = row_offsets[row + 1] - row_offsets[row];
            double delta                = double(length) - stats.row_length_mean;
            variance   += (delta * delta);
            stats.row_length_skewness   += (delta * delta * delta);
        }
        variance                    /= num_rows;
        stats.row_length_std_dev    = sqrt(variance);
        stats.row_length_skewness   = (stats.row_length_skewness / num_rows) / pow(stats.row_length_std_dev, 3.0);
        stats.row_length_variation  = stats.row_length_std_dev / stats.row_length_mean;

        return stats;
    }


    /**
     * Display log-histogram to stdout
     */
    void DisplayHistogram()
    {
        // Initialize
        OffsetT log_counts[9];
        for (OffsetT i = 0; i < 9; i++)
        {
            log_counts[i] = 0;
        }

        // Scan
        OffsetT max_log_length = -1;
        OffsetT max_length = -1;
        for (OffsetT row = 0; row < num_rows; row++)
        {
            OffsetT length = row_offsets[row + 1] - row_offsets[row];
            if (length > max_length)
                max_length = length;

            OffsetT log_length = -1;
            while (length > 0)
            {
                length /= 10;
                log_length++;
            }
            if (log_length > max_log_length)
            {
                max_log_length = log_length;
            }

            log_counts[log_length + 1]++;
        }
        printf("CSR matrix (%d rows, %d columns, %d non-zeros, max-length %d):\n", (int) num_rows, (int) num_cols, (int) num_nonzeros, (int) max_length);
        for (OffsetT i = -1; i < max_log_length + 1; i++)
        {
            printf("\tDegree 1e%d: \t%d (%.2f%%)\n", i, log_counts[i + 1], (float) log_counts[i + 1] * 100.0 / num_cols);
        }
        fflush(stdout);
    }


    /**
     * Display matrix to stdout
     */
    void Display()
    {
        printf("Input Matrix (%d vertices, %d nonzeros):\n", (int) num_rows, (int) num_nonzeros);
        for (OffsetT row = 0; row < num_rows; row++)
        {
            printf("%d [@%d, #%d]: ", row, row_offsets[row], row_offsets[row + 1] - row_offsets[row]);
            for (OffsetT col_offset = row_offsets[row]; col_offset < row_offsets[row + 1]; col_offset++)
            {
                printf("%d (%f), ", column_indices[col_offset], values[col_offset]);
            }
            printf("\n");
        }
        fflush(stdout);
    }


};

