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

    double      diag_dist_mean;         // mean
    double      diag_dist_std_dev;      // sample std dev
    double      pearson_r;              // coefficient of variation

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
                "\t diag_dist_mean: %.2f\n"
                "\t diag_dist_std_dev: %.2f\n"
                "\t pearson_r: %f\n"
                "\t row_length_mean: %.5f\n"
                "\t row_length_std_dev: %.5f\n"
                "\t row_length_variation: %.5f\n"
                "\t row_length_skewness: %.5f\n",
                    num_rows,
                    num_cols,
                    num_nonzeros,
                    diag_dist_mean,
                    diag_dist_std_dev,
                    pearson_r,
                    row_length_mean,
                    row_length_std_dev,
                    row_length_variation,
                    row_length_skewness);
        else
            printf(
                "%d, "
                "%d, "
                "%d, "
                "%.2f, "
                "%.2f, "
                "%f, "
                "%.5f, "
                "%.5f, "
                "%.5f, "
                "%.5f, ",
                    num_rows,
                    num_cols,
                    num_nonzeros,
                    diag_dist_mean,
                    diag_dist_std_dev,
                    pearson_r,
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
 * TLSR matrix type
 ******************************************************************************/

/**
 * TLSR sparse format matrix
 */
template<
    typename ValueT,
    typename OffsetT>
struct TlsrMatrix
{
    // Return which stripe the given column is within
    OffsetT Stripe(OffsetT col)
    {
        return col / alpha;
    }

    // In the context of the transpose of the tuples, sort by stripe, then rows, then columns
    struct CooComparator
    {
        TlsrMatrix &gsr_matrix;

        CooComparator(TlsrMatrix &gsr_matrix) : gsr_matrix(gsr_matrix) {}

        template <typename CooTuple>
        bool operator()(const CooTuple &a, const CooTuple &b) const
        {
            if (gsr_matrix.Stripe(a.row) < gsr_matrix.Stripe(b.row))
                return true;
            else if (gsr_matrix.Stripe(b.row) < gsr_matrix.Stripe(a.row))
                return false;

            if (a.col < b.col)
                return true;
            else if (b.col < a.col)
                return false;

            return (a.row < b.row);
        }
    };

    struct RowTuple
    {
        OffsetT row;
        OffsetT col_offset;
//        bool    flagged;
    };

    struct StripeTuple
    {
        OffsetT row_offset;
//        bool    flagged;
    };


    OffsetT         num_rows;
    OffsetT         num_cols;
    OffsetT         num_nonzeros;
    OffsetT         alpha;
    OffsetT         beta;
    ValueT*         values;
    OffsetT*        column_indices;
    RowTuple*       rows;
    StripeTuple*    stripes;


    /**
     * Constructor (transpose of COO)
     */
    TlsrMatrix(
        OffsetT                     alpha,                  ///< Number of coarse graph stripes
        CooMatrix<ValueT, OffsetT>  &coo_matrix,            ///< Source matrix (edge-tuples are row->col)
        bool                        verbose = false)
    :
        num_rows(coo_matrix.num_cols),
        num_cols(coo_matrix.num_rows),
        num_nonzeros(coo_matrix.num_nonzeros),
        alpha(alpha),
        beta((num_cols + alpha - 1) / alpha),
        column_indices(new OffsetT[num_nonzeros]),
        values(new ValueT[num_nonzeros]),
        stripes(new StripeTuple[beta + 1]),
        rows(NULL)
    {
        // In the context of the transpose of the tuples, sort by stripe, then rows, then columns
        if (verbose) printf("Ordering %d slices, %d cols each)...", beta, alpha); fflush(stdout);
        std::stable_sort(coo_matrix.coo_tuples, coo_matrix.coo_tuples + num_nonzeros, CooComparator(*this));
        if (verbose) printf("done."); fflush(stdout);

        // Iterate through nonzero tuples
        std::vector<RowTuple>   row_tuples;
        OffsetT                 prev_stripe     = 0;
        OffsetT                 prev_row        = -1;
        for (OffsetT current_nz = 0; current_nz < num_nonzeros; current_nz++)
        {
            OffsetT current_stripe  = Stripe(coo_matrix.coo_tuples[current_nz].row);
            OffsetT current_row     = coo_matrix.coo_tuples[current_nz].col;

            // Fill out dense stripes up to and including the current stripe
            while (prev_stripe <= current_stripe)
            {
                prev_row = -1;
                StripeTuple new_stripe  = {row_tuples.size()};
                stripes[prev_stripe]    = new_stripe;
                ++prev_stripe;
            }

            if (prev_row != current_row)
            {
                // New row
                RowTuple new_row = {current_row, current_nz};
//                RowTuple new_row = {current_row, current_nz, false};
                row_tuples.push_back(new_row);
                prev_row = current_row;
            }

            column_indices[current_nz]    = coo_matrix.coo_tuples[current_nz].row;
            values[current_nz]            = coo_matrix.coo_tuples[current_nz].val;
        }

        // Fill out remaining stripes
        while (prev_stripe < beta + 1)
        {
            StripeTuple new_stripe = {row_tuples.size()};
            stripes[prev_stripe] = new_stripe;
            prev_stripe++;
        }

        // Copy row tuples
        rows = new RowTuple[row_tuples.size() + 1];
        for (int i = 0; i < row_tuples.size(); ++i)
            rows[i] = row_tuples[i];

        rows[row_tuples.size()].row         = -1;
        rows[row_tuples.size()].col_offset  = num_nonzeros;


        double disjoint_rows = 0.0;
        std::vector<OffsetT> last_stripe(num_rows, -2);
        for (OffsetT s = 0; s < beta; ++s)
        {
            for (OffsetT r = stripes[s].row_offset; r < stripes[s + 1].row_offset; ++r)
            {
                OffsetT row = rows[r].row;
                if (last_stripe[row] != s - 1)
                {
                    disjoint_rows += 1.0;
                    last_stripe[row] = s;
                }
            }
        }


//        double edge_efficiency      = double(num_nonzeros) / (double(row_tuples.size()) * alpha);
//        double row_efficiency       = double(num_rows) / row_tuples.size();

        double edge_efficiency      = double(num_nonzeros) / (double(row_tuples.size()) * alpha);
        double row_efficiency       = double(num_rows) / row_tuples.size();


        double edgerow_efficiency   = edge_efficiency * row_efficiency;

        printf(" <stripe rows %d, disjiont rows %f, edge %f, row %f, edgerow %f, geomean %f > ",
            row_tuples.size(), disjoint_rows / row_tuples.size(), edge_efficiency, row_efficiency, edgerow_efficiency, sqrt(edgerow_efficiency));
    }


    /**
     * Unflag all flagged rows and stripes
     */
    void Unflag()
    {
//        for (int i = 0; i < beta + 1; ++i)
//            stripes[i].flagged = false;

//        for (int i = 0; i < stripes[beta].row_offset; ++i)
//            rows[i].flagged = false;
    }


    /**
     * Clear
     */
    void Clear()
    {
        if (values)         delete[] values;
        if (column_indices) delete[] column_indices;
        if (rows)           delete[] rows;
        if (stripes)        delete[] stripes;

        values          = NULL;
        column_indices  = NULL;
        rows            = NULL;
        stripes         = NULL;
    }

    /**
     * Destructor
     */
    ~TlsrMatrix()
    {
        Clear();
    }


    /**
     * Display matrix to stdout
     */
    void Display()
    {
        printf("Input Matrix (%d vertices, %d nonzeros, %d alpha):\n", (int) num_rows, (int) num_nonzeros, (int) alpha);
        for (int stripe = 0; stripe < beta; ++stripe)
        {
            printf("Stripe %d\n", stripe);
            for (int row_offset = stripes[stripe].row_offset; row_offset < stripes[stripe + 1].row_offset; ++row_offset)
            {
                printf("\t");
                for (int col_offset = rows[row_offset].col_offset; col_offset < rows[row_offset + 1].col_offset; ++col_offset)
                {
                    printf("(%d->%d), ", column_indices[col_offset], rows[row_offset].row);
                }
                printf("\n");
            }
        }
        fflush(stdout);
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


    // Which allocation method to use
    bool IsNumaMalloc()
    {
#ifdef CUB_MKL
        return ((numa_available() >= 0) && (numa_num_task_nodes() > 1));
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
            values          = (ValueT*) numa_alloc_onnode(sizeof(ValueT) * num_nonzeros, 1);
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
        stats.diag_dist_mean            = mean;
        double variance                 = ss_tot / samples;
        stats.diag_dist_std_dev         = sqrt(variance);

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
        variance                    = 0.0;
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


/******************************************************************************
 * R2W matrix type
 ******************************************************************************/

/**
 * R2W sparse format matrix
 */
template<
    typename ValueT,
    typename OffsetT>
struct R2wMatrix
{
    /// Stripe descriptor
    struct StripeTuple
    {
        OffsetT     col_begin;
        OffsetT     begin;

        StripeTuple() {}

        StripeTuple(
            OffsetT col_begin,
            OffsetT begin)
        :
            col_begin(col_begin),
            begin(begin) {}

        void Display()
        {
            printf("col_begin: %d, begin: %d\n", col_begin, begin);
        }
    };


    /// Sort by columns, then rows
    struct CooComparator
    {
        template <typename CooTuple>
        bool operator()(const CooTuple &a, const CooTuple &b) const
        {
            return ((a.col < b.col) || ((a.col == b.col) && (a.row < b.row)));
        }
    };

    OffsetT     num_rows;
    OffsetT     num_cols;
    OffsetT     num_nonzeros;


    template <
        typename CooTupleItrT,
        typename StripeItrT>
    bool CheckSplit(
        CooTupleItrT    coo_tuples,
        StripeItrT      stripe)
    {
        StripeItrT next_stripe  = stripe;
        ++next_stripe;

        OffsetT nonzeros        = next_stripe->begin - stripe->begin;
        OffsetT width           = next_stripe->col_begin - stripe->col_begin;
        OffsetT split_width     = (width >> 1);

        if (nonzeros == 0)
        {
            // Empty
            return false;
        }

        if (width <= 1)
        {
            // Cannot be split
            return false;
        }

        OffsetT split_col_begin     = stripe->col_begin + split_width;
        OffsetT num_rows            = 1;
        OffsetT num_split_rows      = 1;

        for (int i = stripe->begin + 1; i < next_stripe->begin; ++i)
        {
            if (coo_tuples[i-1].col != coo_tuples[i].col)
            {
                // Changed row
                ++num_rows;
                ++num_split_rows;
            }
            else if ((coo_tuples[i - 1].row < split_col_begin) && (coo_tuples[i].row >= split_col_begin))
            {
                // Changed split-stripe
                ++num_split_rows;
            }
        }

        double volume_compression   = (double(num_rows) * width) / (double(num_split_rows) * split_width);
        double row_expansion        = double(num_split_rows) / num_rows;
        double row_compression      = (double(width) * num_rows) / (double(split_width) * num_split_rows);

        printf("volume_compression: %f, row_expansion: %f, row compression: %f\n",
            volume_compression, row_expansion, row_compression);

        return (row_compression > 1.0) ;
    }


    /**
     * Constructor
     */
    R2wMatrix(
        CooMatrix<ValueT, OffsetT>  &coo_matrix,
        bool                        verbose = false)
    :
        num_rows(coo_matrix.num_rows),
        num_cols(coo_matrix.num_cols),
        num_nonzeros(coo_matrix.num_nonzeros)
    {
        // Sort by columns, then rows
        if (verbose) printf("Ordering..."); fflush(stdout);
        std::stable_sort(coo_matrix.coo_tuples, coo_matrix.coo_tuples + num_nonzeros, CooComparator());
        if (verbose) printf("done.\n"); fflush(stdout);

        // Initialize list of stripes
        std::list<StripeTuple> stripes(2);
        stripes.front() = StripeTuple(0, 0);
        stripes.back()  = StripeTuple(num_cols, num_nonzeros);

        CheckSplit(coo_matrix.coo_tuples, stripes.begin());
    }


    /**
     * Clear
     */
    void Clear()
    {
    }


    /**
     * Destructor
     */
    ~R2wMatrix()
    {
        Clear();
    }


    /**
     * Display matrix to stdout
     */
    void Display()
    {

    }


};

/******************************************************************************
 * Matrix transformations
 ******************************************************************************/

// Comparator for ordering rows by degree (lowest first), then by row-id (lowest first)
template <typename OffsetT>
struct OrderByLow
{
    OffsetT* row_degrees;
    OrderByLow(OffsetT* row_degrees) : row_degrees(row_degrees) {}

    bool operator()(const OffsetT &a, const OffsetT &b)
    {
        if (row_degrees[a] < row_degrees[b])
            return true;
        else if (row_degrees[a] > row_degrees[b])
            return false;
        else
            return (a < b);
    }
};

// Comparator for ordering rows by degree (highest first), then by row-id (lowest first)
template <typename OffsetT>
struct OrderByHigh
{
    OffsetT* row_degrees;
    OrderByHigh(OffsetT* row_degrees) : row_degrees(row_degrees) {}

    bool operator()(const OffsetT &a, const OffsetT &b)
    {
        if (row_degrees[a] > row_degrees[b])
            return true;
        else if (row_degrees[a] < row_degrees[b])
            return false;
        else
            return (a < b);
    }
};



/**
 * Reverse Cuthill-McKee
 */
template <typename ValueT, typename OffsetT>
void RcmNewLabels(
    CsrMatrix<ValueT, OffsetT>&     matrix,
    OffsetT*                        relabel_indices)
{
    // Initialize row degrees
    OffsetT* row_degrees_in     = new OffsetT[matrix.num_rows];
    OffsetT* row_degrees_out    = new OffsetT[matrix.num_rows];
    for (OffsetT row = 0; row < matrix.num_rows; ++row)
    {
        row_degrees_in[row]         = 0;
        row_degrees_out[row]        = matrix.row_offsets[row + 1] - matrix.row_offsets[row];
    }
    for (OffsetT nonzero = 0; nonzero < matrix.num_nonzeros; ++nonzero)
    {
        row_degrees_in[matrix.column_indices[nonzero]]++;
    }

    // Initialize unlabeled set 
    typedef std::set<OffsetT, OrderByLow<OffsetT> > UnlabeledSet;
    typename UnlabeledSet::key_compare  unlabeled_comp(row_degrees_in);
    UnlabeledSet                        unlabeled(unlabeled_comp);
    for (OffsetT row = 0; row < matrix.num_rows; ++row)
    {
        relabel_indices[row]    = -1;
        unlabeled.insert(row);
    }

    // Initialize queue set
    std::deque<OffsetT> q;

    // Process unlabeled vertices (traverse connected components)
    OffsetT relabel_idx = 0;
    while (!unlabeled.empty())
    {
        // Seed the unvisited frontier queue with the unlabeled vertex of lowest-degree
        OffsetT vertex = *unlabeled.begin();
        q.push_back(vertex);

        while (!q.empty())
        {
            vertex = q.front();
            q.pop_front();

            if (relabel_indices[vertex] == -1)
            {
                // Update this vertex
                unlabeled.erase(vertex);
                relabel_indices[vertex] = relabel_idx;
                relabel_idx++;

                // Sort neighbors by degree
                OrderByLow<OffsetT> neighbor_comp(row_degrees_in);
                std::sort(
                    matrix.column_indices + matrix.row_offsets[vertex],
                    matrix.column_indices + matrix.row_offsets[vertex + 1],
                    neighbor_comp);

                // Inspect neighbors, adding to the out frontier if unlabeled
                for (OffsetT neighbor_idx = matrix.row_offsets[vertex];
                    neighbor_idx < matrix.row_offsets[vertex + 1];
                    ++neighbor_idx)
                {
                    OffsetT neighbor = matrix.column_indices[neighbor_idx];
                    q.push_back(neighbor);
                }
            }
        }
    }

/*
    // Reverse labels
    for (OffsetT row = 0; row < matrix.num_rows; ++row)
    {
        relabel_indices[row] = matrix.num_rows - relabel_indices[row] - 1;
    }
*/

    // Cleanup
    if (row_degrees_in) delete[] row_degrees_in;
    if (row_degrees_out) delete[] row_degrees_out;
}


/**
 * Reverse Cuthill-McKee
 */
template <typename ValueT, typename OffsetT>
void RcmRelabel(
    CsrMatrix<ValueT, OffsetT>&     csr_matrix,
    bool                            verbose = false)
{
    // Do not process if not square
    if (csr_matrix.num_cols != csr_matrix.num_rows)
    {
        if (verbose) {
            printf("RCM transformation ignored (not square)\n"); fflush(stdout);
        }
        return;
    }

    // Initialize relabel indices
    OffsetT* relabel_indices = new OffsetT[csr_matrix.num_rows];

    if (verbose) {
        printf("RCM relabeling... "); fflush(stdout);
    }

    RcmNewLabels(csr_matrix, relabel_indices);

    if (verbose) {
        printf("done. Reconstituting... "); fflush(stdout);
    }

    // Create a COO matrix from the relabel indices
    CooMatrix<ValueT, OffsetT> coo_matrix;
    coo_matrix.InitCsrRelabel(csr_matrix, relabel_indices);

    if (relabel_indices) delete[] relabel_indices;

    csr_matrix.Clear();
    csr_matrix.Init(coo_matrix, verbose);

    if (verbose) {
        printf("done. "); fflush(stdout);
    }
}




