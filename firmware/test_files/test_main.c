/*
   Cortex-M55 example

   Copyright (c) 2020 Arm Limited (or its affiliates). All rights reserved.
   Use, modification and redistribution of this file is subject to your possession of a
   valid End User License Agreement for the Arm Product of which these examples are part of
   and your compliance with all applicable terms and conditions of such licence agreement.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "test_matrix.h"
#include "sparse_nnsupportfunctions.h"
#include "sparse_nnfunctions.h"

#include "arm_nnsupportfunctions.h"

#include "uart.h"

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_RESET "\x1b[0m"

int main(void)
{
    uart_init();

    const compressed_sparsity comp_sp = {
        .bitmaps = bitmaps,
        .bitmasks = bitmasks,
        .delta_indices = delta_indices,
        .row_offsets = row_offsets,
        .group_minimums = minimums,
        .nnze = nnze};

    const uint32_t avg_row_len = nnze / matrix_rows;

    uint32_t values_idx = 0;
    uint32_t bitmasks_idx = 0;
    uint32_t groups_idx = 0;

    cmsis_nn_conv_params dummy_conv_params = {
        .activation.min = INT32_MIN,
        .activation.max = INT32_MAX,
    };

    int32_t multiplier = 1;
    int32_t shift = 0;
    cmsis_nn_per_channel_quant_params dummy_quant_params = {
        .multiplier = &multiplier,
        .shift = &shift,
    };
    int32_t bias = 0;

    for (size_t row = 0; row < matrix_rows; row++)
    {

        bool pass = true;
        int32_t reference_result = 0;
        int32_t results[] = {0, 0, 0, 0};
        int32_t sum_col = 0;

        const uint32_t row_len = avg_row_len + row_offsets[row];
        const uint32_t slope = (matrix_cols + row_len / 2) / row_len;
        const uint32_t num_groups = (row_len + 15) / 16;
        printf("Row: %3d, Len: %3d Groups: %3d ", (int)row, (int)row_len, (int)num_groups);

        // CMSIS Kernel for reference
        arm_nn_mat_mul_core_4x_s8(
            matrix_cols,
            matrix_cols,
            dummy_input,
            &reference[row * matrix_cols],
            1,
            &dummy_conv_params,
            &dummy_quant_params,
            &bias,
            results);
        printf("Result CMSIS: %8d ", (int)results[0]);
        reference_result = results[0];

        // Extract and buffer row offsets and indices
        group_offsets(
            comp_sp.group_minimums + groups_idx,
            slope,
            group_buffer,
            num_groups);

        uint32_t bitmasks_idx_temp = bitmasks_idx;
        (void)sparse_extract_row_indices(
            &comp_sp,
            row_len,
            num_groups,
            slope,
            &bitmasks_idx_temp,
            groups_idx,
            idx_buffer);

        // Single Row x Buffer
        (void)sparse_mat_mul_core_1x_s8(
            dummy_input,
            row_len,
            values + values_idx,
            idx_buffer,
            group_buffer,
            results);
        printf("Result MatMul1: %8d ", (int)results[0]);
        pass = pass && (reference_result == results[0]);

        // 4 Rows x Buffer
        (void)sparse_mat_mul_core_4x_s8(
            matrix_cols,
            dummy_input,
            row_len,
            values + values_idx,
            idx_buffer,
            group_buffer,
            results);
        printf("Result MatMul4: %8d ", (int)results[0]);
        pass = pass && (reference_result == results[0]);

        // Value Buffering
        bitmasks_idx_temp = bitmasks_idx;
        int8_t *value_buffer = (int8_t *)idx_buffer;
        memset(value_buffer, 0, matrix_cols);
        (void)sparse_extract_row_values(
            &comp_sp,
            row_len,
            num_groups,
            slope,

            values + values_idx,
            &bitmasks_idx_temp,
            groups_idx,

            value_buffer);

        arm_nn_mat_mul_core_1x_s8(
            matrix_cols,
            0,
            dummy_input,
            value_buffer,
            1,
            &dummy_conv_params,
            &dummy_quant_params,
            &bias,
            results);
        printf("Result Value Buffering: %8d ", (int)results[0]);
        pass = pass && (reference_result == results[0]);

        // Matrix-Vector Product (handles unpacking internally)
        int32_t acc, rhs_sum;
        (void)sparse_eval_row(
            &comp_sp,
            row_len,
            num_groups,
            slope,
            values + values_idx,
            &bitmasks_idx,
            groups_idx,
            dummy_input,
            &acc,
            &rhs_sum);
        printf("Result MatVec: %8d -- ", (int)acc);
        pass = pass && (reference_result == acc);

        if (pass == true)
        {
            printf(ANSI_COLOR_GREEN "PASS" ANSI_COLOR_RESET "\r\n");
        }
        else
        {
            printf(ANSI_COLOR_RED "FAIL" ANSI_COLOR_RESET "\r\n");
        }

        values_idx += row_len;
        groups_idx += num_groups;
    }
    exit(1);
}
