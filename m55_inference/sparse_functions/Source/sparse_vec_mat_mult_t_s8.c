#include "sparse_nnfunctions.h"
#include "sparse_nnsupportfunctions.h"

#include "arm_nnsupportfunctions.h"

arm_status sparse_vec_mat_mult_t_s8(
    const compressed_sparsity *comp_sp,
    const q7_t *lhs,
    const q7_t *rhs,
    const q31_t *bias,
    q7_t *dst,
    const int32_t lhs_offset,
    const int32_t rhs_offset,
    const int32_t dst_offset,
    const int32_t dst_multiplier,
    const int32_t dst_shift,
    const int32_t rhs_cols,
    const int32_t rhs_rows,
    const int32_t activation_min,
    const int32_t activation_max)
{
    int32_t lhs_sum = 0;
    {
        const int32_t col_loop_cnt = (rhs_cols + 15) / 16;
        uint32_t col_cnt = (uint32_t)rhs_cols;
        const int8_t *lhs_vec = lhs;
        for (int i = 0; i < col_loop_cnt; i++)
        {
            mve_pred16_t p = vctp8q(col_cnt);
            col_cnt -= 16;

            const int8x16_t input = vldrbq_z_s8(lhs_vec, p);
            lhs_sum = vaddvaq_p_s8(lhs_sum, input, p);
            lhs_vec += 16;
        }
    }

    const uint32_t avg_row_len = comp_sp->nnze / rhs_rows;
    uint32_t values_idx = 0;
    uint32_t bitmasks_idx = 0;
    uint32_t groups_idx = 0;

    for (int row = 0; row < rhs_rows; row++)
    {
        const uint32_t num_row_elems = avg_row_len + comp_sp->row_offsets[row];
        uint32_t slope = 0;
        const uint32_t num_row_groups = (num_row_elems + SIMD_GROUP_SIZE - 1) / SIMD_GROUP_SIZE;
        if (num_row_elems != 0)
        {
            slope = (rhs_cols + num_row_elems / 2) / num_row_elems;
        }

        int32_t acc_0 = 0;
        const int8_t *lhs_vec = lhs;
        const int8_t *rhs_0 = rhs;
        int32_t rhs_sum_0 = 0;

        (void)sparse_eval_row(
            comp_sp,
            num_row_elems,
            num_row_groups,
            slope,

            &rhs_0[values_idx],
            &bitmasks_idx,
            groups_idx,

            lhs_vec,

            &acc_0,
            &rhs_sum_0);
        if (bias)
        {
            acc_0 += *bias;
            bias++;
        }
        const int32_t offsets = (rhs_sum_0 * lhs_offset) + (lhs_sum * rhs_offset) + (lhs_offset * rhs_offset * rhs_cols);
        acc_0 += offsets;
        acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);
        acc_0 += dst_offset;

        // Clamp the result
        acc_0 = MAX(acc_0, activation_min);
        *dst = MIN(acc_0, activation_max);
        dst++;

        groups_idx += num_row_groups;
        values_idx += num_row_elems;
    }
    return ARM_MATH_SUCCESS;
}
