#ifndef _SPARSE_NNSUPPORTFUNCTIONS_H
#define _SPARSE_NNSUPPORTFUNCTIONS_H

#include "arm_math_types.h"
#include "arm_nn_types.h"

#include "sparse_nnfunctions.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern "C"
{
#endif

    arm_status sparse_mat_mul_core_1x_s8(
        const int8_t *row_base,

        const uint32_t num_elements,
        const int8_t *sparse_values,
        const uint8_t *sparse_indices,
        const int16_t *sparse_offsets,

        int32_t *output);

    arm_status sparse_mat_mul_core_4x_s8(
        const int32_t offset,
        const int8_t *row_base,

        const uint32_t num_elements,
        const int8_t *sparse_values,
        const uint8_t *sparse_indices,
        const int16_t *sparse_offsets,

        int32_t *output);

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
        const int32_t activation_max);

#ifdef __cplusplus
}
#endif

#endif
