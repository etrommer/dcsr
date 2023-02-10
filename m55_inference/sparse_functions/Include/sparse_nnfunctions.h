#ifndef _SPARSE_NNFUNCTIONS_H
#define _SPARSE_NNFUNCTIONS_H

#include "arm_nn_math_types.h"
#include "arm_nn_types.h"
#include "arm_mve.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern "C"
{
#endif

#define SIMD_GROUP_SIZE 16
#define BASE_BITS 4      //< Number of bits used to encode the unextended base value
#define EXTENSION_BITS 4 //< Maximum amount of bits that can be extended. Here: (5,6,7,8) = 4

    typedef struct
    {
        const uint8_t *bitmaps;
        const uint16_t *bitmasks;
        const uint8_t *delta_indices;
        const int16_t *row_offsets;
        const int8_t *group_minimums;
        const uint32_t nnze;
    } compressed_sparsity;

    arm_cmsis_nn_status sparse_convolve_wrapper_s8(
        const cmsis_nn_context *ctx,
        const compressed_sparsity *comp_sp,
        const cmsis_nn_conv_params *conv_params,
        const cmsis_nn_per_channel_quant_params *quant_params,
        const cmsis_nn_dims *input_dims,
        const int8_t *input_data,
        const cmsis_nn_dims *filter_dims,
        const int8_t *filter_data,
        const cmsis_nn_dims *bias_dims,
        const int32_t *bias_data,
        const cmsis_nn_dims *output_dims,
        int8_t *output_data);

    int32_t sparse_convolve_wrapper_s8_get_buffer_size(
        const cmsis_nn_conv_params *conv_params,
        const cmsis_nn_dims *input_dims,
        const cmsis_nn_dims *filter_dims,
        const cmsis_nn_dims *output_dims);

    arm_cmsis_nn_status sparse_convolve_1x1_s8_fast(
        const cmsis_nn_context *ctx,
        const compressed_sparsity *comp_sp,
        const cmsis_nn_conv_params *conv_params,
        const cmsis_nn_per_channel_quant_params *quant_params,
        const cmsis_nn_dims *input_dims,
        const int8_t *input_data,
        const cmsis_nn_dims *filter_dims,
        const int8_t *filter_data,
        const cmsis_nn_dims *bias_dims,
        const int32_t *bias_data,
        const cmsis_nn_dims *output_dims,
        int8_t *output_data);

    int32_t sparse_convolve_1x1_s8_fast_get_buffer_size(
        const cmsis_nn_dims *input_dims);

    arm_cmsis_nn_status sparsely_connected_s8(
        const cmsis_nn_context *ctx,
        const compressed_sparsity *comp_sp,
        const cmsis_nn_fc_params *fc_params,
        const cmsis_nn_per_tensor_quant_params *quant_params,
        const cmsis_nn_dims *input_dims,
        const int8_t *input,
        const cmsis_nn_dims *filter_dims,
        const int8_t *kernel,
        const cmsis_nn_dims *bias_dims,
        const int32_t *bias,
        const cmsis_nn_dims *output_dims,
        int8_t *output);

    int32_t sparsely_connected_s8_get_buffer_size(
        const cmsis_nn_dims *filter_dims);

    arm_cmsis_nn_status sparse_extract_row_indices(
        const compressed_sparsity *comp_sp,
        const uint32_t row_elements,
        const uint32_t row_groups,
        const uint32_t slope,

        uint32_t *bitmasks_idx,
        uint32_t groups_idx,

        uint8_t *indices_buffer);

    arm_cmsis_nn_status sparse_extract_row_values(
        const compressed_sparsity *comp_sp,
        const uint32_t row_elements,
        const uint32_t row_groups,
        const uint32_t slope,

        const int8_t *sparse_values,
        uint32_t *bitmasks_idx,
        const uint32_t groups_idx,

        const int8_t *values_buffer);

    arm_cmsis_nn_status sparse_eval_row(
        const compressed_sparsity *comp_sp,
        const uint32_t row_elements,
        const uint32_t row_groups,
        const uint32_t slope,

        const int8_t *sparse_values,
        uint32_t *bitmasks_idx,
        const uint32_t groups_idx,

        const int8_t *row_base,

        int32_t *result,
        int32_t *rhs_sum);

    arm_cmsis_nn_status group_offsets(
        const int8_t *offsets,
        const uint8_t slope,
        int16_t *buffer,
        const uint32_t row_groups);

    int32_t sparse_row_sum_s8(
        const uint32_t row_elements,
        const int8_t *sparse_values);
#ifdef __cplusplus
}
#endif

#endif
