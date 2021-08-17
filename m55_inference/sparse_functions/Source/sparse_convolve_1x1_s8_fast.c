#include "sparse_nnfunctions.h"
#include "sparse_nnsupportfunctions.h"

#include "arm_nnsupportfunctions.h"

#include <ARMCM55.h>
#include <pmu_armv8.h>

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

arm_status sparse_convolve_1x1_s8_fast(const cmsis_nn_context *ctx,
                                       const compressed_sparsity *comp_sp,
                                       const cmsis_nn_conv_params *conv_params,
                                       const cmsis_nn_per_channel_quant_params *quant_params,
                                       const cmsis_nn_dims *input_dims,
                                       const q7_t *input_data,
                                       const cmsis_nn_dims *filter_dims,
                                       const q7_t *filter_data,
                                       const cmsis_nn_dims *bias_dims,
                                       const int32_t *bias_data,
                                       const cmsis_nn_dims *output_dims,
                                       q7_t *output_data)
{
    if (input_dims->c % 4 != 0 || conv_params->padding.w != 0 || conv_params->padding.h != 0 ||
        conv_params->stride.w != 1 || conv_params->stride.h != 1)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }

    (void)ctx;
    (void)filter_dims;
    (void)bias_dims;

#ifndef ARM_MATH_MVEI
#error Trying to compile without MVEI support
#endif

    const int32_t col_len = input_dims->w * input_dims->h * input_dims->n;
    const int32_t output_ch = output_dims->c;
    const int32_t input_ch = input_dims->c;
    const int32_t input_offset = conv_params->input_offset;
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;

    const uint32_t avg_row_len = comp_sp->nnze / output_ch;
    uint32_t values_idx = 0;
    uint32_t bitmasks_idx = 0;
    uint32_t groups_idx = 0;
#ifndef VALUE_BUFFERING
    int16_t *group_offset_buf = (int16_t *)(ctx->buf + 2 * input_dims->c);
#endif

    int extract = 0;
    int inference = 0;

    // Flipped iteration order when compared to CMSIS kernel
    // This keeps the filter index constant in the inner loop
    // and allows for buffering of unpacked sparse indices/values
    for (int i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
    {
        int cycle_count_before = ARM_PMU_Get_CCNTR();
        q7_t *output_data_temp = output_data + i_out_ch;

        const uint32_t num_row_elems = avg_row_len + comp_sp->row_offsets[i_out_ch];
        uint32_t slope = 0;
        const uint32_t num_row_groups = (num_row_elems + SIMD_GROUP_SIZE - 1) / SIMD_GROUP_SIZE;
        if (num_row_elems != 0)
        {
            slope = (input_ch + num_row_elems / 2) / num_row_elems;
        }

#ifndef VALUE_BUFFERING
        // Pre-Compute indices for this filter row and store in buffer
        (void)sparse_extract_row_indices(
            comp_sp,
            num_row_elems,
            num_row_groups,
            slope,
            &bitmasks_idx,
            groups_idx,
            ctx->buf);

        // Pre-Compute pointer offsets for this filter row and store in buffer
        (void)group_offsets(
            comp_sp->group_minimums + groups_idx,
            slope,
            group_offset_buf,
            num_row_groups);

        // Sum of row values for quantization
        const int32_t sum_row = sparse_row_sum_s8(
            num_row_elems,
            filter_data+values_idx
        );
#else
        memset(ctx->buf, 0, input_ch);
        // Reconstruct dense matrix row
        (void)sparse_extract_row_values(
            comp_sp,
            num_row_elems,
            num_row_groups,
            slope,

            filter_data + values_idx,
            &bitmasks_idx,
            groups_idx,

            ctx->buf);
#endif
        int cycle_count_extract = ARM_PMU_Get_CCNTR();
        for (int i_items = 0; i_items <= (col_len - 4); i_items += 4)
        {

            int32_t temp_out[4];
#ifndef VALUE_BUFFERING
            // Matmul with buffered 8-Bit indices
            (void)sparse_mat_mul_core_4x_s8(
                input_ch,
                input_data + i_items * input_ch,
                num_row_elems,
                filter_data+values_idx,
                ctx->buf,
                group_offset_buf,
                temp_out);
#else
            int32_t sum_row = 0;
            (void)arm_nn_mat_mul_core_4x_s8(
                input_ch,
                input_ch,
                input_data + i_items * input_ch,
                ctx->buf,
                &sum_row,
                temp_out
            );
#endif

            int32x4_t res = vldrwq_s32(temp_out);
            if (bias_data)
            {
                res = vaddq_n_s32(res, bias_data[i_out_ch]);
            }
            res = vaddq_n_s32(res, sum_row * input_offset);
            res = arm_requantize_mve(res, output_mult[i_out_ch], output_shift[i_out_ch]);
            res = vaddq_n_s32(res, out_offset);

            res = vmaxq_s32(res, vdupq_n_s32(out_activation_min));
            res = vminq_s32(res, vdupq_n_s32(out_activation_max));

            const uint32x4_t scatter_offset = {
                0, (uint32_t)output_ch, (uint32_t)output_ch * 2, (uint32_t)output_ch * 3};
            vstrbq_scatter_offset_s32(output_data_temp, scatter_offset, res);
            output_data_temp += (4 * output_ch);
        }
        /* Handle left over elements */
        for (int i_items = (col_len & ~0x3); i_items < col_len; i_items++)
        {
            int32_t acc = 0;
#ifndef VALUE_BUFFERING
            (void)sparse_mat_mul_core_1x_s8(
                input_data + i_items * input_ch,
                num_row_elems,
                filter_data+values_idx,
                ctx->buf,
                group_offset_buf,
                &acc);
#else
            int32_t sum_row = 0;
            (void)arm_nn_mat_mul_core_1x_s8(
                input_ch,
                input_data + i_items * input_ch,
                ctx->buf,
                &sum_row,
                &acc);
#endif

            if (bias_data)
            {
                acc += bias_data[i_out_ch];
            }
            acc += sum_row * input_offset;
            acc = arm_nn_requantize(acc, output_mult[i_out_ch], output_shift[i_out_ch]);
            acc += out_offset;

            acc = MAX(acc, out_activation_min);
            acc = MIN(acc, out_activation_max);
            *output_data_temp = acc;
            output_data_temp += output_ch;
        }

        int cycle_count_inference = ARM_PMU_Get_CCNTR();
        extract += cycle_count_extract - cycle_count_before;
        inference += cycle_count_inference - cycle_count_extract;

        groups_idx += num_row_groups;
        values_idx += num_row_elems;
    }
    /* printf("Extraction: %d Inference: %d\r\n", extract, inference);*/

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

int32_t sparse_convolve_1x1_s8_fast_get_buffer_size(const cmsis_nn_dims *input_dims)
{
#ifndef VALUE_BUFFERING
    // Indices for 1 row + Pre-calculated group offsets
    return input_dims->c + ((input_dims->c + 15) / 16) * sizeof(uint16_t);
#else
    // Values for 1 row
    return input_dims->c;
#endif
}

/**
 * @} end of NNConv group
 */
