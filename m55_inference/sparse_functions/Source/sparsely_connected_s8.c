#include "sparse_nnfunctions.h"
#include "sparse_nnsupportfunctions.h"

arm_status sparsely_connected_s8(
    const cmsis_nn_context *ctx,
    const compressed_sparsity *comp_sp,
    const cmsis_nn_fc_params *fc_params,
    const cmsis_nn_per_tensor_quant_params *quant_params,
    const cmsis_nn_dims *input_dims,
    const q7_t *input,
    const cmsis_nn_dims *filter_dims,
    const q7_t *kernel,
    const cmsis_nn_dims *bias_dims,
    const int32_t *bias,
    const cmsis_nn_dims *output_dims,
    q7_t *output)
{
    (void)bias_dims;
    (void)ctx;
    int32_t batch_cnt = input_dims->n;

    while (batch_cnt)
    {
        sparse_vec_mat_mult_t_s8(
            comp_sp,
            input,
            kernel,
            bias,
            output,
            fc_params->input_offset,
            fc_params->filter_offset,
            fc_params->output_offset,
            quant_params->multiplier,
            quant_params->shift,
            filter_dims->n, /* col_dim or accum_depth */
            output_dims->c, /* row_dim or output_depth */
            fc_params->activation.min,
            fc_params->activation.max);
        input += filter_dims->n;
        output += output_dims->c;
        batch_cnt--;
    }
    return (ARM_MATH_SUCCESS);
}

int32_t sparsely_connected_s8_get_buffer_size(const cmsis_nn_dims *filter_dims)
{
    (void)filter_dims;
    return 0;
}

/**
 * @} end of FC group
 */
