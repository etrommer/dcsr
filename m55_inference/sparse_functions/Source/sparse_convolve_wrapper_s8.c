#include "sparse_nnfunctions.h"

arm_cmsis_nn_status sparse_convolve_wrapper_s8(const cmsis_nn_context *ctx,
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
                                               int8_t *output_data)
{
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (input_dims->c % 4 == 0) &&
        (conv_params->stride.w == 1) && (conv_params->stride.h == 1) && (filter_dims->w == 1) && (filter_dims->h == 1))
    {
        return sparse_convolve_1x1_s8_fast(ctx,
                                           comp_sp,
                                           conv_params,
                                           quant_params,
                                           input_dims,
                                           input_data,
                                           filter_dims,
                                           filter_data,
                                           bias_dims,
                                           bias_data,
                                           output_dims,
                                           output_data);
    }
    else
    {
        // Only supports PW convolution at the moment
        return ARM_MATH_ARGUMENT_ERROR;
    }
}

int32_t sparse_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const cmsis_nn_dims *output_dims)
{
    (void)output_dims;
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (input_dims->c % 4 == 0) &&
        (conv_params->stride.w == 1) && (conv_params->stride.h == 1) && (filter_dims->w == 1) && (filter_dims->h == 1))
    {
        return sparse_convolve_1x1_s8_fast_get_buffer_size(input_dims);
    }
    else
    {
        // Only supports PW convolution at the moment
        return 0;
    }
}

/**
 * @} end of NNConv group
 */
