#include "relative_indexing.h"

void extract_relative(
        const int16_t row_len,
        const uint32_t offset,
        const int8_t *values,
        const uint8_t *delta_indices,
        int8_t *buffer)
{
    uint32_t acc = 0;
    for(uint32_t i = offset; i < offset + row_len; i++) {
        uint8_t relative_idx = delta_indices[i/2];
        if (i % 2 == 0) {
            relative_idx >>= 4;
        } else {
            relative_idx &= 0xf;
        }
        acc += relative_idx;
        buffer[acc] = values[i];
    }
}

void relative_eval_row(
    const int16_t row_len,
    const uint32_t offset,
    const int8_t *values,
    const uint8_t *delta_indices,
    const int8_t *activations,
    int32_t *result,
    int32_t *rhs_sum
    ) {

    uint8_t simd_buffer[16];

    int32_t acc = 0;
    int32_t sum_tmp = 0;

    uint32_t relative_acc = 0;
    uint32_t base_ptr_offset = 0;

    for(uint32_t i = offset; i < offset + row_len; i += 16) {
        base_ptr_offset += relative_acc;
        relative_acc = 0;

        for(uint32_t lane = 0; lane < 16; lane++){
            uint8_t relative_idx = delta_indices[(i+lane)/2];
            if ((i+lane) % 2 == 0) {
                relative_idx >>= 4;
            } else {
                relative_idx &= 0xf;
            }
            relative_acc += relative_idx;
            simd_buffer[lane] = relative_acc;
        }

        const mve_pred16_t p = vctp8q(offset + row_len - i);
        const int8x16_t a = vldrbq_z_s8(values + i, p);
        const uint8x16_t uidx = vldrbq_z_u8(simd_buffer, p);

        const int8x16_t v = vldrbq_gather_offset_z_s8(activations + base_ptr_offset, uidx, p);

        sum_tmp = vaddvaq_s8(sum_tmp, a);
        acc = vmladavaq_s8(acc, a, v);
    }

    *result = acc;
    *rhs_sum = sum_tmp;

    return ARM_MATH_SUCCESS;
}
