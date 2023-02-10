#include "sparse_nnsupportfunctions.h"

arm_cmsis_nn_status sparse_mat_mul_core_1x_s8(
    const int8_t *row_base,
    const uint32_t num_elements,
    const int8_t *sparse_values,
    const uint8_t *sparse_indices,
    const int16_t *sparse_offsets,
    int32_t *output)
{

    int32_t acc = 0;
    const int8_t *ip_row = row_base;

    int32_t sum_tmp = 0;
    uint32_t cnt = num_elements;
/* #ifndef ASM_KERNEL*/
#if 0
    for (size_t col = 0; col < num_elements; col += 16)
    {

        /* Mask extra lanes */
        mve_pred16_t p = vctp8q(num_elements - col);
        const uint8x16_t uidx = vldrbq_z_u8(sparse_indices + col, p);

        /* Load sparse matrix values*/
        const int8x16_t a = vldrbq_z_s8(sparse_values + col, p);
        /* sum_tmp = vaddvaq_s8(sum_tmp, a);*/

        /* MAC Operations*/
        ip_row += sparse_offsets[col / 16];
        const int8x16_t v = vldrbq_gather_offset_z_s8(ip_row, uidx, p);
        acc = vmladavaq_s8(acc, a, v);
    }
#else
    __asm__ volatile(
        "   vldrb.8         q0, [%[col]], 16     \n"
        "   wlstp.8         lr, %[cnt], 1f       \n"
        "2:                                      \n"
        "   vldrb.8         q1, [%[ind]], 16     \n"
        "   ldrsh           %[cnt], [%[off]], 2      \n"
        "   add             %[row0], %[row0], %[cnt] \n"
        "   vldrb.8         q2, [%[row0], q1]    \n"
        "   vmladava.s8     %[out0], q0, q2      \n"
        "   vldrb.8         q0, [%[col]], 16     \n"
        "   letp            lr, 2b               \n"
        "1:                                      \n"
        :
        [col] "+r"(sparse_values),
        [ind] "+r"(sparse_indices),
        [off] "+r"(sparse_offsets),
        [row0] "+r"(ip_row),
        [out0] "+Te"(acc),
        [cnt] "+r"(cnt)
        :
        : "q0", "q1", "q2", "memory", "r14");
#endif
    output[0] = acc;

    return ARM_CMSIS_NN_SUCCESS;
}
