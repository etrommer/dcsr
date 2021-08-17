#include "sparse_nnsupportfunctions.h"

arm_status sparse_mat_mul_core_4x_s8(
    const int32_t offset,
    const int8_t *row_base,

    const uint32_t num_elements,
    const int8_t *sparse_values,
    const uint8_t *sparse_indices,
    const int16_t *sparse_offsets,

    int32_t *output)
{
    int32_t acc_n0 = 0;
    int32_t acc_n1 = 0;
    int32_t acc_n2 = 0;
    int32_t acc_n3 = 0;

    const int8_t *ip_row_0 = row_base;
    const int8_t *ip_row_1 = row_base + offset;
    const int8_t *ip_row_2 = row_base + (2 * offset);
    const int8_t *ip_row_3 = row_base + (3 * offset);

    uint32_t cnt = num_elements;

#ifndef ASM_KERNEL
    for (size_t col = 0; col < num_elements; col += 16)
    {
        /* Mask extra lanes */
        mve_pred16_t p = vctp8q(num_elements - col);
        const uint8x16_t uidx = vldrbq_z_u8(sparse_indices + col, p);

        /* Load sparse matrix values*/
        const int8x16_t a = vldrbq_z_s8(sparse_values + col, p);

        /* MAC Operations*/
        ip_row_0 += sparse_offsets[col / 16];
        const int8x16_t v_0 = vldrbq_gather_offset_z_s8(ip_row_0, uidx, p);
        acc_n0 = vmladavaq_s8(acc_n0, a, v_0);

        ip_row_1 += sparse_offsets[col / 16];
        const int8x16_t v_1 = vldrbq_gather_offset_z_s8(ip_row_1, uidx, p);
        acc_n1 = vmladavaq_s8(acc_n1, a, v_1);

        ip_row_2 += sparse_offsets[col / 16];
        const int8x16_t v_2 = vldrbq_gather_offset_z_s8(ip_row_2, uidx, p);
        acc_n2 = vmladavaq_s8(acc_n2, a, v_2);

        ip_row_3 += sparse_offsets[col / 16];
        const int8x16_t v_3 = vldrbq_gather_offset_z_s8(ip_row_3, uidx, p);
        acc_n3 = vmladavaq_s8(acc_n3, a, v_3);
    }
#else
    __ASM volatile(
        "   vldrb.8         q0, [%[col]], 16     \n"
        "   wlstp.8         lr, %[cnt], 1f       \n"
        "2:                                      \n"
        "   vldrb.8         q1, [%[ind]], 16     \n"
        "   ldrsh           %[cnt], [%[off]], 2      \n"

        "   add             %[row0], %[row0], %[cnt] \n"

        "   vldrb.8         q2, [%[row0], q1]    \n"
        "   add             %[row1], %[row1], %[cnt] \n"
        "   vmladava.s8     %[out0], q0, q2      \n"

        "   vldrb.8         q3, [%[row1], q1]    \n"
        "   add             %[row2], %[row2], %[cnt] \n"
        "   vmladava.s8     %[out1], q0, q3      \n"

        "   vldrb.8         q4, [%[row2], q1]    \n"
        "   add             %[row3], %[row3], %[cnt] \n"
        "   vmladava.s8     %[out2], q0, q4      \n"

        "   vldrb.8         q3, [%[row3], q1]    \n"
        "   vmladava.s8     %[out3], q0, q3      \n"

        "   vldrb.8         q0, [%[col]], 16     \n"
        "   letp            lr, 2b               \n"
        "1:                                      \n"
        :
        [col] "+r"(sparse_values),
        [ind] "+r"(sparse_indices),
        [off] "+r"(sparse_offsets),
        [row0] "+r"(ip_row_0),
        [row1] "+r"(ip_row_1),
        [row2] "+r"(ip_row_2),
        [row3] "+r"(ip_row_3),
        [out0] "+Te"(acc_n0),
        [out1] "+Te"(acc_n1),
        [out2] "+Te"(acc_n2),
        [out3] "+Te"(acc_n3),
        [cnt] "+r"(cnt)
        :
        : "q0", "q1", "q2", "q3", "q4", "memory", "r14");
#endif

    output[0] = acc_n0;
    output[1] = acc_n1;
    output[2] = acc_n2;
    output[3] = acc_n3;

    return ARM_MATH_SUCCESS;
}
