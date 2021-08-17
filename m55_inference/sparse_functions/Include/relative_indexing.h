#ifndef _RELATIVE_INDEXING_H
#define _RELATIVE_INDEXING_H

#include <stdint.h>

#include "arm_math_types.h"
#include "arm_nn_types.h"

void extract_relative(
        const int16_t row_len,
        const uint32_t offset,
        const int8_t *values,
        const uint8_t *delta_indices,
        int8_t *buffer);

void relative_eval_row(
    const int16_t row_len,
    const uint32_t offset,
    const int8_t *values,
    const uint8_t *delta_indices,
    const int8_t *activations,
    int32_t *result,
    int32_t *rhs_sum);
#endif
