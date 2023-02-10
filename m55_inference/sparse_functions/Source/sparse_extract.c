#include "sparse_nnfunctions.h"
#include "arm_mve.h"

inline int32_t sparse_row_sum_s8(
    const uint32_t row_elements,
    const int8_t *sparse_values)
{
    int32_t sum = 0;
    for (size_t idx = 0; idx < row_elements; idx += 16)
    {
        const mve_pred16_t p = vctp8q(row_elements - idx);
        const int8x16_t a = vldrbq_z_s8(sparse_values + idx, p);
        sum = vaddvaq_s8(sum, a);
    }
    return sum;
}

static inline uint8x16_t base_index(const uint8_t slope)
{
    uint8x16_t slope_vec = vdupq_n_u8(slope);
    uint8x16_t idx_base = vidupq_u8(0, 1);
    /* uint8x16_t idx_base = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};*/
    return vmulq_u8(idx_base, slope_vec);
}

static inline uint8x16_t bit_extend(uint8x16_t base_values, const uint8_t extension_map, const uint16_t *bitmasks, uint32_t *bitmasks_idx)
{
    for (size_t bitposition = 0; bitposition < EXTENSION_BITS; bitposition++)
    {
        // Is extension bit in map set for the current position?
        if ((extension_map & (1 << bitposition)) != 0)
        {
            // Consume the next bitmap for extension
            const mve_pred16_t pred = (mve_pred16_t) * (bitmasks + (*bitmasks_idx)++);
            const uint8x16_t extension_mask = vdupq_n_u8(1 << (BASE_BITS + bitposition));
            base_values = vorrq_m_u8(base_values, base_values, extension_mask, pred);
        }
    }
    return base_values;
}

static inline uint8x16_t load_base_indices(const size_t group_idx, const uint8_t *base_indices, const bool use_upper_nibble)
{
    // Base indices for two consecutive groups are packed into 16 Byites
    // Even groups occupy upper nibble, odd groups occupy lower nibble
    // For index 0-15 of Group A (even) and Group B (odd):
    // | A0B0 | A1B1 | A2B2 | ... | A15B15 |
    uint8x16_t group_base_indices = vldrbq_u8(base_indices + (group_idx / 2) * SIMD_GROUP_SIZE);
    if (use_upper_nibble == true)
    {
        group_base_indices = vshrq_n_u8(group_base_indices, 4);
    }
    else
    {
        const uint8x16_t mask = vdupq_n_u8(0x0f);
        group_base_indices = vandq_u8(group_base_indices, mask);
    }
    return group_base_indices;
}

static inline uint8_t load_extension_map(const size_t group_idx, const uint8_t *bitmaps, const bool use_upper_nibble)
{
    // 4-Bit extension Bitmaps for two groups are packed in one byte.
    // Even group occupies upper nibble, odd group occupies lower nibble.
    // For Groups A-H:
    // | AB | CD | EF | GH |
    uint8_t extension_map = bitmaps[group_idx / 2];
    if (use_upper_nibble == true)
    {
        extension_map = extension_map >> 4;
    }
    return extension_map;
}

// Generate 8-Bit Group indices from 4-Bit base values and
// Extension bitmaps
arm_cmsis_nn_status sparse_extract_row_indices(
    const compressed_sparsity *comp_sp,
    const uint32_t row_elements,
    const uint32_t row_groups,
    const uint32_t slope,

    uint32_t *bitmasks_idx,
    uint32_t groups_idx,

    uint8_t *indices_buffer)
{
    const uint8x16_t idx_base = base_index(slope);
    for (size_t group = groups_idx; group < groups_idx + row_groups; group++)
    {
        bool use_upper_nibble = ((group % 2) == 0);
        uint8x16_t deltas = load_base_indices(group, comp_sp->delta_indices, use_upper_nibble);
        const uint8_t extension_map = load_extension_map(group, comp_sp->bitmaps, use_upper_nibble);
        deltas = bit_extend(deltas, extension_map, comp_sp->bitmasks, bitmasks_idx);

        /* Add extended delta values to base slope*/
        const uint8x16_t uidx = vaddq_u8(idx_base, deltas);

        /* Store to output buffer with lane predication*/
        const mve_pred16_t p = vctp8q(row_elements - SIMD_GROUP_SIZE * (group - groups_idx));
        vst1q_p(indices_buffer + (group - groups_idx) * SIMD_GROUP_SIZE, uidx, p);
    }

    return ARM_CMSIS_NN_SUCCESS;
}

// Generate 8-Bit Group indices from 4-Bit base values and
// Extension bitmaps
arm_cmsis_nn_status sparse_extract_row_values(
    const compressed_sparsity *comp_sp,
    const uint32_t row_elements,
    const uint32_t row_groups,
    const uint32_t slope,

    const int8_t *sparse_values,
    uint32_t *bitmasks_idx,
    const uint32_t groups_idx,

    const int8_t *values_buffer)
{
    const uint8x16_t idx_base = base_index(slope);
    const int8_t *ip_row = values_buffer;
    int32_t group_offset = 0;

    for (size_t group = groups_idx; group < groups_idx + row_groups; group++)
    {
        bool use_upper_nibble = ((group % 2) == 0);
        uint8x16_t deltas = load_base_indices(group, comp_sp->delta_indices, use_upper_nibble);
        const uint8_t extension_map = load_extension_map(group, comp_sp->bitmaps, use_upper_nibble);
        deltas = bit_extend(deltas, extension_map, comp_sp->bitmasks, bitmasks_idx);

        /* Add extended delta values to base slope*/
        const uint8x16_t uidx = vaddq_u8(idx_base, deltas);
        const mve_pred16_t p = vctp8q(row_elements - 16 * (group - groups_idx));
        const int8x16_t a = vldrbq_z_s8(sparse_values + (group - groups_idx) * 16, p);

        /* Adjust base pointers */
        group_offset += (int32_t)comp_sp->group_minimums[group];
        ip_row += group_offset;
        group_offset = slope * SIMD_GROUP_SIZE;

        /* MAC Operations*/
        vstrbq_scatter_offset_p_s8(ip_row, uidx, a, p);
    }
    return ARM_CMSIS_NN_SUCCESS;
}

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
    int32_t *rhs_sum)
{
    const uint8x16_t idx_base = base_index(slope);
    int32_t acc = 0;
    const int8_t *ip_row = row_base;
    int32_t sum_tmp = 0;
    int32_t group_offset = 0;

    for (size_t group = groups_idx; group < groups_idx + row_groups; group++)
    {
        bool use_upper_nibble = ((group % 2) == 0);
        uint8x16_t deltas = load_base_indices(group, comp_sp->delta_indices, use_upper_nibble);
        const uint8_t extension_map = load_extension_map(group, comp_sp->bitmaps, use_upper_nibble);
        deltas = bit_extend(deltas, extension_map, comp_sp->bitmasks, bitmasks_idx);

        /* Add extended delta values to base slope*/
        const uint8x16_t uidx = vaddq_u8(idx_base, deltas);
        const mve_pred16_t p = vctp8q(row_elements - 16 * (group - groups_idx));
        const int8x16_t a = vldrbq_z_s8(sparse_values + (group - groups_idx) * 16, p);

        /* Adjust base pointers */
        group_offset += comp_sp->group_minimums[group];
        ip_row += group_offset;
        group_offset = slope * SIMD_GROUP_SIZE;

        /* MAC Operations*/
        const int8x16_t v = vldrbq_gather_offset_z_s8(ip_row, uidx, p);
        sum_tmp = vaddvaq_s8(sum_tmp, a);
        acc = vmladavaq_s8(acc, a, v);
    }

    *result = acc;
    *rhs_sum = sum_tmp;

    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status group_offsets(
    const int8_t *offsets,
    const uint8_t slope,
    int16_t *buffer,
    const uint32_t row_groups)
{
    int32_t group_offset = 0;
    for (size_t group = 0; group < row_groups; group++)
    {
        group_offset += offsets[group];
        buffer[group] = group_offset;
        group_offset = slope * SIMD_GROUP_SIZE;
    }
    return ARM_CMSIS_NN_SUCCESS;
}
