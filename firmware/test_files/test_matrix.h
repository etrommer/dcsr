#ifndef TEST_MATRIX_H
#define TEST_MATRIX_H

#include <stdint.h>

extern const int8_t reference[];
extern const int8_t values[];
extern const uint16_t bitmasks[];
extern const uint8_t bitmaps[];
extern const uint8_t delta_indices[];
extern const int8_t minimums[];
extern const int16_t row_offsets[];
extern const uint32_t nnze;
extern const uint32_t matrix_rows;
extern const uint32_t matrix_cols;

extern const int8_t dummy_input[];
extern uint8_t idx_buffer[];
extern int16_t group_buffer[];
#endif
