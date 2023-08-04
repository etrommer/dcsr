#ifndef DEBUG_PRINT_H
#define DEBUG_PRINT_H

#include <arm_mve.h>
#include <stdio.h>

void debug_print_u8(uint8x16_t v);
void debug_print_s8(int8x16_t v);

#endif
