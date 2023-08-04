#include "debug_print.h"

void debug_print_u8(uint8x16_t v) {
    uint8_t values[16];
    values[0] = vgetq_lane_u8(v, 0);
    values[1] = vgetq_lane_u8(v, 1);
    values[2] = vgetq_lane_u8(v, 2);
    values[3] = vgetq_lane_u8(v, 3);
    values[4] = vgetq_lane_u8(v, 4);
    values[5] = vgetq_lane_u8(v, 5);
    values[6] = vgetq_lane_u8(v, 6);
    values[7] = vgetq_lane_u8(v, 7);
    values[8] = vgetq_lane_u8(v, 8);
    values[9] = vgetq_lane_u8(v, 9);
    values[10] = vgetq_lane_u8(v, 10);
    values[11] = vgetq_lane_u8(v, 11);
    values[12] = vgetq_lane_u8(v, 12);
    values[13] = vgetq_lane_u8(v, 13);
    values[14] = vgetq_lane_u8(v, 14);
    values[15] = vgetq_lane_u8(v, 15);

    for(int i = 0; i < 16; i++) {
        printf("%2d ", values[i]);
        if (i == 3 || i == 7 || i == 11) {
            printf(" | ");
        }
    }
    printf("\n");
}

void debug_print_s8(int8x16_t v) {
    int8_t values[16];
    values[0] = vgetq_lane_s8(v, 0);
    values[1] = vgetq_lane_s8(v, 1);
    values[2] = vgetq_lane_s8(v, 2);
    values[3] = vgetq_lane_s8(v, 3);
    values[4] = vgetq_lane_s8(v, 4);
    values[5] = vgetq_lane_s8(v, 5);
    values[6] = vgetq_lane_s8(v, 6);
    values[7] = vgetq_lane_s8(v, 7);
    values[8] = vgetq_lane_s8(v, 8);
    values[9] = vgetq_lane_s8(v, 9);
    values[10] = vgetq_lane_s8(v, 10);
    values[11] = vgetq_lane_s8(v, 11);
    values[12] = vgetq_lane_s8(v, 12);
    values[13] = vgetq_lane_s8(v, 13);
    values[14] = vgetq_lane_s8(v, 14);
    values[15] = vgetq_lane_s8(v, 15);

    for(int i = 0; i < 16; i++) {
        printf("%2d ", values[i]);
        if (i == 3 || i == 7 || i == 11) {
            printf(" | ");
        }
    }
    printf("\n");
}
