TARGET = $(OBJ_DIR)/mve_sparse_test

CSOURCES = \
				RTE/Device/ARMCM55/startup_ARMCM55.c \
				RTE/Device/ARMCM55/system_ARMCM55.c \
				hardfault_handler.c \
				test_files/test_matrix.c \
				test_files/test_main.c \
				test_files/debug_print.c \
				uart.c \
				retarget.c \
				sparse_functions/Source/sparse_mat_mul_core_1x_s8.c \
				sparse_functions/Source/sparse_mat_mul_core_4x_s8.c \
				sparse_functions/Source/sparse_extract.c \

INCLUDE_DIRS = \
				$(GCC_DIR)/arm-none-eabi/include \
				RTE \
				sparse_functions/Include \
				test_files \
				. \

CXXSOURCES = \
