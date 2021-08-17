include cmsis_nn.mk
include tflite.mk

OBJ_DIR = build
TARGET = $(OBJ_DIR)/mve_sparse_tflu

AS = $(ARM_GCC_DIR)/bin/arm-none-eabi-as
LD = $(ARM_GCC_DIR)/bin/arm-none-eabi-g++
CXX = $(ARM_GCC_DIR)/bin/arm-none-eabi-g++
CC = $(ARM_GCC_DIR)/bin/arm-none-eabi-gcc
OC = $(ARM_GCC_DIR)/bin/arm-none-eabi-objcopy
OD = $(ARM_GCC_DIR)/bin/arm-none-eabi-objdump
OS = $(ARM_GCC_DIR)/bin/arm-none-eabi-size
RM = rm

INCLUDE_DIRS += \
			$(ARM_GCC_DIR)/arm-none-eabi/include \
			. \
			sparse_functions/Include \
			RTE \

ASOURCES += 

CSOURCES += hardfault_handler.c \
			RTE/Device/ARMCM55/startup_ARMCM55.c \
			RTE/Device/ARMCM55/system_ARMCM55.c \
			file_utils.c \
			sparse_functions/Source/sparse_convolve_1x1_s8_fast.c \
			sparse_functions/Source/sparse_convolve_wrapper_s8.c \
			sparse_functions/Source/sparse_mat_mul_core_4x_s8.c \
			sparse_functions/Source/sparse_mat_mul_core_1x_s8.c \
			sparse_functions/Source/sparsely_connected_s8.c \
			sparse_functions/Source/sparse_vec_mat_mult_t_s8.c \
			sparse_functions/Source/sparse_extract.c \
			sparse_functions/Source/relative_indexing.c \
			uart.c \
			retarget.c \

# Overwrite sources with test files when testing
ifdef TEST
include test_files/test.mk
include cmsis_nn.mk
endif

DEFS += -DARMCM55 -DCPU_M55=1 -DCORTEX_M_GENERIC
# DEFS += -DINTERMEDIATE_RESULTS
# DEFS += -DASM_KERNEL
DEFS += -DVALUE_BUFFERING
MCUFLAGS = -mcpu=cortex-m55 -mfloat-abi=hard -mthumb -funsigned-char -mlittle-endian
OPTIMIZE = -O3 -flto

WFLAGS = -Wall -Wextra
ASMFLAGS = $(MCUFLAGS) $(DEFS) $(OPTIMIZE) -fmessage-length=0 -c
CFLAGS = -c $(MCUFLAGS) $(DEFS) $(OPTIMIZE) $(WFLAGS) $(addprefix -I,$(INCLUDE_DIRS)) -std=c11 -ffunction-sections -fdata-sections
CXXFLAGS = -c $(MCUFLAGS) $(DEFS) $(OPTIMIZE) $(WFLAGS) $(addprefix -I,$(INCLUDE_DIRS)) -std=c++11 -ffunction-sections -fdata-sections
LIBS = -lnosys

ifdef FPGA
LSCRIPT = ./platform.ld
else
LSCRIPT = ./gcc_arm.ld
endif

LDFLAGS += $(MCUFLAGS) \
		--entry=Reset_Handler \
		-static -g\
		-Xlinker -Map=$(TARGET).map \
		-T$(LSCRIPT) \
		-Wl,-gc-sections \
		$(OPTIMIZE) \
		--specs=nosys.specs \
		--specs=nano.specs \

OBJECTS = $(addsuffix .o,$(addprefix $(OBJ_DIR)/,$(basename $(ASOURCES))))
OBJECTS += $(addsuffix .o,$(addprefix $(OBJ_DIR)/,$(basename $(CSOURCES))))
OBJECTS += $(addsuffix .o,$(addprefix $(OBJ_DIR)/,$(basename $(CXXSOURCES))))

.PHONY: clean all sim size

all: $(TARGET).axf

$(TARGET).axf: $(TARGET).elf
	cp $< $@

$(TARGET).elf: $(OBJECTS)
	@echo Linking: $@
	$(LD) -o $@ $^ $(LIBS) $(LDFLAGS)
	$(OD) -h -S -C --no-addresses $(TARGET).elf > $(TARGET).lst

size: $(TARGET).elf
	@echo
	@echo == Object size ==
	@$(OS) --format=berkeley $<

$(OBJ_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	@echo %% $(notdir $<)
	@$(CXX) -c -o $@ $(CXXFLAGS) $<

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	@echo %% $(notdir $<)
	@$(CC) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR)/%.o: %.s
	@mkdir -p $(dir $@)
	@echo %% $(notdir $<)
	@$(CXX) -x assembler-with-cpp -c -o $@ $(CXXFLAGS) $<

clean:
	@echo Cleaning:
	$(RM) -rf $(OBJ_DIR)/*

sim: $(TARGET).elf
	FVP_Corstone_SSE-300_Ethos-U55 -a $< -f config.txt
