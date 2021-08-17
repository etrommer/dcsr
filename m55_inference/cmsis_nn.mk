INCLUDE_DIRS += \
			$(CMSIS_DIR) \
			$(CMSIS_DIR)/CMSIS/Core/Include \
			$(CMSIS_DIR)/CMSIS/DSP/Include \
			$(CMSIS_DIR)/CMSIS/NN/Include \

CSOURCES += \
			$(shell find $(CMSIS_DIR)/CMSIS/NN/Source -type f -name '*.c') \

DEFS += -DCMSIS_NN
