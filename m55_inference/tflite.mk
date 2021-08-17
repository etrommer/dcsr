CXXSOURCES += $(shell find ./tensorflow -type f -name '*.cc') \

CSOURCES += \
			tensorflow/lite/c/common.c \

INCLUDE_DIRS += \
			third_party/flatbuffers/include \
			third_party/gemmlowp \
			third_party/ruy \

DEFS += -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DTF_LITE_MCU_DEBUG_LOG
