/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdio.h>
#include <stdlib.h>

#include <ARMCM55.h>
#include <pmu_armv8.h>

#include "tensorflow/lite/micro/examples/hello_world/main_functions.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

#include "file_utils.h"
#include "uart.h"

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.

void debug_log_printf(const char *s)
{
    printf(s);
}

int main(int argc, char *argv[])
{
    uart_init();

    // TODO: Fix magic numbers
    volatile uint32_t * const cpdlstate = (uint32_t *) 0xE001E300;
    volatile uint32_t * const ccr       = (uint32_t *) 0xE000ED14;
    *cpdlstate = 0;
    *ccr |= 1 << 19;

    RegisterDebugLogCallback(debug_log_printf);

    ARM_PMU_Enable();
    ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);

    file_utils_init();
    setup();
    loop();
    file_close();
    exit(1);

    while (true)
    {
    }
}
