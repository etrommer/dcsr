/*
 * Copyright (c) 2019-2020 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "uart.h"

// armclang retargeting
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6100100)
#include <rt_misc.h>
#include <rt_sys.h>

/* Standard IO device handles. */
#define STDIN  0x8001
#define STDOUT 0x8002
#define STDERR 0x8003

#define RETARGET(fun) _sys##fun

#else
/*
 * This type is used by the _ I/O functions to denote an open
 * file.
 */
typedef int FILEHANDLE;

/*
 * Open a file. May return -1 if the file failed to open.
 */
extern FILEHANDLE _open(const char * /*name*/, int /*openmode*/);

/* Standard IO device handles. */
#define STDIN  0x00
#define STDOUT 0x01
#define STDERR 0x02

#define RETARGET(fun) fun

#endif

/* Standard IO device name defines. */
const char __stdin_name[] __attribute__((aligned(4)))  = "STDIN";
const char __stdout_name[] __attribute__((aligned(4))) = "STDOUT";
const char __stderr_name[] __attribute__((aligned(4))) = "STDERR";

void _ttywrch(int ch) {
    (void)fputc(ch, stdout);
}

FILEHANDLE RETARGET(_open)(const char *name, int openmode) {
    (void)openmode;

    if (strcmp(name, __stdin_name) == 0) {
        return (STDIN);
    }

    if (strcmp(name, __stdout_name) == 0) {
        return (STDOUT);
    }

    if (strcmp(name, __stderr_name) == 0) {
        return (STDERR);
    }

    return -1;
}

int RETARGET(_write)(FILEHANDLE fh, const unsigned char *buf, unsigned int len, int mode) {
    (void)mode;

    switch (fh) {
    case STDOUT:
    case STDERR: {
        int c;

        while (len-- > 0) {
            c = fputc(*buf++, stdout);
            if (c == EOF) {
                return EOF;
            }
        }

        return 0;
    }
    default:
        return EOF;
    }
}

int RETARGET(_read)(FILEHANDLE fh, unsigned char *buf, unsigned int len, int mode) {
    (void)mode;

    switch (fh) {
    case STDIN: {
        int c;

        while (len-- > 0) {
            c = fgetc(stdin);
            if (c == EOF) {
                return EOF;
            }

            *buf++ = (unsigned char)c;
        }

        return 0;
    }
    default:
        return EOF;
    }
}

int RETARGET(_istty)(FILEHANDLE fh) {
    switch (fh) {
    case STDIN:
    case STDOUT:
    case STDERR:
        return 1;
    default:
        return 0;
    }
}

int RETARGET(_close)(FILEHANDLE fh) {
    if (RETARGET(_istty(fh))) {
        return 0;
    }

    return -1;
}

int RETARGET(_seek)(FILEHANDLE fh, long pos) {
    (void)fh;
    (void)pos;

    return -1;
}

int RETARGET(_ensure)(FILEHANDLE fh) {
    (void)fh;

    return -1;
}

long RETARGET(_flen)(FILEHANDLE fh) {
    if (RETARGET(_istty)(fh)) {
        return 0;
    }

    return -1;
}

int RETARGET(_tmpnam)(char *name, int sig, unsigned maxlen) {
    (void)name;
    (void)sig;
    (void)maxlen;

    return 1;
}

char *RETARGET(_command_string)(char *cmd, int len) {
    (void)len;

    return cmd;
}

void RETARGET(_exit)(int return_code) {
    exit(return_code);
}

int system(const char *cmd) {
    (void)cmd;

    return 0;
}

time_t time(time_t *timer) {
    time_t current;

    current = 0; // To Do !! No RTC implemented

    if (timer != NULL) {
        *timer = current;
    }

    return current;
}

void _clock_init(void) {
#if 0
    // Example implementation based on SysTick
    // For instance, use a counting var in a SysTick interrupt handler
    // for clock() to use
    SysTick->LOAD = (uint32_t) ((SystemCoreClock/100)-1UL);
    NVIC_SetPriority (SysTick_IRQn, (1UL << __NVIC_PRIO_BITS) - 1UL);
    SysTick->VAL = 0UL;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_TICKINT_Msk | SysTick_CTRL_ENABLE_Msk;
#endif
}

clock_t clock(void) {
    return (clock_t)-1;
}

int remove(const char *arg) {
    (void)arg;
    return 0;
}

int rename(const char *oldn, const char *newn) {
    (void)oldn;
    (void)newn;
    return 0;
}

void exit(int code) {
    uart_putc((char)0x4);
    uart_putc((char)code);
    while (1) {}
}

int fputc(int ch, FILE *f) {
    (void)(f);
    return uart_putc(ch);
}

int fgetc(FILE *f) {
    (void)f;
    return uart_putc(uart_getc());
}

#ifndef ferror
/* arm-none-eabi-gcc with newlib uses a define for ferror */
int ferror(FILE *f) {
    (void)f;
    return EOF;
}
#endif

