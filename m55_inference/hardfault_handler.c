/*
Copyright (c) 2020 Arm Limited (or its affiliates). All rights reserved.
Use, modification and redistribution of this file is subject to your possession of a
valid End User License Agreement for the Arm Product of which these examples are part of
and your compliance with all applicable terms and conditions of such licence agreement.
*/

#include <stdio.h>
#include <stdlib.h>

/* HardFault handler implementation that prints a message
   then exits the program early.
 */
void HardFault_Handler(void)
{
    printf("HardFault occurred!\n");
    exit(1);
}

