#ifndef FILE_UTILS_H
#define FILE_UTILS_H
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif
    void file_utils_init(void);
    void file_write_array(int8_t *arr, size_t len);
    void file_close(void);
#ifdef __cplusplus
}
#endif
#endif // FILE_UTILS_H
