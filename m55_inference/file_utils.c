#include "file_utils.h"

#ifdef INTERMEDIATE_RESULTS
static FILE *f = NULL;
extern void initialise_monitor_handles(void);

void file_utils_init(void)
{
    initialise_monitor_handles();
    f = fopen("intermediate_results_sparse.txt", "w+");
}

void file_write_array(int8_t *arr, size_t len)
{
    fprintf(f, "Length: %d", len);
    for (size_t i = 0; i < len; i++)
    {
        if (i % 16 == 0)
        {
            fprintf(f, "\n");
        }
        fprintf(f, "%4d ", arr[i]);
    }
    fprintf(f, "\n");
}

void file_close()
{
    fclose(f);
}
#else
void file_utils_init(void)
{
}
void file_write_array(int8_t *arr, size_t len)
{
}
void file_close()
{
}
#endif // INTERMEDIATE_RESULTS
