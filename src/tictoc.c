#include "tictoc.h"

struct timeval __tic_toc_start;

/**
 * Starts a timer
 */
void tic() {
    gettimeofday(&__tic_toc_start, 0);
}
/**
 * Returns time passed since tic
 * @return 
 */
const double toc() {
    struct timeval stop;
    gettimeofday(&stop, 0);
    
    return (double)(stop.tv_sec - __tic_toc_start.tv_sec) + (double)(stop.tv_usec - __tic_toc_start.tv_usec)/1.0e6;
}