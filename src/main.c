#include <stdlib.h>
#include <omp.h>
#include <stdio.h>


#include "conv.h"

#define FLOPS(t,w,h,s) (double)( \
                        ((w-(s-1)/2)*(h-(s-1)/2)*s*s) + ((s*s-s*((s-1)/2))*(w+h)-4*(s-1)/2))  /  (double)t
//                      binnen

void makeDataAligned(data_t ** pdata, convsize_t width, convsize_t height) {
    convsize_t size = width * height;
    
    data_t * data = 0;
    
    data = aligned_alloc(16, size*sizeof(data_t));
    
    if (!data)
        exit(1);
    
    unsigned int seed;
    
    #pragma omp parallel default(none) shared(data, size) 
    {
        unsigned long seed = omp_get_thread_num();
        #pragma omp for private(seed)
        for(convsize_t i = 0; i<size; i++) {
            data[i] = (data_t)rand_r(&seed)/(data_t)RAND_MAX;
        }
    }
    
    *pdata = data;
}

int main(int nargs, char ** args) {
    
    if (nargs != 3 && nargs != 1) {
        printf("Invalid usage, use ./bin [width] [height]");
    }
    
    convsize_t w,h;
    h=w=1000;
    
    if (nargs == 3) {
        w = atoll(args[1]);
        h = atoll(args[2]);
    }
    
    data_t *data;
    
    makeDataAligned(&data, w, h);
    
    const data_t stencil3[] = { 
                          1.0, 1.0, 1.0,
                          1.0, 1.0/9.0, 1.0,
                          1.0, 1.0, 1.0
    };
    __builtin___clear_cache(data, data+w*h);
    double time3 = conv3(&data, w, h, stencil3);
    printf("stencil 3x3: %f seconds, %e flops\n", time3, FLOPS(time3, w, h, 3));
    
    free(data);
    makeDataAligned(&data, w, h);
    
    const data_t stencil5[] = { 
                          0.0, 0.0, 1.0, 0.0, 0.0,
                          0.0, 1.0, 2.0, 1.0, 0.0,
                          1.0, 2.0, 3.0, 2.0, 1.0,
                          0.0, 1.0, 2.0, 1.0, 0.0,
                          0.0, 0.0, 1.0, 0.0, 0.0
    };
    
    __builtin___clear_cache(data, data+w*h);
    double time5 = conv5(&data, w, h, stencil5);
    printf("stencil 5x5: %f seconds, %e flops\n", time5, FLOPS(time5, w, h, 5));
    return 0;
};