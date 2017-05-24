#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>


#include "conv.h"

#define FLOPS(t,w,h,s) (double)( \
                        ((w-(s-1)/2)*(h-(s-1)/2)*s*s) + ((s*s-s*((s-1)/2))*(w+h)-4*(s-1)/2))  /  (double)t
//                      binnen

void makeDataAligned(data_t ** pdata, convsize_t width, convsize_t height) {
    convsize_t size = width * height;
    
    data_t * data = 0;
    
    data = aligned_alloc(32, size*sizeof(data_t));
    
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
    
    if (nargs != 3 && nargs != 1 && nargs != 4) {
        printf("Invalid usage, use ./bin [width] [height]");
    }
    
    convsize_t w,h, times;
    h=w=1000;
    times = 1;
    if (nargs == 3) {
        w = atoll(args[1]);
        h = atoll(args[2]);
    }
    if (nargs == 4) {
        w = atoll(args[1]);
        h = atoll(args[2]);
        times = atoll(args[3]);
    }
    
    data_t *data;
    FILE *f;
    f = fopen("report.csv", "a");
    
    if (f==0) {
        printf("Failed to open report.csv\n");
        exit(1);
    }
    for(int i = 0; i<times; i++) {
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
        
        free(data);
       

        if (ftell(f)==0) {
            fprintf(f, "exe; width; height; time3; time5; flops3; flops5\n");
        }
        
        fprintf(f,"%s; %zu; %zu; %e; %e; %e; %e\n", args[0], w, h, time3, time5, FLOPS(time3,w,h,5), FLOPS(time5,w,h,5));
        
    }
    fclose(f);
    return 0;
};