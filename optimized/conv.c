#include <stdlib.h>
#include <sys/time.h>

#include "conv.h"
#include "tictoc.h"

#define get(X,Y) inData[(Y*width)+X]
#define set(X,Y,V) outData[(Y*width)+X] = V

double conv3(float** data, unsigned long width, unsigned long height, const float* filter) {

    data_t * outData = 0, *inData = *data;

    outData = aligned_alloc(32, width * height * sizeof (data_t));
    
    tic();
    
    
    register float kern __attribute__ ((__vector_size__ (32))); //register of size 8float
    for(int i = 0; i<9; i++)
        kern[i] = filter[i];
    register float kernl = filter[9];
    register float acc1,acc2,acc3;
    register float f1,f2,f3,f4,f5,f6;

    for (int y = 1; y < height-1; y++) {
        __builtin_prefetch(&get(y+1,0)); //prefetch next 4 lines (y+1 is used in this loop, should be preloaded already)
        __builtin_prefetch(&get(y+2,0));
        for (int x = 1; x < width-1; x++) {
            //No prefetching here, we expect the CPU to predict this (Linear access)
            f1 = get(x-1,y-1) * kern[0];
            f2 = get(x-1,y  ) * kern[3];
            f3 = get(x-1,y+1) * kern[6];
            
            f1 += get(x ,y-1) * kern[1];
            f2 += get(x ,y  ) * kern[4];
            f3 += get(x ,y+1) * kern[7];
            
            f1 += get(x+1,y-1) * kern[2];
            f2 += get(x+1,y  ) * kern[3];
            f3 += get(x+1,y+1) * kernl;
            
            set(x,y,f1+f2+f3);
        }
    }
    double time = toc();
    free(inData);
    *data = outData;

    return time;
}

double conv5(float** data, unsigned long width, unsigned long height, const float* filter) {

    data_t * outData = 0, *inData = *data;

    outData = aligned_alloc(32, width * height * sizeof (data_t));

    tic();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned int filterItem = 0;
            float filterSum = 0.0f;
            float smoothPix = 0.0f;

            for (int fy = y - 2; fy < y + 3; fy++) {
                for (int fx = x - 2; fx < x + 3; fx++) {
                    if (((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width))) {
                        filterItem++;
                        continue;
                    }


                    outData[(y * width) + x] += inData[(fy * width) + fx] * filter[filterItem];
                    filterSum += filter[filterItem];
                    filterItem++;

                }
            }
        }
    }
    double time = toc();
    free(inData);
    *data = outData;

    return time;
}