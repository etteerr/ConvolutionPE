#include <stdlib.h>
#include <sys/time.h>
#include <xmmintrin.h>

#include "conv.h"
#include "tictoc.h"
#include "iaca.h"

#define get(X,Y) inData[((Y)*width)+(X)]
#define set(X,Y,V) outData[((Y)*width)+(X)] = V

double conv3(float** data, unsigned long width, unsigned long height, const float* filter) {

    data_t * outData = 0, *inData = *data;

    outData = aligned_alloc(32, width * height * sizeof (data_t));

    tic();


    register float kern __attribute__ ((__vector_size__(32))); //register of size 8float
    for (int i = 0; i < 9; i++)
        kern[i] = filter[i];
    register float kernl = filter[9];
    register float f1, f2, f3, f4, f5, f6;

    //Edge case y==0
    for (int x = 1; x < width - 1; x++) {
        //No prefetching here, we expect the CPU to predict this (Linear access)
        f1 = get(x - 1, 0) * kern[3];
        f2 = get(x - 1, 1) * kern[6];

        f3 = get(x, 0) * kern[4];
        f1 += get(x, 1) * kern[7];

        f2 += get(x + 1, 0) * kern[3];
        f3 += get(x + 1, 1) * kernl;

        set(x, 0, f1 + f2 + f3);
    }

    //middle
    for (int y = 1; y < height - 1; y++) {
        __builtin_prefetch(&get(y + 1, 0)); //prefetch next 4 lines (y+1 is used in this loop, should be preloaded already)
        __builtin_prefetch(&get(y + 2, 0));

        //Border x==0
        f1 = get(0, y - 1) * kern[1];
        f2 = get(0, y) * kern[4];
        f3 = get(0, y + 1) * kern[7];

        f1 += get(0 + 1, y - 1) * kern[2];
        f2 += get(0 + 1, y) * kern[3];
        f3 += get(0 + 1, y + 1) * kernl;

        set(0, y, f1 + f2 + f3);

        //middle
        //        IACA_START;
        for (int x = 1; x < width - 1; x++) {
            //No prefetching here, we expect the CPU to predict this (Linear access)
            f1 = get(x - 1, y - 1) * kern[0];
            f2 = get(x - 1, y) * kern[3];
            f3 = get(x - 1, y + 1) * kern[6];

            f1 += get(x, y - 1) * kern[1];
            f2 += get(x, y) * kern[4];
            f3 += get(x, y + 1) * kern[7];

            f1 += get(x + 1, y - 1) * kern[2];
            f2 += get(x + 1, y) * kern[3];
            f3 += get(x + 1, y + 1) * kernl;

            set(x, y, f1 + f2 + f3);
        }
        //        IACA_END;
        //BOrder x==width-1
        f1 = get(width - 1 - 1, y - 1) * kern[0];
        f2 = get(width - 1 - 1, y) * kern[3];
        f3 = get(width - 1 - 1, y + 1) * kern[6];

        f1 += get(width - 1, y - 1) * kern[1];
        f2 += get(width - 1, y) * kern[4];
        f3 += get(width - 1, y + 1) * kern[7];

        set(width - 1, y, f1 + f2 + f3);
    }
    //Border case y==height-1
    for (int x = 1; x < width - 1; x++) {
        //No prefetching here, we expect the CPU to predict this (Linear access)
        f4 = get(x - 1, height - 2) * kern[0];
        f5 = get(x - 1, height - 1) * kern[3];

        f6 = get(x, height - 2) * kern[1];
        f4 += get(x, height - 1) * kern[4];

        f5 += get(x + 1, height - 2) * kern[2];
        f6 += get(x + 1, height - 1) * kern[5];

        set(x, height - 1, f4 + f5 + f6);
    }

    double time = toc();
    free(inData);
    *data = outData;

    return time;
}

double conv5(float** data, unsigned long width, unsigned long height, const float* filter) {

    data_t * outData = 0, *inData;
    inData = *data;

    outData = aligned_alloc(32, width * height * sizeof (data_t));

    tic();


    float kern[25] __attribute__ ((aligned(32)));
    for (int i = 0; i < 25; i += 5) {
        kern[i] = filter[i];
        kern[i + 1] = filter[i + 1];
        kern[i + 2] = filter[i + 2];
        kern[i + 3] = filter[i + 3];
        kern[i + 4] = filter[i + 4];
    }
    register float f1, f2, f3, f4, f5, f6;

    //Edge case y==0
    for (int x = 2; x < width - 2; x++) {
        //No prefetching here, we expect the CPU to predict this (Linear access)
        f1 = get(x - 2, 0) * kern[5 * 2 + 0];
        f2 = get(x - 1, 0) * kern[5 * 2 + 1];
        f3 = get(x, 0) * kern[5 * 2 + 2]; //middle
        f4 = get(x + 1, 0) * kern[5 * 2 + 3];
        f5 = get(x + 2, 0) * kern[5 * 2 + 4];

        f1 += get(x - 2, 1) * kern[5 * 3 + 0];
        f2 += get(x - 1, 1) * kern[5 * 3 + 1];
        f3 += get(x, 1) * kern[5 * 3 + 2];
        f4 += get(x + 1, 1) * kern[5 * 3 + 3];
        f5 += get(x + 2, 1) * kern[5 * 3 + 4];

        f1 += get(x - 2, 2) * kern[5 * 4 + 0];
        f2 += get(x - 1, 2) * kern[5 * 4 + 1];
        f3 += get(x, 2) * kern[5 * 4 + 2];
        f4 += get(x + 1, 2) * kern[5 * 4 + 3];
        f5 += get(x + 2, 2) * kern[5 * 4 + 4];

        set(x, 0, f1 + f2 + f3 + f4 + f5);
    }

    //Edge case y==1
    for (int x = 2; x < width - 2; x++) {
        //No prefetching here, we expect the CPU to predict this (Linear access)
        f1 = get(x - 2, 0) * kern[5 * 1 + 0];
        f2 = get(x - 1, 0) * kern[5 * 1 + 1];
        f3 = get(x, 0) * kern[5 * 1 + 2];
        f4 = get(x + 1, 0) * kern[5 * 1 + 3];
        f5 = get(x + 2, 0) * kern[5 * 1 + 4];

        f1 += get(x - 2, 1) * kern[5 * 2 + 0];
        f2 += get(x - 1, 1) * kern[5 * 2 + 1];
        f3 += get(x, 1) * kern[5 * 2 + 2]; //middle
        f4 += get(x + 1, 1) * kern[5 * 2 + 3];
        f5 += get(x + 2, 1) * kern[5 * 2 + 4];

        f1 += get(x - 2, 2) * kern[5 * 3 + 0];
        f2 += get(x - 1, 2) * kern[5 * 3 + 1];
        f3 += get(x, 2) * kern[5 * 3 + 2];
        f4 += get(x + 1, 2) * kern[5 * 3 + 3];
        f5 += get(x + 2, 2) * kern[5 * 3 + 4];

        f1 += get(x - 2, 3) * kern[5 * 4 + 0];
        f2 += get(x - 1, 3) * kern[5 * 4 + 1];
        f3 += get(x, 3) * kern[5 * 4 + 2];
        f4 += get(x + 1, 3) * kern[5 * 4 + 3];
        f5 += get(x + 2, 3) * kern[5 * 4 + 4];

        set(x, 1, f1 + f2 + f3 + f4 + f5);
    }

    //middle
    for (int y = 2; y < height - 2; y++) {
        __builtin_prefetch(&get(y + 1, 0)); //prefetch next 4 lines (y+1 to 2 is used in this loop, should be preloaded already)
        __builtin_prefetch(&get(y + 2, 0));
        __builtin_prefetch(&get(y + 3, 0));
        __builtin_prefetch(&get(y + 4, 0));


        //Border x==0
        f1 = get(0, y - 2) * kern[5 * 0 + 2];
        f2 = get(1, y - 2) * kern[5 * 0 + 3];
        f3 = get(2, y - 2) * kern[5 * 0 + 4];

        f4 = get(0, y - 1) * kern[5 * 1 + 2];
        f5 = get(1, y - 1) * kern[5 * 1 + 3];
        f1 += get(2, y - 1) * kern[5 * 1 + 4];

        f2 += get(0, y) * kern[5 * 2 + 2]; //middle
        f3 += get(1, y) * kern[5 * 2 + 3];
        f4 += get(2, y) * kern[5 * 2 + 4];

        f5 += get(0, y + 1) * kern[5 * 3 + 2];
        f1 += get(1, y + 1) * kern[5 * 3 + 3];
        f2 += get(2, y + 1) * kern[5 * 2 + 4];

        f3 += get(0, y + 2) * kern[5 * 4 + 2];
        f4 += get(1, y + 2) * kern[5 * 4 + 3];
        f5 += get(2, y + 2) * kern[5 * 4 + 4];


        set(0, y, f1 + f2 + f3 + f4 + f5);

        //Border x==1
        f1 = get(-1, y - 2) * kern[5 * 0 + 1];
        f2 = get(0, y - 2) * kern[5 * 0 + 2];
        f3 = get(1, y - 2) * kern[5 * 0 + 3];
        f4 = get(2, y - 2) * kern[5 * 0 + 4];

        f5 = get(-1, y - 1) * kern[5 * 1 + 1];
        f1 += get(0, y - 1) * kern[5 * 1 + 2];
        f2 += get(1, y - 1) * kern[5 * 1 + 3];
        f3 += get(2, y - 1) * kern[5 * 1 + 4];

        f4 += get(-1, y) * kern[5 * 2 + 1];
        f5 += get(0, y) * kern[5 * 2 + 2]; //middle
        f1 += get(1, y) * kern[5 * 2 + 3];
        f2 += get(2, y) * kern[5 * 2 + 4];

        f3 += get(-1, y + 1) * kern[5 * 3 + 1];
        f4 += get(0, y + 1) * kern[5 * 3 + 2];
        f5 += get(1, y + 1) * kern[5 * 3 + 3];
        f1 += get(2, y + 1) * kern[5 * 2 + 4];

        f2 += get(-1, y + 2) * kern[5 * 4 + 1];
        f3 += get(0, y + 2) * kern[5 * 4 + 2];
        f4 += get(1, y + 2) * kern[5 * 4 + 3];
        f5 += get(2, y + 2) * kern[5 * 4 + 4];


        set(1, y, f1 + f2 + f3 + f4 + f5);

        //middle
                IACA_START;
        for (int x = 1; x < width - 1; x++) {
            //No prefetching here, we expect the CPU to predict this (Linear access)
            f1 = get(x - 2, y - 2) * kern[5 * 0 + 0];
            f2 = get(x - 1, y - 2) * kern[5 * 0 + 1];
            f3 = get(x, y - 2) * kern[5 * 0 + 2];
            f4 = get(x + 1, y - 2) * kern[5 * 0 + 3];
            f5 = get(x + 2, y - 2) * kern[5 * 0 + 4];

            f1 += get(x - 2, y - 1) * kern[5 * 2 + 0];
            f2 += get(x - 1, y - 1) * kern[5 * 1 + 1];
            f3 += get(x, y - 1) * kern[5 * 1 + 2];
            f4 += get(x + 1, y - 1) * kern[5 * 1 + 3];
            f5 += get(x + 2, y - 1) * kern[5 * 1 + 4];

            f1 += get(x - 2, y) * kern[5 * 2 + 0];
            f2 += get(x - 1, y) * kern[5 * 2 + 1];
            f3 += get(x, y) * kern[5 * 2 + 2]; //middle
            f4 += get(x + 1, y) * kern[5 * 2 + 3];
            f5 += get(x + 2, y) * kern[5 * 2 + 4];

            f1 += get(x - 2, y + 1) * kern[5 * 2 + 0];
            f2 += get(x - 1, y + 1) * kern[5 * 3 + 1];
            f3 += get(x, y + 1) * kern[5 * 3 + 2];
            f4 += get(x + 1, y + 1) * kern[5 * 3 + 3];
            f5 += get(x + 2, y + 1) * kern[5 * 2 + 4];

            f1 += get(x - 2, y + 2) * kern[5 * 2 + 0];
            f2 += get(x - 1, y + 2) * kern[5 * 4 + 1];
            f3 += get(x, y + 2) * kern[5 * 4 + 2];
            f4 += get(x + 1, y + 2) * kern[5 * 4 + 3];
            f5 += get(x + 2, y + 2) * kern[5 * 4 + 4];


            set(x, y, f1 + f2 + f3 + f4 + f5);
        }
                IACA_END;

        //Border x==width-2
        f1 = get(width - 2 - 2, y - 2) * kern[5 * 0 + 0];
        f2 = get(width - 2 - 1, y - 2) * kern[5 * 0 + 1];
        f3 = get(width - 2, y - 2) * kern[5 * 0 + 2];
        f4 = get(width - 2 + 1, y - 2) * kern[5 * 0 + 3];

        f1 += get(width - 2 - 2, y - 1) * kern[5 * 2 + 0];
        f2 += get(width - 2 - 1, y - 1) * kern[5 * 1 + 1];
        f3 += get(width - 2, y - 1) * kern[5 * 1 + 2];
        f4 += get(width - 2 + 1, y - 1) * kern[5 * 1 + 3];

        f1 += get(width - 2 - 2, y) * kern[5 * 2 + 0];
        f2 += get(width - 2 - 1, y) * kern[5 * 2 + 1];
        f3 += get(width - 2, y) * kern[5 * 2 + 2]; //middle
        f4 += get(width - 2 + 1, y) * kern[5 * 2 + 3];

        f1 += get(width - 2 - 2, y + 1) * kern[5 * 2 + 0];
        f2 += get(width - 2 - 1, y + 1) * kern[5 * 3 + 1];
        f3 += get(width - 2, y + 1) * kern[5 * 3 + 2];
        f4 += get(width - 2 + 1, y + 1) * kern[5 * 3 + 3];

        f1 += get(width - 2 - 2, y + 2) * kern[5 * 2 + 0];
        f2 += get(width - 2 - 1, y + 2) * kern[5 * 4 + 1];
        f3 += get(width - 2, y + 2) * kern[5 * 4 + 2];
        f4 += get(width - 2 + 1, y + 2) * kern[5 * 4 + 3];

        set(width - 2, y, f1 + f2 + f3 + f4);

        //BOrder x==width-1
        f1 = get(width - 1 - 2, y - 2) * kern[5 * 0 + 0];
        f2 = get(width - 1 - 1, y - 2) * kern[5 * 0 + 1];
        f3 = get(width - 1, y - 2) * kern[5 * 0 + 2];

        f4 = get(width - 1 - 2, y - 1) * kern[5 * 2 + 0];
        f5 = get(width - 1 - 1, y - 1) * kern[5 * 1 + 1];
        f1 += get(width - 1, y - 1) * kern[5 * 1 + 2];

        f2 += get(width - 1 - 2, y) * kern[5 * 2 + 0];
        f3 += get(width - 1 - 1, y) * kern[5 * 2 + 1];
        f4 += get(width - 1, y) * kern[5 * 2 + 2]; //middle

        f5 += get(width - 1 - 2, y + 1) * kern[5 * 2 + 0];
        f1 += get(width - 1 - 1, y + 1) * kern[5 * 3 + 1];
        f2 += get(width - 1, y + 1) * kern[5 * 3 + 2];

        f3 += get(width - 1 - 2, y + 2) * kern[5 * 2 + 0];
        f4 += get(width - 1 - 1, y + 2) * kern[5 * 4 + 1];
        f5 += get(width - 1, y + 2) * kern[5 * 4 + 2];


        set(width - 1, y, f1 + f2 + f3 + f4 + f5);
    }
    //Border case y==height-2
    for (int x = 1; x < width - 1; x++) {
        //No prefetching here, we expect the CPU to predict this (Linear access)
        f1 = get(x - 2, height - 2 - 2) * kern[5 * 0 + 0];
        f2 = get(x - 1, height - 2 - 2) * kern[5 * 0 + 1];
        f3 = get(x, height - 2 - 2) * kern[5 * 0 + 2];
        f4 = get(x + 1, height - 2 - 2) * kern[5 * 0 + 3];
        f5 = get(x + 2, height - 2 - 2) * kern[5 * 0 + 4];

        f1 += get(x - 2, height - 2 - 1) * kern[5 * 2 + 0];
        f2 += get(x - 1, height - 2 - 1) * kern[5 * 1 + 1];
        f3 += get(x, height - 2 - 1) * kern[5 * 1 + 2];
        f4 += get(x + 1, height - 2 - 1) * kern[5 * 1 + 3];
        f5 += get(x + 2, height - 2 - 1) * kern[5 * 1 + 4];

        f1 += get(x - 2, height - 2) * kern[5 * 2 + 0];
        f2 += get(x - 1, height - 2) * kern[5 * 2 + 1];
        f3 += get(x, height - 2) * kern[5 * 2 + 2]; //middle
        f4 += get(x + 1, height - 2) * kern[5 * 2 + 3];
        f5 += get(x + 2, height - 2) * kern[5 * 2 + 4];

        f1 += get(x - 2, height - 2 + 1) * kern[5 * 2 + 0];
        f2 += get(x - 1, height - 2 + 1) * kern[5 * 3 + 1];
        f3 += get(x, height - 2 + 1) * kern[5 * 3 + 2];
        f4 += get(x + 1, height - 2 + 1) * kern[5 * 3 + 3];
        f5 += get(x + 2, height - 2 + 1) * kern[5 * 2 + 4];

        set(x, height - 2, f1 + f2 + f3 + f4 + f5);
    }
    //Border case y==height-1
    for (int x = 1; x < width - 1; x++) {
        //No prefetching here, we expect the CPU to predict this (Linear access)
        f1 = get(x - 2, height - 1 - 2) * kern[5 * 0 + 0];
        f2 = get(x - 1, height - 1 - 2) * kern[5 * 0 + 1];
        f3 = get(x, height - 1 - 2) * kern[5 * 0 + 2];
        f4 = get(x + 1, height - 1 - 2) * kern[5 * 0 + 3];
        f5 = get(x + 2, height - 1 - 2) * kern[5 * 0 + 4];

        f1 += get(x - 2, height - 1 - 1) * kern[5 * 2 + 0];
        f2 += get(x - 1, height - 1 - 1) * kern[5 * 1 + 1];
        f3 += get(x, height - 1 - 1) * kern[5 * 1 + 2];
        f4 += get(x + 1, height - 1 - 1) * kern[5 * 1 + 3];
        f5 += get(x + 2, height - 1 - 1) * kern[5 * 1 + 4];

        f1 += get(x - 2, height - 1) * kern[5 * 2 + 0];
        f2 += get(x - 1, height - 1) * kern[5 * 2 + 1];
        f3 += get(x, height - 1) * kern[5 * 2 + 2]; //middle
        f4 += get(x + 1, height - 1) * kern[5 * 2 + 3];
        f5 += get(x + 2, height - 1) * kern[5 * 2 + 4];

        set(x, height - 1, f1 + f2 + f3 + f4 + f5);
    }

    double time = toc();

    free(inData);

    *data = outData;

    return time;
}