#include <stdlib.h>
#include <sys/time.h>

#include <x86intrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include "conv.h"
#include "tictoc.h"

#define set 0xFFFFFFFF
#define uset 0x0

/**
 * Calculates a convolution by AVX256:
 *  FMA with data * kernel + acc
 *  HADD to sum results (https://www.codeproject.com/KB/cpp/874396/Fig1.jpg)
 * @param data
 * @param width
 * @param height
 * @param filter
 * @return 
 */
double conv3(float** data, unsigned long width, unsigned long height, const float* filter) {

    data_t * outData = 0, *inData = *data;


    outData = aligned_alloc(16, width * height * sizeof (data_t));
    
    __m256 ktop, kmid, kbot; //Kernel (Fits 8 float)
    __m256 dtop, dmid, dbot; //Data
    __m256 vacc1, vacc2;
    __m256i mask;
    tic();
    
    //Init kernel
    ktop = (__m256) { filter[0], filter[1], filter[2], filter[0], filter[1], filter[2], 0.0, 0.0} ;
    kmid = (__m256) { filter[3], filter[4], filter[5], filter[3], filter[4], filter[5], 0.0, 0.0} ;
    kbot = (__m256) { filter[6], filter[7], filter[8], filter[6], filter[7], filter[8], 0.0, 0.0} ;
    mask = (__m256i) {set,set,set,set,set,set,uset,uset};
    
    //Center cases
    for (int y = 1; y < height-1; y++) {
        for (int x = 1; x < width-1; x++) {
            dtop = _mm256_maskload_ps(inData[((y-1)*width)+x-1], mask);
            vacc1 = _mm256_fmadd_ps(dtop, ktop, vacc1);
            dmid = _mm256_maskload_ps(inData[((y+0)*width)+x-1], mask);
            vacc1 = _mm256_fmadd_ps(dmid, kmid, vacc1);
            dbot = _mm256_maskload_ps(inData[((y+1)*width)+x-1], mask);
            vacc1 = _mm256_fmadd_ps(dbot, kbot, vacc1);
            vacc2 = vacc1;
            //c = {A1+A2, A3+A4, B1+B2, B3+B4, A5+A6, A7+A8, B5+B6, B7+B8}
            //     _____         _____         _____         _____
            //_mm256_hadd_ps(acc,acc); //Does not surve much purpose
            outData[(y*width)+x] = vacc1[0] + vacc1[1] + vacc1[3];
            outData[(y*width)+x+3] = vacc2[4] + vacc2[5] + vacc2[6];
        }
    }
    double time = toc();
    free(inData);
    *data = outData;

    return time;
}

double conv5(float** data, unsigned long width, unsigned long height, const float* filter) {

    data_t * outData = 0, *inData = *data;

    outData = aligned_alloc(16, width * height * sizeof (data_t));

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