#include <stdlib.h>
#include <sys/time.h>

#include <x86intrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include "conv.h"
#include "tictoc.h"


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

    data_t * outData = 0, *inData;
    
    inData = *data;


    outData = aligned_alloc(16, width * height * sizeof (data_t));
    
    register __m256 dtop, dmid, dbot, dtarget1, dtarget2, dtarget3; //Vector data storage
    register __m256 k1,k2,k3; //1-8 kernels (missing 9)
    register __m256 r1,r2,r3; //free to use
    register data_t fk1, fk2, fk3; //Rest kernel
    register data_t f1,f2,f3; //free to use
    register __m256i smask = (__m256i){0,-1,-1,-1,-1,-1,-1,0};
    
    //start timer
    tic();
    
    //Init kernel
    k1 = k2 = k3 = _mm256_loadu_ps(filter); //load unalligned
    fk1 = fk2 = fk3 = filter[8]; //rest
    
    //Center cases
    for (int y = 1; y < height-1; y++) {
        for (int x = 0; x < width-6; x+=6) {
            register int rcount = 1;
            //Load data
            dtop = _mm256_loadu_ps(&inData[((y-1)*width)+x]); //Support row
            dmid = _mm256_loadu_ps(&inData[( y   *width)+x]); //target row
            dbot = _mm256_loadu_ps(&inData[((y+1)*width)+x]); //Support row
            
            for(int i=0; i<5; i++){
                //Extract 3 kernel targets
                dtarget1 = (__m256) {dtop[i+0], dtop[i+1], dtop[i+2], dmid[i+0], dmid[i+1], dmid[i+2], dbot[i+0], dbot[i+1]}; //Is correctly optimized
                f1 = dbot[i+2];
                dtarget2 = (__m256) {dtop[i+1], dtop[i+2], dtop[i+3], dmid[i+1], dmid[i+2], dmid[i+3], dbot[i+1], dbot[i+2]}; //Is correctly optimized
                f2 = dbot[i+3];
                dtarget3 = (__m256) {dtop[i+2], dtop[i+3], dtop[i+4], dmid[i+2], dmid[i+3], dmid[i+4], dbot[i+2], dbot[i+3]}; //Is correctly optimized
                f3 = dbot[i+4];
                
                //Vector products
                dtarget1 *= k1;
                dtarget2 *= k2;
                dtarget3 *= k3;
                
                //scalar products (Seperate port from vector products)
                f1 *= fk1;
                f2 *= fk2;
                f3 *= fk3;
                
                //Sum A+B, A+B, B+C, B+C
                dtarget1 = _mm256_hadd_ps(dtarget1, dtarget1);
                f1 += dtarget1[0] + dtarget1[2];
                dtarget2 = _mm256_hadd_ps(dtarget2, dtarget2);
                f2 += dtarget2[0] + dtarget2[2];
                dtarget3 = _mm256_hadd_ps(dtarget3, dtarget3);
                f3 += dtarget3[0] + dtarget3[2];
                
                //store results in register 'r1'
                r1[rcount++] = f1;
                r1[rcount++] = f2;
                r1[rcount++] = f3;
            }
            
            //store final results
            _mm256_maskstore_ps(&inData[(y*width)+x], smask, r1);
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