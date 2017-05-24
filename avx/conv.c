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


    outData = aligned_alloc(32, width * height * sizeof (data_t));

    register __m256 dtop1, dmid1, dbot1, dtop2, dmid2, dbot2, dtopp, dmidp, dbotp, dtarget1, dtarget2; //Vector data storage
    register __m256 k1, k2, k3; //1-8 kernels (missing 9)
    register __m256 r1;
    register __m256i maskk;
    register data_t fk1, fk2, fk3; //Rest kernel
    register data_t f1, f2, f3; //free to use

    //start timer
    tic();

    //Init kernel
    k1 = k2 = k3 = _mm256_loadu_ps(filter); //load unalligned
    fk1 = fk2 = fk3 = filter[8]; //rest

    //Calculate rest width
    int wrest = width - (width % 8);
    int x;
    //Center cases
    for (int y = 0; y < height; y++) {
        //bootstrap
        dtop1 = dmid1 = dbot1 = _mm256_setzero_ps();
        if (y == 0)
            dtop2 = dtop1;
        else
            dtop2 = _mm256_load_ps(&inData[((y - 1) * width)]); //Support row

        dmid2 = _mm256_load_ps(&inData[(y * width)]); //target row

        if ((y+1)  < height)
            dbot2 = _mm256_load_ps(&inData[((y + 1) * width)]); //Support row
        else
            dbot2 = dtop1;

        for (x = 0; x < width - 8; x += 8) {
            register int rcount = 1;
            //Load data (shift loaded data)
            dtop1 = dtop2;
            dmid1 = dmid2;
            dbot1 = dbot2;
            if (x + 16 < width) {
                if (y != 0) dtop2 = _mm256_load_ps(&inData[((y - 1) * width) + x + 8]); //Support row next
                dmid2 = _mm256_load_ps(&inData[(y * width) + x + 8]); //target row next
                if ((y+1) < height) dbot2 = _mm256_load_ps(&inData[((y + 1) * width) + x + 8]); //Support row next
                else dbot2 = _mm256_setzero_ps();
            } else {
                maskk = (__m256i){(x + 8 < width) - 1, (x + 8 < width) - 1,
                    (x + 8 < width) - 1, (x + 8 < width) - 1, (x + 8 < width) - 1,
                    (x + 8 < width) - 1, (x + 8 < width) - 1, (x + 8 < width) - 1};
                dtop2 = _mm256_maskload_ps(&inData[((y - 1) * width) + x + 8], maskk); //Support row next
                dmid2 = _mm256_maskload_ps(&inData[(y * width) + x + 8], maskk); //target row next
                if ((y+1)  < height) dbot2 = _mm256_maskload_ps(&inData[((y + 1) * width) + x + 8], maskk); //Support row next
                else dbot2 = _mm256_setzero_ps();
            }
            //First element is special
            if ((y+1)  < height) dtarget1 = (__m256){inData[((y - 1) * width) + x - 1], dtop1[0], dtop1[1], inData[((y) * width) + x - 1], dmid1[0], dmid1[1], inData[((y + 1) * width) + x - 1], dbot1[0]};
            else dtarget1 = (__m256){inData[((y - 1) * width) + x - 1], dtop1[0], dtop1[1], inData[((y) * width) + x - 1], dmid1[0], dmid1[1], 0.0, dbot1[0]};
            f1 = dbot1[1];

            dtarget1 *= k1;
            f1 *= fk1;
            dtarget1 = _mm256_hadd_ps(dtarget1, dtarget1);
            f1 += dtarget1[0] + dtarget1[2];
            r1[0] = f1;
            //second to one before last is generic
            for (int i = 0; i < 6; i += 2) {
                //Extract 3 kernel targets
                dtarget1 = (__m256){dtop1[i + 0], dtop1[i + 1], dtop1[i + 2], dmid1[i + 0], dmid1[i + 1], dmid1[i + 2], dbot1[i + 0], dbot1[i + 1]}; //Is correctly optimized
                f1 = dbot1[i + 2];
                dtarget2 = (__m256){dtop1[i + 1], dtop1[i + 2], dtop1[i + 3], dmid1[i + 1], dmid1[i + 2], dmid1[i + 3], dbot1[i + 1], dbot1[i + 2]}; //Is correctly optimized
                f2 = dbot1[i + 3];
                //Vector products
                dtarget1 *= k1;
                dtarget2 *= k2;

                //scalar products (Seperate port from vector products)
                f1 *= fk1;
                f2 *= fk2;

                //Sum A+B, A+B, B+C, B+C
                dtarget1 = _mm256_hadd_ps(dtarget1, dtarget1);
                f1 += dtarget1[0] + dtarget1[2];
                dtarget2 = _mm256_hadd_ps(dtarget2, dtarget2);
                f2 += dtarget2[0] + dtarget2[2];

                //store results in register 'r1'
                r1[rcount++] = f1;
                r1[rcount++] = f2;
            }
            //Last element is special
            dtarget1 = (__m256){dtop1[7], dtop1[8], dtop2[0], dmid1[7], dmid1[8], dmid2[0], dbot1[7], dbot1[8]};
            f1 = dbot2[0];
            dtarget1 *= k1;
            f1 *= fk1;
            dtarget1 = _mm256_hadd_ps(dtarget1, dtarget1);
            f1 += dtarget1[0] + dtarget1[2];
            r1[8] = f1;
            //store final results
            if (x + 16 < width)
                _mm256_store_ps(&outData[(y * width) + x], r1);
            else
                _mm256_maskstore_ps(&outData[(y * width) + x], maskk, r1);
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