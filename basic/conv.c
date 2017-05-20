#include <stdlib.h>
#include <sys/time.h>

#include "conv.h"
#include "tictoc.h"

double conv3(float** data, unsigned long width, unsigned long height, const float* filter) {

    data_t * outData = 0, *inData = *data;


    outData = aligned_alloc(16, width * height * sizeof (data_t));

    tic();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned int filterItem = 0;
            float filterSum = 0.0f;
            float smoothPix = 0.0f;

            for (int fy = y - 1; fy < y + 1; fy++) {
                for (int fx = x - 1; fx < x + 1; fx++) {
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