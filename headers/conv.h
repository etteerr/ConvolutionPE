#pragma once

#define data_t float
#define convsize_t unsigned long


/**
 * Convolves data and puts the resulting data to *data.
 *  Allocation may be done in conv
 *  stencil is the kernel, 3x3
 * @param data
 * @param stencil a 3x3 kernel
 * @return 
 */
double conv3(data_t ** data, convsize_t w, convsize_t h,  const data_t * stencil);

/**
 * Convolves data and puts the resulting data to *data.
 *  Allocation may be done in conv
 *  stencil is the kernel, 5x5
 * @param data
 * @param stencil a 5x5 kernel
 * @return 
 */
double conv5(data_t ** data, convsize_t w, convsize_t h, const data_t * stencil);