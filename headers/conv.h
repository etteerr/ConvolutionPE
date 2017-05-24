/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   conv.h
 * Author: Erwin Diepgrond <e.j.diepgrond@gmail.com>
 *
 * Created on May 24, 2017, 6:04 PM
 */

#ifndef CONV_H
#define CONV_H

#ifdef __cplusplus
extern "C" {
#endif

#define data_t float
#define convsize_t unsigned long


    /**
     * Convolves data and puts the resulting data to *data.
     *  Allocation may be done in conv
     *  stencil is the kernel, 3x3
     * @param data
     * @param stencil a 3x3 kernel
     * @return time taken in seconds
     */
    double conv3(data_t ** data, convsize_t w, convsize_t h, const data_t * stencil);

    /**
     * Convolves data and puts the resulting data to *data.
     *  Allocation may be done in conv
     *  stencil is the kernel, 5x5
     * @param data
     * @param stencil a 5x5 kernel
     * @return time taken in seconds
     */
    double conv5(data_t ** data, convsize_t w, convsize_t h, const data_t * stencil);


#ifdef __cplusplus
}
#endif

#endif /* CONV_H */

