/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tictoc.h
 * Author: Erwin Diepgrond <e.j.diepgrond@gmail.com>
 *
 * Created on May 17, 2017, 11:59 AM
 */

#ifndef TICTOC_H
#define TICTOC_H
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif
    /**
     * Starts a timer
     */
    void tic();
    /**
     * Returns time passed since tic
     * @return 
     */
    const double toc();


#ifdef __cplusplus
}
#endif

#endif /* TICTOC_H */

