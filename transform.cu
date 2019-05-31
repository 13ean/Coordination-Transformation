#ifndef TRANSFORM_CU
#define TRANSFORM_CU

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

// includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#include "transform_common.h"



__constant__ double c_geoTransform[6];

#ifdef __cplusplus
extern "C" {
#endif // __cpluscplus

void setGeoTransform(double *h_geoTransform)
{
    cudaMemcpyToSymbol(c_geoTransform, h_geoTransform, 6 * sizeof(double));
}


/*
__device__ __host__ inline double getSquareOfEccentricity(double inverseFlatenning) {
    return 2 * inverseFlatenning - inverseFlatenning * inverseFlatenning;
}

__device__ __host__ inline double getEpsilon(double squareOfEccentricity) {
    return squareOfEccentricity / (1 - squareOfEccentricity);
}

__device__ __host__ inline double3 geocentirc2Geographic(double3 coord, double semiMajor, double inverseFlatenning) {
    double squareOfEccentricity = getSquareOfEccentricity(inverseFlatenning);
    double epsilon = getEpsilon(squareOfEccentricity);
    //TODO
    double3 res;
    res.x = 
}
*/


__device__ inline double2 PLCoord2ProjCoord(int2 plCoord) {
    double2 res;    
    res.x = c_geoTransform[0] + plCoord.x * c_geoTransform[1] + plCoord.y * c_geoTransform[2];
    res.y = c_geoTransform[3] + plCoord.x * c_geoTransform[4] + plCoord.y * c_geoTransform[5];
    return res;
}

// Transverse Mercator
__device__ inline double2 projCoord2GeoCoord(double2 projCoord) {
    double2 res;

    return res;
}

__device__ inline double2 geoCoord2ProjCoord(double2 geoCoord) {
    double2 res;



    return res;
}

__device__ inline int2 projCoord2PLCoord(double2 projCoord) {
    double dTemp = c_geoTransform[1] * c_geoTransform[5] - c_geoTransform[2] * c_geoTransform[4];
    int2 res;
    res.x = static_cast<int>((c_geoTransform[5] * (projCoord.x - c_geoTransform[0]) - 
             c_geoTransform[2] * (projCoord.y - c_geoTransform[3])) / dTemp + 0.5);
    res.y = static_cast<int>((c_geoTransform[4] * (projCoord.x - c_geoTransform[0]) - 
             c_geoTransform[1] * (projCoord.y - c_geoTransform[3])) / dTemp + 0.5);

    return res;
}

__global__ void computeTransform2(double2 *output, int width, int height, 
    double3 geoTransformx, double3 geoTransformy) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < width && iy < height) {
        int pos = iy * width + ix;
        output[pos].x = geoTransformx.x + ix * geoTransformx.y + iy * geoTransformx.z;
        output[pos].y = geoTransformy.x + ix * geoTransformy.y + iy * geoTransformy.z;
    }
}

void transformCuda(double2 *output, int width, int height, 
    dim3 blockSize, dim3 gridSize, 
    double3 geoTransformx, double3 geoTransformy) {

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t startCUDA;
    checkCudaErrors(cudaEventCreate(&startCUDA));

    cudaEvent_t stopCUDA;
    checkCudaErrors(cudaEventCreate(&stopCUDA));

    // Record the start event
    checkCudaErrors(cudaEventRecord(startCUDA, NULL));

    computeTransform2<<<gridSize, blockSize>>>(output, width, height, geoTransformx, geoTransformy);

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stopCUDA, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stopCUDA));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, startCUDA, stopCUDA));
    printf("GPU Time: %lfms\n", msecTotal);
    getLastCudaError("kernel failed");
}

#ifdef __cplusplus
}
#endif // __cpluscplus

#endif // TRANSFORM_CU