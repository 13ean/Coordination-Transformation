#ifndef TRANSFORM_COMMON_H
#define TRANSFORM_COMMON_H

#include <cuda_runtime.h>


typedef unsigned int uint;
typedef unsigned char uchar;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

double getEpsilon(double squaredEccentricity);


double3 pl2Geographic(int2 plCoord, double *geotransform);
int2 geographic2Pl(double3 coord, double *geotransform, double temp);
double3 noneGreenwichGeographic2GreenwichGeographic(double3 coord);
double3 geographic2Geocentric(double3 coord, double semiMajor, double squaredEccentircity);
double3 geocentric2Geocentric(double3 coord, double *transParam, int nCoef = 7);
double3 geocentric2Geographic(double3 coord, double semiMajor, double semiMinor, 
                                double squaredEccentircity, double epsilon);


void coordinationTansformCPU();

void setGeoTransform(double *h_geoTransform);
void transformCuda(double2 *output, int width, int height, 
    dim3 blockSize, dim3 gridSize, 
    double3 geoTransformx, double3 geoTransformy);



#ifdef __cplusplus
}
#endif // __cplusplus


#endif // TRANSFORM_COMMON_H