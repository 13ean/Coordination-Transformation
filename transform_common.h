#ifndef TRANSFORM_COMMON_H
#define TRANSFORM_COMMON_H

#include <cuda_runtime.h>


#define M_PI 3.14159265358979323846

enum ResampleMode{MODE_NEAREST, MODE_BILINEAR, MODE_BICUBIC, MODE_FAST_BICUBIC};


typedef unsigned int uint;
typedef unsigned char uchar;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

double getEpsilon(double squaredEccentricity);


double3 pl2Geographic(int2 plCoord, const double *geotransform);
int2 geographic2Pl(double3 coord, const double *geotransform, const double temp);
double3 noneGreenwichGeographic2GreenwichGeographic(double3 coord);
double3 geographic2Geocentric(double3 coord, const double semiMajor, const double squaredEccentircity);
double3 geocentric2WGS84(double3 coord, const double *transParam, int nCoef = 7);
double3 WGS84ToGeocentric(double3 coord, const double* padfCoef, int nCoef = 7);
double3 geocentric2Geographic(double3 coord, const double semiMajor, const double semiMinor, 
                                const double squaredEccentircity, const double epsilon);
void geoGraphic2GeoGraphicByWGS84(int2 *coord, int size,
                                const char *pszSourceSRS, const char *pszTargetSRS,
                                const double *padfSrcGeoTransform);

void coordinationTansformCPU();

void setGeoTransform(double *h_srcGeoTransform, double *h_dstGeoTransform);
void transformCuda(double2 *output, int width, int height, 
    dim3 blockSize, dim3 gridSize, 
    double3 geoTransformx, double3 geoTransformy);


void initTexture(int imageWidth, int imageHeight, uchar *h_data);
void unbindTexture();
void freeTexture();
void initConstant(double *h_srcGeoTransform, double *h_srcToWGS84, double *h_srcDatum,
 					double *h_dstGeoTransform, double *h_dstToWGS84, double *h_dstDatum);
void copyBackConstant(double *h_srcGeoTransform, double *h_srcToWGS84, double *h_srcDatum,
 						double *h_dstGeoTransform, double *h_dstToWGS84, double *h_dstDatum);
void initParameter(double *h_srcGeoTransform, double *h_srcToWGS84, double *h_srcDatum,
 						double *h_dstGeoTransform, double *h_dstToWGS84, double *h_dstDatum,
						double *d_srcGeoTransform, double *d_srcToWGS84, double *d_srcDatum,
 						double *d_dstGeoTransform, double *d_dstToWGS84, double *d_dstDatum);
void freeParameter(double *d_srcGeoTransform, double *d_srcToWGS84, double *d_srcDatum,
 						double *d_dstGeoTransform, double *d_dstToWGS84, double *d_dstDatum);
void transformGPU(int width, int height, int2 *coord,
                        dim3 blockSize, dim3 gridSize);
void transformGPUTest(int width, int height, int2 *coord,
                        dim3 blockSize, dim3 gridSize);
//void transformGPU(int width, int height, int2 *coord,
//                        dim3 blockSize, dim3 gridSize,
//						double *srcGeoTransform, double *srcToWGS84, double *srcDatum,
//						double *dstGeoTransform, double *dstToWGS84, double *dstDatum);
void render(int width, int height, float tx, float ty, float scale, float cx, float cy,
             dim3 blockSize, dim3 gridSize, int filter_mode, uchar *output, int2 *coord);


#ifdef __cplusplus
}
#endif // __cplusplus


#endif // TRANSFORM_COMMON_H
