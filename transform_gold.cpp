#include "transform_common.h"
#include <time.h>
#include <stdio.h>
#include <math.h>

#include "gdal_alg.h"
#include "cpl_string.h"
#include "ogr_srs_api.h"
#include "ogr_spatialref.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus



double getEpsilon(double squaredEccentricity) {
    return squaredEccentricity / (1 - squaredEccentricity);
}


double3 noneGreenwichGeographic2GreenwichGeographic(double3 coord) {
    double3 res;
    //TODO

    return res;
}

double3 pl2Geographic(int2 plCoord, double *geotransform) {
    double3 res;
    res.x = geotransform[0] + plCoord.x * geotransform[1] + plCoord.y * geotransform[2];
    res.y = geotransform[3] + plCoord.x * geotransform[4] + plCoord.y * geotransform[5];
    res.z = 0;

    return res;
}

int2 geographic2Pl(double3 coord, double *geotransform, double dTemp) {
    int2 res;

    res.x = static_cast<int>((geotransform[5] * (coord.x - geotransform[0]) - 
                             geotransform[2] * (coord.y - geotransform[3])) / dTemp + 0.5);
    res.y = static_cast<int>((geotransform[1] * (coord.y - geotransform[3]) - 
                             geotransform[4] * (coord.x - geotransform[0])) / dTemp + 0.5);

    return res;
}

double3 geographic2Geocentric(double3 coord, double semiMajor, double squaredEccentricity) {
    double verticalRadius = semiMajor / sqrt(1 - squaredEccentricity * pow(sin(coord.x), 2));
    
    double3 res;
    res.x = (verticalRadius + coord.z) * cos(coord.x) * cos(coord.y);
    res.y = (verticalRadius + coord.z) * cos(coord.x) * sin(coord.y);
    res.z = ((1 - squaredEccentricity) * verticalRadius + coord.z) * sin(coord.x);
    return res;
}

double3 geocentric2Geographic(double3 coord, double semiMajor, double semiMinor, 
                                double squaredEccentricity, double epsilon) {
    
    double p = sqrt(coord.x * coord.x + coord.y * coord.y);
    double q = atan2(coord.z * semiMajor, p * semiMinor);

    double3 res;
    res.x = atan2(coord.z + epsilon * semiMinor * pow(sin(q), 3), p - squaredEccentricity * semiMajor * pow(cos(q), 3));
    res.y = atan2(coord.y, coord.x);
    double verticalRadius = semiMajor / sqrt(1 - squaredEccentricity * pow(sin(res.x), 2));
    res.z = p / cos(res.x) - verticalRadius;
    return res;
}

// 
double3 geocentric2Geocentric(double3 coord, double* padfCoef, int nCoef) {
    double3 res;
    if (nCoef == 7) {
        res.x = padfCoef[6] * (coord.x - padfCoef[5] * coord.y + padfCoef[4] * coord.z) + padfCoef[0];
        res.y = padfCoef[6] * (coord.x * padfCoef[5] + coord.y - padfCoef[3] * coord.z) + padfCoef[1];
        res.z = padfCoef[6] * (-coord.x * padfCoef[4] + coord.y * padfCoef[3] + coord.z) + padfCoef[2];
    }
    return res;
}

void tansformCpu(double* output_x, double *output_y, int width, int height, double *padfTransform) {
    time_t start = clock();
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; ++j) {
            int pos = i * width + j;
            output_x[pos] = padfTransform[0] + j * padfTransform[1] + i * padfTransform[2];
            output_y[pos] = padfTransform[3] + j * padfTransform[4] + i * padfTransform[5];
        }
    }
    time_t end = clock();
    double timeCPU = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU Time: %lfms\n", timeCPU);
}

#ifdef __cplusplus
}
#endif // __cplusplus