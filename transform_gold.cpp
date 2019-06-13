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


double second2Radian(double second) {
    return second / 60 / 60 / 180 * M_PI;
}

double degree2Radian(double degree) {
    return degree / 180 * M_PI;
}

double3 noneGreenwichGeographic2GreenwichGeographic(double3 coord) {
    double3 res;
    //TODO

    return res;
}

double3 pl2Geographic(int2 plCoord, const double *geotransform) {
    double3 res;
    res.x = geotransform[0] + plCoord.x * geotransform[1] + plCoord.y * geotransform[2];
    res.y = geotransform[3] + plCoord.x * geotransform[4] + plCoord.y * geotransform[5];
    res.z = 0;

    return res;
}

int2 geographic2Pl(double3 coord, const double *geotransform, const double dTemp) {
    int2 res;

    res.x = static_cast<int>((geotransform[5] * (coord.x - geotransform[0]) - 
                             geotransform[2] * (coord.y - geotransform[3])) / dTemp + 0.5);
    res.y = static_cast<int>((geotransform[1] * (coord.y - geotransform[3]) - 
                             geotransform[4] * (coord.x - geotransform[0])) / dTemp + 0.5);

    return res;
}

double3 geographic2Geocentric(double3 coord, const double semiMajor, const double squaredEccentricity) {
    double verticalRadius = semiMajor / sqrt(1 - squaredEccentricity * pow(sin(coord.x), 2));
    
    double3 res;
    res.x = (verticalRadius + coord.z) * cos(coord.x) * cos(coord.y);
    res.y = (verticalRadius + coord.z) * cos(coord.x) * sin(coord.y);
    res.z = ((1 - squaredEccentricity) * verticalRadius + coord.z) * sin(coord.x);
    return res;
}

double3 geocentric2Geographic(double3 coord, const double semiMajor, const double semiMinor, 
                                const double squaredEccentricity, const double epsilon) {
    
    double p = sqrt(coord.x * coord.x + coord.y * coord.y);
    double q = atan2(coord.z * semiMajor, p * semiMinor);

    double3 res;
    res.x = atan2(coord.z + epsilon * semiMinor * pow(sin(q), 3), p - squaredEccentricity * semiMajor * pow(cos(q), 3));
    res.y = atan2(coord.y, coord.x);
    double verticalRadius = semiMajor / sqrt(1 - squaredEccentricity * pow(sin(res.x), 2));
    res.z = p / cos(res.x) - verticalRadius;
    return res;
}

// to be identified
double3 WGS84ToGeocentric(double3 coord, const double* padfCoef, int nCoef) {
    double3 res;
    if (nCoef == 7) {
        res.x = (coord.x + padfCoef[5] * coord.y - padfCoef[4] * coord.z) / padfCoef[6] - padfCoef[0];
        res.y = (-coord.x * padfCoef[5] + coord.y + padfCoef[3] * coord.z) / padfCoef[6] - padfCoef[1];
        res.z = (coord.x * padfCoef[4] - coord.y * padfCoef[3] + coord.z) / padfCoef[6] - padfCoef[2];
    }
    return res;
}

double3 geocentric2WGS84(double3 coord, const double* padfCoef, int nCoef) {
    double3 res;
    if (nCoef == 7) {
        res.x = padfCoef[6] * (coord.x - padfCoef[5] * coord.y + padfCoef[4] * coord.z) + padfCoef[0];
        res.y = padfCoef[6] * (coord.x * padfCoef[5] + coord.y - padfCoef[3] * coord.z) + padfCoef[1];
        res.z = padfCoef[6] * (-coord.x * padfCoef[4] + coord.y * padfCoef[3] + coord.z) + padfCoef[2];
    }
    return res;
}



// from source to target by WGS84
void geoGraphic2GeoGraphicByWGS84(int2 *coord, int size,
                                const char *pszSourceSRS, const char *pszTargetSRS,
                                const double *padfSrcGeoTransform) {
    OGRSpatialReference poSourceSRS(pszSourceSRS);
    OGRSpatialReference poTargetSRS(pszTargetSRS);

    double targetSemiMajor = poTargetSRS.GetSemiMajor();
    double targetSemiMinor = poTargetSRS.GetSemiMinor();
    double targetSquaredEccentricity = poTargetSRS.GetSquaredEccentricity();
    double targetInvFlattening = poTargetSRS.GetInvFlattening();
    double targetEpsilon = getEpsilon(targetSquaredEccentricity);
    double targetOffset = poTargetSRS.GetPrimeMeridian();
    double targetAngularUnits = poTargetSRS.GetAngularUnits();
    double targetCoef[7] = {};
    double sourceSemiMajor = poSourceSRS.GetSemiMajor();
    double sourceSemiMinor = poSourceSRS.GetSemiMinor();
    double sourceSquaredEccentricity = poSourceSRS.GetSquaredEccentricity();
    double sourceInvFlattening = poSourceSRS.GetInvFlattening();
    double sourceEpsilon = getEpsilon(sourceSquaredEccentricity);
    double sourceOffset = poSourceSRS.GetPrimeMeridian();
    double sourceAngularUnits = poSourceSRS.GetAngularUnits();
    double sourceCoef[7] = {};
    if (OGRERR_NONE != poTargetSRS.GetTOWGS84(targetCoef) || OGRERR_NONE != poTargetSRS.GetTOWGS84(sourceCoef)) {
        printf("SRS TOWGS84 isn't available\n, Used as WGS84\n");
    }
    for (int i = 3; i <= 5; ++i) {
        targetCoef[i] *= targetAngularUnits / 60 /60;
        sourceCoef[i] *= sourceAngularUnits / 60 /60;
    }
    targetCoef[6] = 1 + targetCoef[6] * 1e-6;
    sourceCoef[6] = 1 + sourceCoef[6] * 1e-6;

    // convert 
    double sourceSpan = padfSrcGeoTransform[1] * padfSrcGeoTransform[5] - padfSrcGeoTransform[2] * padfSrcGeoTransform[4];
    
    
    for (int i = 0; i < size; ++i) {
        double3 res = pl2Geographic(coord[i], padfSrcGeoTransform);
        res = geographic2Geocentric(res, sourceSemiMajor, sourceSquaredEccentricity);
        res = geocentric2WGS84(res, sourceCoef);
        res = WGS84ToGeocentric(res, targetCoef);
        res = geocentric2Geographic(res, targetSemiMajor, targetSemiMinor, targetSquaredEccentricity, targetEpsilon);
        coord[i].x = res.x;
        coord[i].y = res.y;
    }
}

#ifdef __cplusplus
}
#endif // __cplusplus