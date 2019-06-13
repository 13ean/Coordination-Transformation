#ifndef TRANSFORM_KERNEL_CUH
#define TRANSFORM_KERNEL_CUH

#include "transform_common.h"


__constant__ double c_srcGeoTransform[6];
__constant__ double c_srcToWGS84[7];
__constant__ double c_srcDatum[4]; // semiMajor, semiMinor, squared eccentricity, epsilon
__constant__ double c_dstGeoTransform[6];
__constant__ double c_dstToWGS84[7];
__constant__ double c_dstDatum[4];

texture<uchar, 2, cudaReadModeNormalizedFloat> tex;
texture<uchar, 2, cudaReadModeElementType> tex2;    // need to use cudaReadModeElementType for tex2Dgather



// source transform pixel/line coordinate to projCoord
__device__ inline double3 srcPLCoord2ProjCoord(int2 plCoord) {
    double3 res;    
    res.x = c_srcGeoTransform[0] + plCoord.x * c_srcGeoTransform[1] + plCoord.y * c_srcGeoTransform[2];
    res.y = c_srcGeoTransform[3] + plCoord.x * c_srcGeoTransform[4] + plCoord.y * c_srcGeoTransform[5];
    res.z = 0.0;
    return res;
}

// tatget transform pixel/line coordinate to projCoord
__device__ inline double3 dstPLCoord2ProjCoord(int2 plCoord) {
    double3 res;    
    res.x = c_dstGeoTransform[0] + plCoord.x * c_dstGeoTransform[1] + plCoord.y * c_dstGeoTransform[2];
    res.y = c_dstGeoTransform[3] + plCoord.x * c_dstGeoTransform[4] + plCoord.y * c_dstGeoTransform[5];
    res.z = 0.0;
    return res;
}

// source transform projection coordinate to geographic coordinate
__device__ inline double3 srcProjCoord2GeoCoord(double3 projCoord) {
    double3 res = make_double3(0.0, 0.0, 0.0);
    // TODO

    return res;
}

// target transform projection coordinate to geographic coordinate
__device__ inline double3 dstProjCoord2GeoCoord(double3 projCoord) {
    double3 res = make_double3(0.0, 0.0, 0.0);
    // TODO

    return res;
}


// source transform geographic coordinate to geocentric coordinate
 __device__ inline double3 srcGeoGraphic2Geocentric(double3 coord)
{
    double3 res;
    double verticalRadius = c_srcDatum[0] / sqrt(1 - c_srcDatum[2] * pow(sin(coord.x), 2));
    res.x = (verticalRadius + coord.z) * cos(coord.x) * cos(coord.y);
    res.y = (verticalRadius + coord.z) * cos(coord.x) * sin(coord.y);
    res.z = ((1 - c_srcDatum[2]) * verticalRadius + coord.z) * sin(coord.x);
    return res;
}

// target transform geographic coordinate to geocentric coordinate
 __device__ inline double3 dstGeoGraphic2Geocentric(double3 coord)
{
    double3 res;
    double verticalRadius = c_dstDatum[0] / sqrt(1 - c_dstDatum[2] * pow(sin(coord.x), 2));
    res.x = (verticalRadius + coord.z) * cos(coord.x) * cos(coord.y);
    res.y = (verticalRadius + coord.z) * cos(coord.x) * sin(coord.y);
    res.z = ((1 - c_dstDatum[2]) * verticalRadius + coord.z) * sin(coord.x);
    return res;
}

// source transform geocentric coordinate to WGS84 coordinate
 __device__ inline double3 srcGeocentric2WGS84(double3 coord)
{
    double3 res;
    res.x = c_srcToWGS84[6] * (coord.x - c_srcToWGS84[5] * coord.y + c_srcToWGS84[4] * coord.z) + c_srcToWGS84[0];
    res.y = c_srcToWGS84[6] * (coord.x * c_srcToWGS84[5] + coord.y - c_srcToWGS84[3] * coord.z) + c_srcToWGS84[1];
    res.z = c_srcToWGS84[6] * (-coord.x * c_srcToWGS84[4] + coord.y * c_srcToWGS84[3] + coord.z) + c_srcToWGS84[2];
    return res;
}

// target transform geocentric coordinate to WGS84 coordinate
__device__ inline double3 dstGeocentric2WGS84(double3 coord)
{
    double3 res;
    res.x = c_dstToWGS84[6] * (coord.x - c_dstToWGS84[5] * coord.y + c_dstToWGS84[4] * coord.z) + c_dstToWGS84[0];
    res.y = c_dstToWGS84[6] * (coord.x * c_dstToWGS84[5] + coord.y - c_dstToWGS84[3] * coord.z) + c_dstToWGS84[1];
    res.z = c_dstToWGS84[6] * (-coord.x * c_dstToWGS84[4] + coord.y * c_dstToWGS84[3] + coord.z) + c_dstToWGS84[2];
    return res;
}

// source transform WGS84 coordinate to geocentric coordinate
__device__ inline double3 srcWGS84ToGeocentirc(double3 coord)
{
    double3 res;
    
    res.x = (coord.x + c_srcToWGS84[5] * coord.y - c_srcToWGS84[4] * coord.z) / c_srcToWGS84[6] - c_srcToWGS84[0];
    res.y = (-coord.x * c_srcToWGS84[5] + coord.y + c_srcToWGS84[3] * coord.z) / c_srcToWGS84[6] - c_srcToWGS84[1];
    res.z = (coord.x * c_srcToWGS84[4] - coord.y * c_srcToWGS84[3] + coord.z) / c_srcToWGS84[6] - c_srcToWGS84[2];

    return res;
}

// target transform WGS84 coordinate to geocentric coordinate
__device__ inline double3 dstWGS84ToGeocentirc(double3 coord)
{
    double3 res;
    
    res.x = (coord.x + c_dstToWGS84[5] * coord.y - c_dstToWGS84[4] * coord.z) / c_dstToWGS84[6] - c_dstToWGS84[0];
    res.y = (-coord.x * c_dstToWGS84[5] + coord.y + c_dstToWGS84[3] * coord.z) / c_dstToWGS84[6] - c_dstToWGS84[1];
    res.z = (coord.x * c_dstToWGS84[4] - coord.y * c_dstToWGS84[3] + coord.z) / c_dstToWGS84[6] - c_dstToWGS84[2];

    return res;
}

// srouce transform geocentric coordinate to geographic coordinate
__device__ inline double3 srcGeocentric2Geographic(double3 coord)
{
    double p = sqrt(coord.x * coord.x + coord.y * coord.y);
    double q = atan2(coord.z * c_srcDatum[0], p * c_srcDatum[1]);

    double3 res;
    res.x = atan2(coord.z + c_srcDatum[3] * c_srcDatum[1] * pow(sin(q), 3), p - c_srcDatum[2] * c_srcDatum[0] * pow(cos(q), 3));
    res.y = atan2(coord.y, coord.x);
    double verticalRadius = c_srcDatum[0] / sqrt(1 - c_srcDatum[2] * pow(sin(res.x), 2));
    res.z = p / cos(res.x) - verticalRadius;
    return res;
}

// target transform geocentric coordinate to geographic coordinate
__device__ inline double3 dstGeocentric2Geographic(double3 coord)
{
    double p = sqrt(coord.x * coord.x + coord.y * coord.y);
    double q = atan2(coord.z * c_dstDatum[0], p * c_dstDatum[1]);

    double3 res;
    res.x = atan2(coord.z + c_dstDatum[3] * c_dstDatum[1] * pow(sin(q), 3), p - c_dstDatum[2] * c_dstDatum[0] * pow(cos(q), 3));
    res.y = atan2(coord.y, coord.x);
    double verticalRadius = c_dstDatum[0] / sqrt(1 - c_dstDatum[2] * pow(sin(res.x), 2));
    res.z = p / cos(res.x) - verticalRadius;
    return res;
}

// source transform geographic coordinate to projection coordinate
__device__ inline double3 srcGeoCoord2ProjCoord(double3 geoCoord) {
    double3 res = make_double3(0.0, 0.0, 0.0);
    //TODO


    return res;
}


// target transform geographic coordinate to projection coordinate
__device__ inline double3 dstGeoCoord2ProjCoord(double3 geoCoord) {
    double3 res = make_double3(0.0, 0.0, 0.0);
    //TODO


    return res;
}

// source projection coordinate to pixel/line coordinate
__device__ inline int2 srcProjCoord2PLCoord(double3 projCoord) {
    double dTemp = c_srcGeoTransform[1] * c_srcGeoTransform[5] - c_srcGeoTransform[2] * c_srcGeoTransform[4];
    int2 res;
    res.x = static_cast<int>((c_srcGeoTransform[5] * (projCoord.x - c_srcGeoTransform[0]) - 
                c_srcGeoTransform[2] * (projCoord.y - c_srcGeoTransform[3])) / dTemp + 0.5);
    res.y = static_cast<int>((c_srcGeoTransform[1] * (projCoord.y - c_srcGeoTransform[3]) - 
                c_srcGeoTransform[4] * (projCoord.x - c_srcGeoTransform[0])) / dTemp + 0.5);

    return res;
}

// target projection coordinate to pixel/line coordinate
__device__ inline int2 dstProjCoord2PLCoord(double3 projCoord) {
    double dTemp = c_dstGeoTransform[1] * c_dstGeoTransform[5] - c_dstGeoTransform[2] * c_dstGeoTransform[4];
    int2 res;
    res.x = static_cast<int>((c_dstGeoTransform[5] * (projCoord.x - c_dstGeoTransform[0]) - 
                c_dstGeoTransform[2] * (projCoord.y - c_dstGeoTransform[3])) / dTemp + 0.5);
    res.y = static_cast<int>(c_dstGeoTransform[1] * (projCoord.y - c_dstGeoTransform[3]) - 
                (c_dstGeoTransform[4] * (projCoord.x - c_dstGeoTransform[0])) / dTemp + 0.5);

    return res;
}

//resample

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
    //    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
float w1(float a)
{
    //    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
    //    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

__device__ float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

// filter 4 values using cubic splines
template<class T>
__device__
T cubicFilter(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

// slow but precise bicubic lookup using 16 texture lookups
template<class T, class R>  // texture data type, return type
__device__
R tex2DBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter<R>(fy,
                          cubicFilter<R>(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
                         );
}

// fast bicubic texture lookup using 4 bilinear lookups
// assumes texture is set to non-normalized coordinates, point sampling
template<class T, class R>  // texture data type, return type
__device__
R tex2DFastBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    R r = g0(fy) * (g0x * tex2D(texref, px + h0x, py + h0y)   +
                    g1x * tex2D(texref, px + h1x, py + h0y)) +
          g1(fy) * (g0x * tex2D(texref, px + h0x, py + h1y)   +
                    g1x * tex2D(texref, px + h1x, py + h1y));
    return r;
}

// higher-precision 2D bilinear lookup
template<class T, class R>  // texture data type, return type
__device__
R tex2DBilinear(const texture<T, 2, cudaReadModeNormalizedFloat> tex, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floorf(x);   // integer position
    float py = floorf(y);
    float fx = x - px;      // fractional position
    float fy = y - py;
    px += 0.5f;
    py += 0.5f;

    return lerp(lerp(tex2D(tex, px, py),        tex2D(tex, px + 1.0f, py), fx),
                lerp(tex2D(tex, px, py + 1.0f), tex2D(tex, px + 1.0f, py + 1.0f), fx), fy);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200

/*
    bilinear 2D texture lookup using tex2Dgather function
    - tex2Dgather() returns the four neighbouring samples in a single texture lookup
    - it is only supported on the Fermi architecture
    - you can select which component to fetch using the "comp" parameter
    - it can be used to efficiently implement custom texture filters

    The samples are returned in a 4-vector in the following order:
    x: (0, 1)
    y: (1, 1)
    z: (1, 0)
    w: (0, 0)
*/

template<class T, class R>  // texture data type, return type
__device__
float tex2DBilinearGather(const texture<T, 2, cudaReadModeElementType> texref, float x, float y, int comp=0)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floorf(x);   // integer position
    float py = floorf(y);
    float fx = x - px;      // fractional position
    float fy = y - py;

    R samples = tex2Dgather(texref, px + 0.5f, py + 0.5f, comp);

    return lerp(lerp((float) samples.w, (float) samples.z, fx),
                lerp((float) samples.x, (float) samples.y, fx), fy);
}

#endif


// transorm coordinate without projection coordinate
__global__ void transformNoProjGPU(int2 *output, int width, int height)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < width && iy < height) {
        int pos = iy * width + ix;
        double3 coord = dstPLCoord2ProjCoord(make_int2(ix, iy));
		coord.x *= M_PI / 180;
		coord.y *= M_PI / 180;
        coord = dstGeoGraphic2Geocentric(coord);
        coord = dstGeocentric2WGS84(coord);
        coord = srcWGS84ToGeocentirc(coord);
        coord = srcGeocentric2Geographic(coord);
		coord.x /= M_PI / 180;
		coord.y /= M_PI / 180;
        output[pos] = srcProjCoord2PLCoord(coord);
    }
}

__global__ void transformTest(int2 *output, int width, int height)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < width && iy < height) {
        int pos = iy * width + ix;
        double3 coord = dstPLCoord2ProjCoord(make_int2(ix, iy));
		coord.x *= M_PI / 180;
		coord.y *= M_PI / 180;
        coord = dstGeoGraphic2Geocentric(coord);
        coord = dstGeocentric2WGS84(coord);
        coord = srcWGS84ToGeocentirc(coord);
        coord = srcGeocentric2Geographic(coord);
		coord.x /= M_PI / 180;
		coord.y /= M_PI / 180;
        output[pos] = srcProjCoord2PLCoord(coord);
    }
}

/*
// transform coordinate with projection coordinate
__global__ void transformWithProjGPU(int2 *output, int width, int height)
{
    // TODO
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < width && iy < height) {
        int pos = iy * width + ix;
        double3 coord = PLCoord2ProjCoord(make_int2(ix, iy));
        coord = geoGraphic2Geocentric(coord);
        coord = geocentric2WGS84(coord);
        coord = WGS84ToGeocentirc(coord);
        coord = geocentric2Geographic(coord);
        output[pos] = projCoord2PLCoord(coord);
    }
}
*/

// render image using normal bilinear texture lookup
__global__ void
d_render(uchar *d_output, int2 *coord, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = coord[i].x;
    float v = coord[i].y;

    if ((x < width) && (y < height))
    {
        // write output color
        float c = tex2D(tex, u, v);
        //float c = tex2DBilinear<uchar, float>(tex, u, v);
        //float c = tex2DBilinearGather<uchar, uchar4>(tex2, u, v, 0) / 255.0f;
        d_output[i] = static_cast<uchar>(c * 0xff);
    }
}

// render image using bicubic texture lookup
__global__ void
d_renderBicubic(uchar *d_output, int2 *coord, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = coord[i].x;
    float v = coord[i].y;

    if ((x < width) && (y < height))
    {
        // write output color
        float c = tex2DBicubic<uchar, float>(tex, u, v);
        d_output[i] = static_cast<uchar>(c * 0xff);
    }
}

// render image using fast bicubic texture lookup
__global__ void
d_renderFastBicubic(uchar *d_output, int2 *coord, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = coord[i].x;
    float v = coord[i].y;

    if ((x < width) && (y < height))
    {
        // write output color
        float c = tex2DFastBicubic<uchar, float>(tex, u, v);
        d_output[i] = static_cast<uchar>(c * 0xff);
    }
}


#endif // TRANSFORM_KERNEL_CUH
