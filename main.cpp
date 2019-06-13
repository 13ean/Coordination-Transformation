/******************************************************************************
 *
 * Project:  Mapinfo Image Warper
 * Purpose:  Commandline program for doing a variety of image warps, including
 *           image reprojection.
 * Author:   Frank Warmerdam <warmerdam@pobox.com>
 *
 ******************************************************************************
 * Copyright (c) 2002, i3 - information integration and imaging
 *                          Fort Collin, CO
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ****************************************************************************/

#include "gdal_alg.h"
#include "cpl_string.h"
#include "ogr_srs_api.h"
#include "ogr_spatialref.h"
#include <time.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>


#include "transform_common.h"

#define ET 1e-10


CPL_CVSID("$Id$")

static GDALDatasetH
GDALWarpCreateOutput( GDALDatasetH hSrcDS, const char *pszFilename,
                      const char *pszFormat, const char *pszSourceSRS,
                      const char *pszTargetSRS, int nOrder,
                      char **papszCreateOptions );

bool
SuggestedWarpOutput( GDALDatasetH hSrcDS, const char* pszSourceSRS, const char* pszTargetSRS,
                     double *adfDstGeoTransform, int *nPixels, int *nLines );
void loadSourceDS();

static double          dfMinX=0.0, dfMinY=0.0, dfMaxX=0.0, dfMaxY=0.0;
static double          dfXRes=0.0, dfYRes=0.0;
static int             nForcePixels=0, nForceLines=0;

/************************************************************************/
/*                               Usage()                                */
/************************************************************************/

static void Usage()

{
    printf(
        "Usage: warp [--version] [--formats]\n"
        "    [-s_srs srs_def] [-t_srs srs_def] [-order n] [-et err_threshold]\n"
        "    [-te xmin ymin xmax ymax] [-tr xres yres] [-ts width height]\n"
        "    [-of format] [-co \"NAME=VALUE\"]* [-gpu] srcfile dstfile\n" );
    exit( 1 );
}

/************************************************************************/
/*                             SanitizeSRS                              */
/************************************************************************/

char *SanitizeSRS( const char *pszUserInput )

{
    OGRSpatialReference hSRS;
    char *pszResult = NULL;

    CPLErrorReset();

    //hSRS = OSRNewSpatialReference( NULL );
    if( hSRS.SetFromUserInput( pszUserInput ) == OGRERR_NONE ) {
        hSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        hSRS.exportToWkt( &pszResult );
    }
    else
    {
        CPLError( CE_Failure, CPLE_AppDefined,
                  "Translating source or target SRS failed:\n%s",
                  pszUserInput );
        exit( 1 );
    }

    return pszResult;
}

/************************************************************************/
/*                                main()                                */
/************************************************************************/

int main( int argc, char ** argv )

{
    GDALDatasetH       hSrcDS, hDstDS;
    const char         *pszFormat = "GTiff";
    char               *pszTargetSRS = NULL;
    char               *pszSourceSRS = NULL;
    const char         *pszSrcFilename = NULL, *pszDstFilename = NULL;
    int                 bCreateOutput = FALSE, i, nOrder = 0;
    void               *hTransformArg, *hGenImgProjArg=NULL, *hApproxArg=NULL;
    char               **papszWarpOptions = NULL;
    double             dfErrorThreshold = 0.125;
    GDALTransformerFunc pfnTransformer = NULL;
    char                **papszCreateOptions = NULL;
	bool				useGPU = false;
	ResampleMode		mode = MODE_NEAREST;

    GDALAllRegister();

/* -------------------------------------------------------------------- */
/*      Parse arguments.                                                */
/* -------------------------------------------------------------------- */
    for( i = 1; i < argc; i++ )
    {
        if( EQUAL(argv[i],"--version") )
        {
            printf( "%s\n", GDALVersionInfo( "--version" ) );
            exit( 0 );
        }
        else if( EQUAL(argv[i],"--formats") )
        {
            int iDr;

            printf( "Supported Formats:\n" );
            for( iDr = 0; iDr < GDALGetDriverCount(); iDr++ )
            {
                GDALDriverH hDriver = GDALGetDriver(iDr);

                printf( "  %s: %s\n",
                        GDALGetDriverShortName( hDriver ),
                        GDALGetDriverLongName( hDriver ) );
            }

            exit( 0 );
        }
        else if( EQUAL(argv[i],"-co") && i < argc-1 )
        {
            papszCreateOptions = CSLAddString( papszCreateOptions, argv[++i] );
            bCreateOutput = TRUE;
        }
        else if( EQUAL(argv[i],"-of") && i < argc-1 )
        {
            pszFormat = argv[++i];
            bCreateOutput = TRUE;
        }
        else if( EQUAL(argv[i],"-t_srs") && i < argc-1 )
        {
            pszTargetSRS = SanitizeSRS(argv[++i]);
        }
        else if( EQUAL(argv[i],"-s_srs") && i < argc-1 )
        {
            pszSourceSRS = SanitizeSRS(argv[++i]);
        }
        else if( EQUAL(argv[i],"-order") && i < argc-1 )
        {
            nOrder = atoi(argv[++i]);
        }
        else if( EQUAL(argv[i],"-et") && i < argc-1 )
        {
            dfErrorThreshold = CPLAtof(argv[++i]);
        }
        else if( EQUAL(argv[i],"-tr") && i < argc-2 )
        {
            dfXRes = CPLAtof(argv[++i]);
            dfYRes = fabs(CPLAtof(argv[++i]));
            bCreateOutput = TRUE;
        }
        else if( EQUAL(argv[i],"-ts") && i < argc-2 )
        {
            nForcePixels = atoi(argv[++i]);
            nForceLines = atoi(argv[++i]);
            bCreateOutput = TRUE;
        }
        else if( EQUAL(argv[i],"-te") && i < argc-4 )
        {
            dfMinX = CPLAtof(argv[++i]);
            dfMinY = CPLAtof(argv[++i]);
            dfMaxX = CPLAtof(argv[++i]);
            dfMaxY = CPLAtof(argv[++i]);
            bCreateOutput = TRUE;
        }
        else if (EQUAL(argv[i],"-r") && i < argc-1)
        {
            if ( EQUAL(argv[++i], "near") )
                mode = MODE_NEAREST;
            else if ( EQUAL(argv[i], "bilinear") )
                mode = MODE_BILINEAR;
            else if ( EQUAL(argv[i], "cubic") )
                mode = MODE_BICUBIC;
        }
		else if (EQUAL(argv[i],"-gpu"))
		{
			useGPU = true;
		}
        else if( argv[i][0] == '-' )
            Usage();
        else if( pszSrcFilename == NULL )
            pszSrcFilename = argv[i];
		else if(pszDstFilename == NULL)
			pszDstFilename = argv[i];
        else
            Usage();
    }

    if( pszDstFilename == NULL )
        Usage();


/* -------------------------------------------------------------------- */
/*      Open source dataset.                                            */
/* -------------------------------------------------------------------- */

    hSrcDS = GDALOpen( pszSrcFilename, GA_ReadOnly );

    if( hSrcDS == NULL ) {
        fprintf(stderr, "Input file %s can't be opened.\n", pszSrcFilename);
        exit( 2 );
    }

/* -------------------------------------------------------------------- */
/*      Check that there's at least one raster band                     */
/* -------------------------------------------------------------------- */
    if ( GDALGetRasterCount( hSrcDS ) == 0 )
    {
        fprintf(stderr, "Input file %s has no raster bands.\n", pszSrcFilename );
        exit( 2 );
    }

    if( pszSourceSRS == NULL )
    {
        if( GDALGetProjectionRef( hSrcDS ) != NULL
            && strlen(GDALGetProjectionRef( hSrcDS )) > 0 )
            pszSourceSRS = CPLStrdup(GDALGetProjectionRef(hSrcDS));

        else if( GDALGetGCPProjection( hSrcDS ) != NULL
                 && strlen(GDALGetGCPProjection( hSrcDS )) > 0
                 && GDALGetGCPCount( hSrcDS ) > 1 )
            pszSourceSRS = CPLStrdup(GDALGetGCPProjection( hSrcDS ));
        else
            pszSourceSRS = CPLStrdup("");
    }

    if( pszTargetSRS == NULL )
        pszTargetSRS = CPLStrdup(pszSourceSRS);
    
    // print source/destination srs
    //printf("Source SRS: %s\n", pszSourceSRS);
    //printf("Target SRS: %s\n", pszTargetSRS);

       
    


/* -------------------------------------------------------------------- */
/*      Load the source image band(s).                                  */
/* -------------------------------------------------------------------- */
    const int nSrcXSize = GDALGetRasterXSize(hSrcDS);
    const int nSrcYSize = GDALGetRasterYSize(hSrcDS);
    const int nBandCount = GDALGetRasterCount( hSrcDS );
    uchar **papabySrcData = (uchar **)malloc(nBandCount * sizeof(uchar *));
    if (papabySrcData == NULL) {
        fprintf(stderr, "warp out of memory.\n");
        exit(1);
    }
    for (int iBand = 0; iBand < nBandCount; ++iBand)
    {
        papabySrcData[iBand] = (uchar*)malloc(nSrcXSize * nSrcYSize * sizeof(uchar));
        if (papabySrcData[iBand] == NULL)
        {
            fprintf(stderr, "warp out of memeory.\n");
            exit(1);
        }
        if (CE_None != GDALRasterIO(
                GDALGetRasterBand(hSrcDS, iBand+1), GF_Read,
                0, 0, nSrcXSize, nSrcYSize,
                papabySrcData[iBand], nSrcXSize, nSrcYSize, GDT_Byte,
                0, 0 )) 
        {
            CPLError( CE_Failure, CPLE_FileIO,
                      "warp GDALRasterIO failure %s",
                      CPLGetLastErrorMsg() );
            exit(2);
        }
    }
    


/* -------------------------------------------------------------------- */
/*      Does the output dataset already exist?                          */
/* -------------------------------------------------------------------- */

    CPLPushErrorHandler( CPLQuietErrorHandler );
    hDstDS = GDALOpen( pszDstFilename, GA_Update );
    CPLPopErrorHandler();

    if( hDstDS != NULL && bCreateOutput )
    {
        fprintf( stderr,
                 "Output dataset %s exists,\n"
                 "but some commandline options were provided indicating a new dataset\n"
                 "should be created.  Please delete existing dataset and run again.",
                 pszDstFilename );
        exit( 1 );
    }

/* -------------------------------------------------------------------- */
/*      If not, we need to create it.                                   */
/* -------------------------------------------------------------------- */

    if( hDstDS == NULL )
    {
        hDstDS = GDALWarpCreateOutput( hSrcDS, pszDstFilename, pszFormat,
                                       pszSourceSRS, pszTargetSRS, nOrder,
                                       papszCreateOptions );
        papszWarpOptions = CSLSetNameValue( papszWarpOptions, "INIT", "0" );
        CSLDestroy( papszCreateOptions );
        papszCreateOptions = NULL;
    }

    if( hDstDS == NULL )
        exit( 1 );




/* -------------------------------------------------------------------- */
/*      Create a transformation object from the source to               */
/*      destination coordinate system.                                  */
/* -------------------------------------------------------------------- */
/*
    // gdal cpu
    time_t startCPU = clock();
    hTransformArg = hGenImgProjArg =
        GDALCreateGenImgProjTransformer( hSrcDS, pszSourceSRS,
                                         hDstDS, pszTargetSRS,
                                         TRUE, 1000.0, nOrder );

    if( hTransformArg == NULL )
        exit( 1 );

    pfnTransformer = GDALGenImgProjTransform;
*/
/* -------------------------------------------------------------------- */
/*      Warp the transformer with a linear approximator unless the      */
/*      acceptable error is zero.                                       */
/* -------------------------------------------------------------------- */
/*
    if( dfErrorThreshold != 0.0 )
    {
        hTransformArg = hApproxArg =
            GDALCreateApproxTransformer( GDALGenImgProjTransform,
                                         hGenImgProjArg, dfErrorThreshold );
        pfnTransformer = GDALApproxTransform;
    }
*/
/* -------------------------------------------------------------------- */
/*      Now actually invoke the warper to do the work.                  */
/* -------------------------------------------------------------------- */
    
/*    
    GDALSimpleImageWarp( hSrcDS, hDstDS, 0, NULL,
                         pfnTransformer, hTransformArg,
                         GDALTermProgress, NULL, papszWarpOptions );
    time_t endCPU = clock();
    printf("Running time: %lfs\n", static_cast<double>(endCPU - startCPU));

    CSLDestroy( papszWarpOptions );

    if( hApproxArg != NULL )
        GDALDestroyApproxTransformer( hApproxArg );

    if( hGenImgProjArg != NULL )
        GDALDestroyGenImgProjTransformer( hGenImgProjArg );


*/
    // our cpu
    OGRSpatialReference poSourceSRS(pszSourceSRS);
    OGRSpatialReference poTargetSRS(pszTargetSRS);

    const char *sourceCode  = poSourceSRS.GetAuthorityCode("GEOGCS");
    const char *targetCode  = poTargetSRS.GetAuthorityCode("GEOGCS");
    //printf("source code: %s\n", sourceCode);
    //printf("target code: %s\n", targetCode);

	double srcDatum[4], dstDatum[4];
    double targetSemiMajor = dstDatum[0] = poTargetSRS.GetSemiMajor();
    double targetSemiMinor = dstDatum[1] =  poTargetSRS.GetSemiMinor();
    double targetSquaredEccentricity = dstDatum[2] = poTargetSRS.GetSquaredEccentricity();
    double targetInvFlattening = poTargetSRS.GetInvFlattening();
    double targetEpsilon = dstDatum[3] = getEpsilon(targetSquaredEccentricity);
    double targetOffset = poTargetSRS.GetPrimeMeridian();
    double targetAngularUnits = poTargetSRS.GetAngularUnits();
	double srcToWGS84[7] = { 0.0 }, dstToWGS84[7] = {0.0};
    double sourceSemiMajor = srcDatum[0] = poSourceSRS.GetSemiMajor();
    double sourceSemiMinor = srcDatum[1] = poSourceSRS.GetSemiMinor();
    double sourceSquaredEccentricity = srcDatum[2] = poSourceSRS.GetSquaredEccentricity();
    double sourceInvFlattening = poSourceSRS.GetInvFlattening();
    double sourceEpsilon = srcDatum[3] = getEpsilon(sourceSquaredEccentricity);
    double sourceOffset = poSourceSRS.GetPrimeMeridian();
    double sourceAngularUnits = poSourceSRS.GetAngularUnits();
    if (OGRERR_NONE != poTargetSRS.GetTOWGS84(dstToWGS84) || OGRERR_NONE != poSourceSRS.GetTOWGS84(srcToWGS84)) {
        printf("TOWGS84 isn't available\n");
        //exit(-1);
    }
    char *sourceProj4, *targetProj4;
    poSourceSRS.exportToProj4(&sourceProj4);
    poTargetSRS.exportToProj4(&targetProj4);
    //printf("source proj4: %s\nsource proj4: %s\n", sourceProj4, targetProj4);
    
    //printf("Semi Major = %lf\nSemi Minor = %lf\nInverse Flatenning = %lf\nsquaredEccentricity = %lf\n", 
    //        semiMajor, semiMinor, invFlattening, squaredEccentricity);
    //for (int i = 0; i < 7; ++i) {
    //    printf("%.10lf\n", coef[i]);
    //}
    for (int i = 3; i <= 5; ++i) {
        srcToWGS84[i] *= M_PI / 180 / 60 /60;
        dstToWGS84[i] *= M_PI / 180 / 60 /60;
    }
    srcToWGS84[6] = 1 + srcToWGS84[6] * 1e-6;
    dstToWGS84[6] = 1 + dstToWGS84[6] * 1e-6;

	/******************************************/
	/*  		projection coordinate         */
	/******************************************/
	/*
    if (poSourceSRS.IsProjected()) {
        // source srs
        double scaleFactor = poTargetSRS.GetProjParm(SRS_PP_SCALE_FACTOR);
        printf("scale factor = %lf\n", scaleFactor);
    }

    if (poTargetSRS.IsProjected()) {
        // target srs
        double scaleFactor = poTargetSRS.GetProjParm(SRS_PP_SCALE_FACTOR);
        printf("scale factor = %lf\n", scaleFactor);
    }
	*/

    int nDstXSize = GDALGetRasterXSize(hDstDS);
    int nDstYSize = GDALGetRasterXSize(hDstDS);

    double targetGeotransform[6], sourceGeotransform[6];
    if (CE_None != GDALGetGeoTransform(hDstDS, targetGeotransform) ||
		GDALGetGeoTransform(hSrcDS, sourceGeotransform)) {
        printf("Error: can't get GeoTransform\n");
        exit(-1);
    }

	
	if (!useGPU) {
    	int2 *output = new int2[nDstXSize * nDstYSize];
    	clock_t start = clock();
    	for (int ix = 0; ix < nDstXSize; ++ix) {
        	for (int iy = 0; iy < nDstYSize; ++iy) {
            	// pixel/line coordinate to geopgrahic coordinate
            	//printf("(%d, %d) -> ", ix, iy);

            	double3 coord = pl2Geographic(make_int2(ix, iy), targetGeotransform);
            	//printf("(%lf, %lf, %lf) -> ", coord.x, coord.y, coord.z);
            	coord.x *= targetAngularUnits;
            	coord.y *= targetAngularUnits;

            	// grographic coordinate to grocentric coordinate
            	coord = geographic2Geocentric(coord, targetSemiMajor, targetSquaredEccentricity);
            	//printf("(%lf, %lf, %lf) -> ", coord.x, coord.y, coord.z);

            	// geocentric To WGS84
          		coord = geocentric2WGS84(coord, dstToWGS84);
            	//printf("(%lf, %lf, %lf) -> ", coord.x, coord.y, coord.z);

				// WGS84 to geocentric
            	coord = WGS84ToGeocentric(coord, srcToWGS84);
            	//printf("(%lf, %lf, %lf) -> ", coord.x, coord.y, coord.z);


            	//geocentric coordinate to geographic coordinate
            	coord = geocentric2Geographic(coord, sourceSemiMajor, sourceSemiMinor, 
                                                    sourceSquaredEccentricity,sourceEpsilon);
            	//printf("(%lf, %lf, %lf) -> ", coord.x, coord.y, coord.z);
            	coord.x /= sourceAngularUnits;
            	coord.y /= sourceAngularUnits;
           		//printf("(%lf, %lf, %lf) -> ", coord.x, coord.y, coord.z);

            	output[ix + iy * nDstXSize] = geographic2Pl(coord, sourceGeotransform, 
                                sourceGeotransform[1] * sourceGeotransform[5] - sourceGeotransform[2] * sourceGeotransform[4]);
           		if (output[ix + iy * nDstXSize].y >= nSrcXSize || output[ix + iy * nDstXSize].y >= nSrcYSize) {
                	printf("Error: out of bound\n");
                	exit(-1);
            	}
				/*
				printf("(%d, %d)\n", output[ix + iy * nDstXSize].x, output[ix + iy * nDstXSize].y);
				int flag = 0;
				scanf("%d", &flag);
				if (!flag) {
					exit(1);
				}
				*/
        	}
    	}
    	clock_t end = clock();
    	printf("time = %lfms\n", (double)(end - start));

    	// IO
    	uchar *image = new uchar[nDstXSize * nDstYSize];
    	for (int iBand = 0; iBand < nBandCount; ++iBand) {
        	for (int ix = 0; ix < nDstXSize; ++ix) {
            	for (int iy = 0; iy < nDstYSize; ++iy) {
                	int pos = ix + iy * nDstXSize;
                	if (output[pos].x < nSrcXSize && output[pos].y < nSrcYSize) {
                    	image[ix + iy * nDstXSize] = papabySrcData[iBand][output[pos].x + output[pos].y * nSrcXSize];
                	} else {
                    	image[ix + iy * nDstXSize] = 0;
                	}
            	}
        	}
        	if( GDALRasterIO(
            	    GDALGetRasterBand(hDstDS,iBand+1), GF_Write,
                		0, 0, nDstXSize, nDstYSize,
                		image, nSrcXSize, nSrcYSize, GDT_Byte,
                		0, 0 ) != CE_None )
        	{
            	CPLError( CE_Failure, CPLE_FileIO,
                	      "GDALSimpleImageWarp GDALRasterIO failure %s",
                    	  CPLGetLastErrorMsg() );
        	}
    	}
		// clean
    	delete[] output;
    	delete[] image;
	}
	else {
    // GPU
		initConstant(sourceGeotransform, srcToWGS84, srcDatum, 
					 targetGeotransform, dstToWGS84, dstDatum);

		/*
		double *d_srcGeoTransform, *d_srcToWGS84, *d_srcDatum,
				*d_dstGeoTransform, *d_dstToWGS84, *d_dstDatum;

		checkCudaErrors(cudaMalloc((void**)&d_srcGeoTransform, 6 * sizeof(double)));
		checkCudaErrors(cudaMemcpy(d_srcGeoTransform, sourceGeotransform, 6 * sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_srcToWGS84, 7 * sizeof(double)));
		checkCudaErrors(cudaMemcpy(d_srcToWGS84, srcToWGS84, 7 * sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_srcDatum, 4 * sizeof(double)));
    	checkCudaErrors(cudaMemcpy(d_srcDatum, srcDatum, 4 * sizeof(double), cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMalloc((void**)&d_dstGeoTransform, 6 * sizeof(double)));
     	checkCudaErrors(cudaMemcpy(d_dstGeoTransform, targetGeotransform, 6 * sizeof(double), cudaMemcpyHostToDevice));
     	checkCudaErrors(cudaMalloc((void**)&d_dstToWGS84, 7 * sizeof(double)));
     	checkCudaErrors(cudaMemcpy(d_dstToWGS84, dstToWGS84, 7 * sizeof(double), cudaMemcpyHostToDevice));
     	checkCudaErrors(cudaMalloc((void**)&d_dstDatum, 4 * sizeof(double)));
     	checkCudaErrors(cudaMemcpy(d_dstDatum, dstDatum, 4 * sizeof(double), cudaMemcpyHostToDevice));
		*/

   		int2 *d_coord;
    	checkCudaErrors(cudaMalloc((void **)&d_coord, nDstXSize * nDstYSize * sizeof(int2)));
   		//double3 *d_coord;
    	//checkCudaErrors(cudaMalloc((void **)&d_coord, nDstXSize * nDstYSize * sizeof(double3)));
    
    	dim3 blockSize(32, 32);
    	dim3 gridSize((nDstXSize + 31) / 32, (nDstYSize + 31) / 32);

		// tansform coordinate
    	transformGPUTest(nDstXSize, nDstYSize, d_coord, blockSize, gridSize);
    

		/*
		checkCudaErrors(cudaFree(d_srcGeoTransform));
		checkCudaErrors(cudaFree(d_srcToWGS84));
		checkCudaErrors(cudaFree(d_srcDatum));
		checkCudaErrors(cudaFree(d_dstGeoTransform));
		checkCudaErrors(cudaFree(d_dstToWGS84));
		checkCudaErrors(cudaFree(d_dstDatum));
		*/


		int2 *h_coord = (int2*)malloc(nDstXSize * nDstYSize * sizeof(int2));
		checkCudaErrors(cudaMemcpy(h_coord, d_coord, nDstXSize * nDstYSize * sizeof(int2), cudaMemcpyDeviceToHost));
		//double3 *h_coord = (double3*)malloc(nDstXSize * nDstYSize * sizeof(double3));
		//checkCudaErrors(cudaMemcpy(h_coord, d_coord, nDstXSize * nDstYSize * sizeof(double3), cudaMemcpyDeviceToHost));

		/*
		for (int i = 0; i < 5; ++i)
		{
			for (int j = 0; j < 5; ++j)
			{
				printf("(%d, %d)\n", h_coord[i + j * nDstXSize].x, h_coord[i + j * nDstXSize].y);
				//printf("(%lf, %lf)\n", h_coord[i + j * nDstXSize].x, h_coord[i + j * nDstXSize].y);
			}
		}
		*/
    	uchar *h_channel = (uchar*)malloc(nDstXSize * nDstYSize * sizeof(uchar));
		if (h_channel == NULL)
		{
			printf("out of memory.\n");
		}
		uchar *d_channel;
		checkCudaErrors(cudaMalloc((void**)&d_channel, nDstXSize * nDstYSize * sizeof(uchar)));
		// resample
		for (int iBand = 0; iBand < nBandCount; ++iBand)
		{
			initTexture(nSrcXSize, nSrcYSize, papabySrcData[iBand]);
			render(nDstXSize, nDstYSize, 0.0, 0.0, 1.0, 0.0, 0.0, blockSize, gridSize, mode, d_channel, d_coord);
			checkCudaErrors(cudaMemcpy(h_channel, d_channel, nDstXSize * nDstYSize * sizeof(uchar), cudaMemcpyDeviceToHost));
        	if( GDALRasterIO(
            	    GDALGetRasterBand(hDstDS,iBand+1), GF_Write,
                	0, 0, nDstXSize, nDstYSize,
                	h_channel, nDstXSize, nDstYSize, GDT_Byte,
                	0, 0 ) != CE_None )
        	{
            	CPLError( CE_Failure, CPLE_FileIO,
                	      "GDALSimpleImageWarp GDALRasterIO failure %s",
                    	  CPLGetLastErrorMsg() );
        	}
			unbindTexture();
		}
    
    	//clean up
    	free(h_channel);
    	checkCudaErrors(cudaFree(d_channel));
		freeTexture();
    	checkCudaErrors(cudaFree(d_coord));
	}
    

/* -------------------------------------------------------------------- */
/*      Cleanup.                                                        */
/* -------------------------------------------------------------------- */

    GDALClose( (GDALDatasetH)hDstDS );
    GDALClose( (GDALDatasetH)hSrcDS );
    
    free(papabySrcData);
    GDALDumpOpenDatasets( stderr );

    GDALDestroyDriverManager();

    exit( 0 );
}

/************************************************************************/
/*                        GDALWarpCreateOutput()                        */
/*                                                                      */
/*      Create the output file based on various commandline options,    */
/*      and the input file.                                             */
/************************************************************************/

static GDALDatasetH
GDALWarpCreateOutput( GDALDatasetH hSrcDS, const char *pszFilename,
                      const char *pszFormat, const char *pszSourceSRS,
                      const char *pszTargetSRS, int nOrder,
                      char **papszCreateOptions )

{
    GDALDriverH hDriver;
    GDALDatasetH hDstDS;
    void *hTransformArg;
    double adfDstGeoTransform[6];
    int nPixels=0, nLines=0;
    GDALColorTableH hCT;

/* -------------------------------------------------------------------- */
/*      Find the output driver.                                         */
/* -------------------------------------------------------------------- */
    hDriver = GDALGetDriverByName( pszFormat );
    if( hDriver == NULL
        || GDALGetMetadataItem( hDriver, GDAL_DCAP_CREATE, NULL ) == NULL )
    {
        int iDr;

        printf( "Output driver `%s' not recognised or does not support\n",
                pszFormat );
        printf( "direct output file creation.  The following format drivers are configured\n"
                "and support direct output:\n" );

        for( iDr = 0; iDr < GDALGetDriverCount(); iDr++ )
        {
            GDALDriverH hDriver = GDALGetDriver(iDr);

            if( GDALGetMetadataItem( hDriver, GDAL_DCAP_CREATE, NULL) != NULL )
            {
                printf( "  %s: %s\n",
                        GDALGetDriverShortName( hDriver  ),
                        GDALGetDriverLongName( hDriver ) );
            }
        }
        printf( "\n" );
        exit( 1 );
    }

/* -------------------------------------------------------------------- */
/*      Create a transformation object from the source to               */
/*      destination coordinate system.                                  */
/* -------------------------------------------------------------------- */
    hTransformArg =
        GDALCreateGenImgProjTransformer( hSrcDS, pszSourceSRS,
                                         NULL, pszTargetSRS,
                                         TRUE, 1000.0, nOrder );

    if( hTransformArg == NULL )
        return NULL;

/* -------------------------------------------------------------------- */
/*      Get approximate output definition.                              */
/* -------------------------------------------------------------------- */
    if( GDALSuggestedWarpOutput( hSrcDS,
                                 GDALGenImgProjTransform, hTransformArg,
                                 adfDstGeoTransform, &nPixels, &nLines )
        != CE_None )
        return NULL;

    GDALDestroyGenImgProjTransformer( hTransformArg );

/* -------------------------------------------------------------------- */
/*      Did the user override some parameters?                          */
/* -------------------------------------------------------------------- */
    if( dfXRes != 0.0 && dfYRes != 0.0 )
    {
        CPLAssert( nPixels == 0 && nLines == 0 );
        if( dfMinX == 0.0 && dfMinY == 0.0 && dfMaxX == 0.0 && dfMaxY == 0.0 )
        {
            dfMinX = adfDstGeoTransform[0];
            dfMaxX = adfDstGeoTransform[0] + adfDstGeoTransform[1] * nPixels;
            dfMaxY = adfDstGeoTransform[3];
            dfMinY = adfDstGeoTransform[3] + adfDstGeoTransform[5] * nLines;
        }

        nPixels = (int) ((dfMaxX - dfMinX + (dfXRes/2.0)) / dfXRes);
        nLines = (int) ((dfMaxY - dfMinY + (dfYRes/2.0)) / dfYRes);
        adfDstGeoTransform[0] = dfMinX;
        adfDstGeoTransform[3] = dfMaxY;
        adfDstGeoTransform[1] = dfXRes;
        adfDstGeoTransform[5] = -dfYRes;
    }

    else if( nForcePixels != 0 && nForceLines != 0 )
    {
        if( dfMinX == 0.0 && dfMinY == 0.0 && dfMaxX == 0.0 && dfMaxY == 0.0 )
        {
            dfMinX = adfDstGeoTransform[0];
            dfMaxX = adfDstGeoTransform[0] + adfDstGeoTransform[1] * nPixels;
            dfMaxY = adfDstGeoTransform[3];
            dfMinY = adfDstGeoTransform[3] + adfDstGeoTransform[5] * nLines;
        }

        dfXRes = (dfMaxX - dfMinX) / nForcePixels;
        dfYRes = (dfMaxY - dfMinY) / nForceLines;

        adfDstGeoTransform[0] = dfMinX;
        adfDstGeoTransform[3] = dfMaxY;
        adfDstGeoTransform[1] = dfXRes;
        adfDstGeoTransform[5] = -dfYRes;

        nPixels = nForcePixels;
        nLines = nForceLines;
    }

    else if( dfMinX != 0.0 || dfMinY != 0.0 || dfMaxX != 0.0 || dfMaxY != 0.0 )
    {
        dfXRes = adfDstGeoTransform[1];
        dfYRes = fabs(adfDstGeoTransform[5]);

        nPixels = (int) ((dfMaxX - dfMinX + (dfXRes/2.0)) / dfXRes);
        nLines = (int) ((dfMaxY - dfMinY + (dfYRes/2.0)) / dfYRes);

        adfDstGeoTransform[0] = dfMinX;
        adfDstGeoTransform[3] = dfMaxY;
    }

/* -------------------------------------------------------------------- */
/*      Create the output file.                                         */
/* -------------------------------------------------------------------- */

    printf( "Creating output file is that %dP x %dL.\n", nPixels, nLines );

    hDstDS = GDALCreate( hDriver, pszFilename, nPixels, nLines,
                         GDALGetRasterCount(hSrcDS),
                         GDALGetRasterDataType(GDALGetRasterBand(hSrcDS,1)),
                         papszCreateOptions );

    if( hDstDS == NULL )
        return NULL;

/* -------------------------------------------------------------------- */
/*      Write out the projection definition.                            */
/* -------------------------------------------------------------------- */
    GDALSetProjection( hDstDS, pszTargetSRS );
    GDALSetGeoTransform( hDstDS, adfDstGeoTransform );

/* -------------------------------------------------------------------- */
/*      Copy the color table, if required.                              */
/* -------------------------------------------------------------------- */
    hCT = GDALGetRasterColorTable( GDALGetRasterBand(hSrcDS,1) );
    if( hCT != NULL )
        GDALSetRasterColorTable( GDALGetRasterBand(hDstDS,1), hCT );

    return hDstDS;
}



// ours
static GDALDatasetH
CreateOutput( GDALDatasetH hSrcDS, const char *pszFilename,
                      const char *pszFormat, const char *pszSourceSRS,
                      const char *pszTargetSRS, int nOrder,
                      char **papszCreateOptions )

{
    GDALDriverH hDriver;
    GDALDatasetH hDstDS;
    double adfDstGeoTransform[6];
    int nPixels=0, nLines=0;
    GDALColorTableH hCT;

/* -------------------------------------------------------------------- */
/*      Find the output driver.                                         */
/* -------------------------------------------------------------------- */
    hDriver = GDALGetDriverByName( pszFormat );
    if( hDriver == NULL
        || GDALGetMetadataItem( hDriver, GDAL_DCAP_CREATE, NULL ) == NULL )
    {
        int iDr;

        printf( "Output driver `%s' not recognised or does not support\n",
                pszFormat );
        printf( "direct output file creation.  The following format drivers are configured\n"
                "and support direct output:\n" );

        for( iDr = 0; iDr < GDALGetDriverCount(); iDr++ )
        {
            GDALDriverH hDriver = GDALGetDriver(iDr);

            if( GDALGetMetadataItem( hDriver, GDAL_DCAP_CREATE, NULL) != NULL )
            {
                printf( "  %s: %s\n",
                        GDALGetDriverShortName( hDriver  ),
                        GDALGetDriverLongName( hDriver ) );
            }
        }
        printf( "\n" );
        exit( 1 );
    }


/* -------------------------------------------------------------------- */
/*      Get output definition.                              */
/* -------------------------------------------------------------------- */
    if( !SuggestedWarpOutput( hSrcDS, pszSourceSRS, pszTargetSRS, 
                            adfDstGeoTransform, &nPixels, &nLines ))
        return NULL;

/* -------------------------------------------------------------------- */
/*      Did the user override some parameters?                          */
/* -------------------------------------------------------------------- */
    if( dfXRes != 0.0 && dfYRes != 0.0 )
    {
        CPLAssert( nPixels == 0 && nLines == 0 );
        if( dfMinX == 0.0 && dfMinY == 0.0 && dfMaxX == 0.0 && dfMaxY == 0.0 )
        {
            dfMinX = adfDstGeoTransform[0];
            dfMaxX = adfDstGeoTransform[0] + adfDstGeoTransform[1] * nPixels;
            dfMaxY = adfDstGeoTransform[3];
            dfMinY = adfDstGeoTransform[3] + adfDstGeoTransform[5] * nLines;
        }

        nPixels = (int) ((dfMaxX - dfMinX + (dfXRes/2.0)) / dfXRes);
        nLines = (int) ((dfMaxY - dfMinY + (dfYRes/2.0)) / dfYRes);
        adfDstGeoTransform[0] = dfMinX;
        adfDstGeoTransform[3] = dfMaxY;
        adfDstGeoTransform[1] = dfXRes;
        adfDstGeoTransform[5] = -dfYRes;
    }

    else if( nForcePixels != 0 && nForceLines != 0 )
    {
        if( dfMinX == 0.0 && dfMinY == 0.0 && dfMaxX == 0.0 && dfMaxY == 0.0 )
        {
            dfMinX = adfDstGeoTransform[0];
            dfMaxX = adfDstGeoTransform[0] + adfDstGeoTransform[1] * nPixels;
            dfMaxY = adfDstGeoTransform[3];
            dfMinY = adfDstGeoTransform[3] + adfDstGeoTransform[5] * nLines;
        }

        dfXRes = (dfMaxX - dfMinX) / nForcePixels;
        dfYRes = (dfMaxY - dfMinY) / nForceLines;

        adfDstGeoTransform[0] = dfMinX;
        adfDstGeoTransform[3] = dfMaxY;
        adfDstGeoTransform[1] = dfXRes;
        adfDstGeoTransform[5] = -dfYRes;

        nPixels = nForcePixels;
        nLines = nForceLines;
    }

    else if( dfMinX != 0.0 || dfMinY != 0.0 || dfMaxX != 0.0 || dfMaxY != 0.0 )
    {
        dfXRes = adfDstGeoTransform[1];
        dfYRes = fabs(adfDstGeoTransform[5]);

        nPixels = (int) ((dfMaxX - dfMinX + (dfXRes/2.0)) / dfXRes);
        nLines = (int) ((dfMaxY - dfMinY + (dfYRes/2.0)) / dfYRes);

        adfDstGeoTransform[0] = dfMinX;
        adfDstGeoTransform[3] = dfMaxY;
    }

/* -------------------------------------------------------------------- */
/*      Create the output file.                                         */
/* -------------------------------------------------------------------- */

    printf( "Creating output file is that %dP x %dL.\n", nPixels, nLines );

    hDstDS = GDALCreate( hDriver, pszFilename, nPixels, nLines,
                         GDALGetRasterCount(hSrcDS),
                         GDALGetRasterDataType(GDALGetRasterBand(hSrcDS,1)),
                         papszCreateOptions );

    if( hDstDS == NULL )
        return NULL;

/* -------------------------------------------------------------------- */
/*      Write out the projection definition.                            */
/* -------------------------------------------------------------------- */
    GDALSetProjection( hDstDS, pszTargetSRS );
    GDALSetGeoTransform( hDstDS, adfDstGeoTransform );

/* -------------------------------------------------------------------- */
/*      Copy the color table, if required.                              */
/* -------------------------------------------------------------------- */
    hCT = GDALGetRasterColorTable( GDALGetRasterBand(hSrcDS,1) );
    if( hCT != NULL )
        GDALSetRasterColorTable( GDALGetRasterBand(hDstDS,1), hCT );

    return hDstDS;
}



// ours
bool
SuggestedWarpOutput( GDALDatasetH hSrcDS, const char *pszSourceSRS, 
                     const char* pszTargetSRS, double* adfOuttGeoTransform,
                     int *pnPixels, int *pnLines) {
    const int nInXSize = GDALGetRasterXSize(hSrcDS);
    const int nInYSize = GDALGetRasterYSize(hSrcDS);

    double sourceGeotransform[6];
    if (CE_None != GDALGetGeoTransform(hSrcDS, sourceGeotransform)) {
        printf("Error: can't get Source GeoTransform\n");
        exit(-1);
    }

    const int nSteps = 20;
    int nSamplePoints = 4 * (nSteps + 1);

    double dfStep = 1.0 / nSteps;
    
    int2 *padCoord = (int2*)malloc(sizeof(int2) * nSamplePoints);

    if (padCoord == nullptr) {
        return false;
    }

    for (int iStep = 0; iStep <= nSteps; ++iStep) {
        double dfRatio = (iStep == nSteps) ? 1.0 : iStep * dfStep;

        // Along top
        padCoord[iStep].x = dfRatio * nInXSize;
        padCoord[iStep].y = 0.0;

        // Along bottom
        padCoord[nSteps + 1 + iStep].x = dfRatio * nInXSize;
        padCoord[nSteps + 1 + iStep].y = nInYSize;

        // Along left.
        padCoord[2 * (nSteps + 1) + iStep].x = 0.0;
        padCoord[2 * (nSteps + 1) + iStep].y = dfRatio * nInYSize;

        // Along right.
        padCoord[3 * (nSteps + 1) + iStep].x = nInXSize;
        padCoord[3 * (nSteps + 1) + iStep].y = dfRatio * nInYSize;
    }

    
    geoGraphic2GeoGraphicByWGS84(padCoord, nSamplePoints, pszSourceSRS, pszTargetSRS, sourceGeotransform);

      

    //TODO

    return false;
}
