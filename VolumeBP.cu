///////////////////////////////////////////////////////////////////////////////////////
// VolumeBP.cu
// Developed by Dongjin Kwon
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2011-2014 Dongjin Kwon
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////

//#define CU_USE_TIMER
//#define CU_USE_CUTIL

#if defined(WIN32) || defined(WIN64)
#define WIN32_LEAN_AND_MEAN
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#ifdef CU_USE_CUTIL
#pragma comment(lib, "cutil64.lib")
#endif
#endif
#include "stdafx.h"
//
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifdef CU_USE_CUTIL
#include "cutil.h"
#endif
//#include "cuPrintf.cu"

#ifndef CUDA_SAFE_CALL
#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);
#endif


// launch console
//#pragma comment(linker, "/entry:WinMainCRTStartup /subsystem:console")


#define CU_USE_3D_BLOCK

//#define O1_USE_OFFSET


typedef float REALV;

#define MAX_K		21
#define MAX_L8_1	81

#define MIN(a,b)  (((a) < (b)) ? (a) : (b))
#define MAX(a,b)  (((a) > (b)) ? (a) : (b))
#define TRUNCATE_MIN(a,b) { if ((a) > (b)) (a) = (b); }
#define TRUNCATE_MAX(a,b) { if ((a) < (b)) (a) = (b); }
#define TRUNCATE TRUNCATE_MIN

#define INFINITE_S 1e10


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
#if CUDA_VERSION <= 4000
static __device__ REALV* ddcv;
static REALV* hOffset[3];
static __device__ REALV** dOffset;
static REALV* hRangeTerm[3];
static __device__ REALV** dRangeTerm;
static REALV* hSO1[3][3];
static REALV** ddSO1[3];
static __device__ REALV*** dSO1;
static REALV* hSO2[3][3];
static REALV** ddSO2[3];
static __device__ REALV*** dSO2;
static REALV* hSO1F2Message[3][6];
static REALV** ddSO1F2Message[3];
static __device__ REALV*** dSO1F2Message;
static REALV* hSO2F3Message[3][9];
static REALV** ddSO2F3Message[3];
static __device__ REALV*** dSO2F3Message;
static REALV* hDualMessage[3];
static __device__ REALV** dDualMessage;
//
static int mesh_x, mesh_y, mesh_z, mesh_ex, mesh_ey, mesh_ez;
static int nL, K, num_d;
static REALV gamma;
static REALV alpha_O1, d_O1;
static REALV alpha_O2, d_O2;
static REALV in_scv_w_O1F2, in_scv_w_O2F2, in_scv_w_O2F3;
static __constant__ int c_mesh_x, c_mesh_y, c_mesh_z, c_mesh_ex, c_mesh_ey, c_mesh_ez;
static __constant__ int c_nL, c_K, c_num_d;
static __constant__ REALV c_gamma;
static __constant__ REALV c_alpha_O1, c_d_O1;
static __constant__ REALV c_alpha_O2, c_d_O2;
static __constant__ REALV c_in_scv_w_O1F2, c_in_scv_w_O2F2, c_in_scv_w_O2F3;
//
static REALV disp_ex[MAX_K];
static REALV disp_ey[MAX_K];
static REALV disp_ez[MAX_K];
static __constant__ REALV c_disp_ex[MAX_K];
static __constant__ REALV c_disp_ey[MAX_K];
static __constant__ REALV c_disp_ez[MAX_K];
//
static int smode;
static __constant__ int c_smode;
//
static int L, L2, L4, L5, L6, L7, L8, L4_1, L8_1;
static __constant__ int c_L, c_L2, c_L4, c_L5, c_L6, c_L7, c_L8, c_L4_1, c_L8_1;
//
static size_t dsize;
static size_t tsize;
static int mx, my, mz;
static size_t msize;		// number of partial block
static int mmode;
//
static int threadsInX, threadsInY, threadsInZ;
static int blocksInX, blocksInY, blocksInZ, blocksInZ_4;
static dim3 Dg, Dg_4, Db;
static float invBlocksInY;
static int tsize_e;
static int Ns;
//
static unsigned long long int iLowerBound;
static __device__ unsigned long long int dLowerBound;
#else
static REALV* ddcv;
static REALV* hOffset[3];
static REALV** dOffset;
static REALV* hRangeTerm[3];
static REALV** dRangeTerm;
static REALV* hSO1[3][3];
static REALV** ddSO1[3];
static REALV*** dSO1;
static REALV* hSO2[3][3];
static REALV** ddSO2[3];
static REALV*** dSO2;
static REALV* hSO1F2Message[3][6];
static REALV** ddSO1F2Message[3];
static REALV*** dSO1F2Message;
static REALV* hSO2F3Message[3][9];
static REALV** ddSO2F3Message[3];
static REALV*** dSO2F3Message;
static REALV* hDualMessage[3];
static REALV** dDualMessage;
//
static int mesh_x, mesh_y, mesh_z, mesh_ex, mesh_ey, mesh_ez;
static int nL, K, num_d;
static REALV gamma;
static REALV alpha_O1, d_O1;
static REALV alpha_O2, d_O2;
static REALV in_scv_w_O1F2, in_scv_w_O2F2, in_scv_w_O2F3;
static __constant__ int c_mesh_x, c_mesh_y, c_mesh_z, c_mesh_ex, c_mesh_ey, c_mesh_ez;
static __constant__ int c_nL, c_K, c_num_d;
static __constant__ REALV c_gamma;
static __constant__ REALV c_alpha_O1, c_d_O1;
static __constant__ REALV c_alpha_O2, c_d_O2;
static __constant__ REALV c_in_scv_w_O1F2, c_in_scv_w_O2F2, c_in_scv_w_O2F3;
//
static REALV disp_ex[MAX_K];
static REALV disp_ey[MAX_K];
static REALV disp_ez[MAX_K];
static __constant__ REALV c_disp_ex[MAX_K];
static __constant__ REALV c_disp_ey[MAX_K];
static __constant__ REALV c_disp_ez[MAX_K];
//
static int smode;
static __constant__ int c_smode;
//
static int L, L2, L4, L5, L6, L7, L8, L4_1, L8_1;
static __constant__ int c_L, c_L2, c_L4, c_L5, c_L6, c_L7, c_L8, c_L4_1, c_L8_1;
//
static size_t dsize;
static size_t tsize;
static int mx, my, mz;
static size_t msize;		// number of partial block
static int mmode;
//
static int threadsInX, threadsInY, threadsInZ;
static int blocksInX, blocksInY, blocksInZ, blocksInZ_4;
static dim3 Dg, Dg_4, Db;
static float invBlocksInY;
static int tsize_e;
static int Ns;
//
static unsigned long long int iLowerBound;
static __device__ unsigned long long int dLowerBound;
#endif
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
#if 0
extern void Trace(const char* szFormat, ...);

void cu_Trace(const char* szFormat, ...)
{
    char szTempBuf[2048];
    va_list vlMarker;

    va_start(vlMarker, szFormat);
    vsprintf(szTempBuf, szFormat, vlMarker);
    va_end(vlMarker);

#if 0
	/*
    OutputDebugString(szTempBuf);
	/*/
	{
	    char szTempBuf2[2048];
#ifndef _DEBUG
		sprintf(szTempBuf2, "[VolReg] %s", szTempBuf);
#else
		sprintf(szTempBuf2, "%s", szTempBuf);
#endif
		OutputDebugString(szTempBuf2);
	}
	//*/
#endif
#if 1
	Trace((const char*)szTempBuf);
#endif
}
#endif

void cu_VolInit(REALV* dvol, REALV**** pvol, int ox, int oy, int oz, int mx, int my, int mz, int vd_x, int vd_y, int vd_z, int vd_s)
{
	REALV* hvol;
	int size = mx * my * mz * vd_s;
	int i, j, k, l;

	hvol = (REALV*)malloc(size * sizeof(REALV));

	for (k = oz; k < min(vd_z, oz+mz); k++) {
		for (j = oy; j < min(vd_y, oy+my); j++) {
			for (i = ox; i < min(vd_x, ox+mx); i++) {
				REALV *q = hvol + (((k-oz)*my + (j-oy))*mx + (i-ox))*vd_s;
				for (l = 0; l < vd_s; l++) {
					*q++ = pvol[k][j][i][l];
				}
			}
		}
	}

	CUDA_SAFE_CALL(cudaMemcpy(dvol, hvol, size * sizeof(REALV), cudaMemcpyHostToDevice));

	free(hvol);
}
void cu_VolInit(short* dvol, short**** pvol, int ox, int oy, int oz, int mx, int my, int mz, int vd_x, int vd_y, int vd_z, int vd_s)
{
	short* hvol;
	int size = mx * my * mz * vd_s;
	int i, j, k, l;

	hvol = (short*)malloc(size * sizeof(short));

	for (k = oz; k < min(vd_z, oz+mz); k++) {
		for (j = oy; j < min(vd_y, oy+my); j++) {
			for (i = ox; i < min(vd_x, ox+mx); i++) {
				short *q = hvol + (((k-oz)*my + (j-oy))*mx + (i-ox))*vd_s;
				for (l = 0; l < vd_s; l++) {
					*q++ = pvol[k][j][i][l];
				}
			}
		}
	}

	CUDA_SAFE_CALL(cudaMemcpy(dvol, hvol, size * sizeof(short), cudaMemcpyHostToDevice));

	free(hvol);
}

void cu_VolCopy(REALV* dvol, REALV**** pvol, int ox, int oy, int oz, int mx, int my, int mz, int vd_x, int vd_y, int vd_z, int vd_s)
{
	REALV* hvol;
	int size = mx * my * mz * vd_s;
	int i, j, k, l;

	hvol = (REALV*)malloc(size * sizeof(REALV));

	CUDA_SAFE_CALL(cudaMemcpy(hvol, dvol, size * sizeof(REALV), cudaMemcpyDeviceToHost));

	for (k = oz; k < min(vd_z, oz+mz); k++) {
		for (j = oy; j < min(vd_y, oy+my); j++) {
			for (i = ox; i < min(vd_x, ox+mx); i++) {
				REALV *q = hvol + (((k-oz)*my + (j-oy))*mx + (i-ox))*vd_s;
				for (l = 0; l < vd_s; l++) {
					pvol[k][j][i][l] = *q++;
				}
			}
		}
	}

	free(hvol);
}
void cu_VolCopy(REALV* dvol, REALV**** pvol, int ox, int oy, int oz, int os, int mx, int my, int mz, int ms, int vd_x, int vd_y, int vd_z, int vd_s)
{
	REALV* hvol;
	int size = mx * my * mz * ms;
	int i, j, k, l;

	hvol = (REALV*)malloc(size * sizeof(REALV));

	CUDA_SAFE_CALL(cudaMemcpy(hvol, dvol, size * sizeof(REALV), cudaMemcpyDeviceToHost));

	for (k = oz; k < min(vd_z, oz+mz); k++) {
		for (j = oy; j < min(vd_y, oy+my); j++) {
			for (i = ox; i < min(vd_x, ox+mx); i++) {
				REALV *q = hvol + (((k-oz)*my + (j-oy))*mx + (i-ox))*ms;
				for (l = os; l < min(vd_s, os+ms); l++) {
					//if (pvol[k][j][i][l] != *q) {
					//	TRACE2("%d %d %d %d diff %f %f\n", i, j, k, l, pvol[k][j][i][l], *q);
					//}
					pvol[k][j][i][l] = *q++;
					//pvol[k][j][i][l] = 1;
				}
			}
		}
	}

	free(hvol);
}
void cu_VolCopy(short* dvol, short**** pvol, int ox, int oy, int oz, int mx, int my, int mz, int vd_x, int vd_y, int vd_z, int vd_s)
{
	short* hvol;
	int size = mx * my * mz * vd_s;
	int i, j, k, l;

	hvol = (short*)malloc(size * sizeof(short));

	CUDA_SAFE_CALL(cudaMemcpy(hvol, dvol, size * sizeof(short), cudaMemcpyDeviceToHost));

	for (k = oz; k < min(vd_z, oz+mz); k++) {
		for (j = oy; j < min(vd_y, oy+my); j++) {
			for (i = ox; i < min(vd_x, ox+mx); i++) {
				short *q = hvol + (((k-oz)*my + (j-oy))*mx + (i-ox))*vd_s;
				for (l = 0; l < vd_s; l++) {
					pvol[k][j][i][l] = *q++;
				}
			}
		}
	}

	free(hvol);
}
void cu_VolCopy(short* dvol, short**** pvol, int ox, int oy, int oz, int os, int mx, int my, int mz, int ms, int vd_x, int vd_y, int vd_z, int vd_s)
{
	short* hvol;
	int size = mx * my * mz * vd_s;
	int i, j, k, l;

	hvol = (short*)malloc(size * sizeof(short));

	CUDA_SAFE_CALL(cudaMemcpy(hvol, dvol, size * sizeof(short), cudaMemcpyDeviceToHost));

	for (k = oz; k < min(vd_z, oz+mz); k++) {
		for (j = oy; j < min(vd_y, oy+my); j++) {
			for (i = ox; i < min(vd_x, ox+mx); i++) {
				short *q = hvol + (((k-oz)*my + (j-oy))*mx + (i-ox))*ms;
				for (l = os; l < min(vd_s, os+ms); l++) {
					pvol[k][j][i][l] = *q++;
				}
			}
		}
	}

	free(hvol);
}

extern "C"
BOOL cu_BP_Check()
{
	int i;
	CUresult cu_res;

	cu_res = cuInit(0);
	if (cu_res != CUDA_SUCCESS) {
		return FALSE;
	}

	{
		int count;
		cudaDeviceProp prop;
		cudaError err;
		//
		err = cudaGetDeviceCount(&count);
		if (err != cudaSuccess) {
			return FALSE;
		}
		//
		for (i = 0; i < count; i++) {
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, i));
			TRACE2("GPU %d: major = %d, minor = %d, totalGlobalMem = %u, multiProcessorCount = %d, kernelExecTimeoutEnabled = %d\n", 
				i, prop.major, prop.minor, prop.totalGlobalMem, prop.multiProcessorCount, prop.kernelExecTimeoutEnabled);
		}
		for (i = 0; i < count; i++) {
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, i));
			if (prop.major >= 2 && prop.totalGlobalMem >= 2000000000) {
				CUDA_SAFE_CALL(cudaSetDevice(i));
				TRACE2("CUDA is running on GPU %d\n", i);
				return TRUE;
			}
		}
		if (i == count) {
			TRACE("There's no suitable GPU in this system\n");
			return FALSE;
		}
	}

	return FALSE;
}

extern "C"
void cu_BP_Allocate(int _mesh_x, int _mesh_y, int _mesh_z, int _mesh_ex, int _mesh_ey, int _mesh_ez,
	int _nL, int _K, int _num_d, REALV _alpha_O1, REALV _d_O1, REALV _alpha_O2, REALV _d_O2, REALV _gamma, REALV* _disp_ex, REALV* _disp_ey, REALV* _disp_ez,
	REALV _in_scv_w_O1F2, REALV _in_scv_w_O2F2, REALV _in_scv_w_O2F3)
{
	int i, j;

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	CUresult cu_res;
	TRACE2("cuInit\n");
	for (i = 0; i < 100; i++) {
		cu_res = cuInit(0);
		if (cu_res == CUDA_SUCCESS) {
			break;
		} else {
			Sleep(1000);
		}
	}
	TRACE2("cuInit returns %d\n", cu_res);
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	{
		int count;
		cudaDeviceProp prop;
		cudaError err;
		//
		TRACE2("cudaGetDeviceCount\n");
		//CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
		for (i = 0; i < 100; i++) {
			err = cudaGetDeviceCount(&count);
			if (err == cudaSuccess) {
				break;
			} else {
				Sleep(1000);
			}
		}
		if (err != cudaSuccess) {
			TRACE("cudaGetDeviceCount returns error\n");
			exit(0);
		}
		//
		for (i = 0; i < count; i++) {
			TRACE2("cudaGetDeviceProperties\n");
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, i));
			TRACE2("GPU %d: major = %d, minor = %d, totalGlobalMem = %u, multiProcessorCount = %d, kernelExecTimeoutEnabled = %d\n", 
				i, prop.major, prop.minor, prop.totalGlobalMem, prop.multiProcessorCount, prop.kernelExecTimeoutEnabled);
		}
		for (i = 0; i < count; i++) {
			TRACE2("cudaGetDeviceProperties\n");
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, i));
			if (prop.major >= 2 && prop.totalGlobalMem >= 2000000000) {
				CUDA_SAFE_CALL(cudaSetDevice(i));
				TRACE2("CUDA is running on GPU %d\n", i);
				break;
			}
		}
		if (i == count) {
			TRACE("There's no suitable GPU in this system\n");
			return;
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	TRACE2("cudaDeviceReset\n");
	CUDA_SAFE_CALL(cudaDeviceReset());
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

	if (K > MAX_K) {
		TRACE("error: K = %d is larger than MAX_K = %d\n", K, MAX_K);
		return;
	}

	mesh_x = _mesh_x;
	mesh_y = _mesh_y;
	mesh_z = _mesh_z;
	mesh_ex = _mesh_ex;
	mesh_ey = _mesh_ey;
	mesh_ez = _mesh_ez;
	nL = _nL;
	K = _K;
	num_d = _num_d;
	//
	gamma = _gamma;
	alpha_O1 = _alpha_O1;
	d_O1 = _d_O1;
	alpha_O2 = _alpha_O2;
	d_O2 = _d_O2;
	//
	in_scv_w_O1F2 = _in_scv_w_O1F2;
	in_scv_w_O2F2 = _in_scv_w_O2F2;
	in_scv_w_O2F3 = _in_scv_w_O2F3;
	//
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mesh_x, &mesh_x, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mesh_y, &mesh_y, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mesh_z, &mesh_z, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mesh_ex, &mesh_ex, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mesh_ey, &mesh_ey, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mesh_ez, &mesh_ez, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_nL, &nL, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_K, &K, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_num_d, &num_d, sizeof(int)));
	//
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_gamma, &gamma, sizeof(REALV)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_alpha_O1, &alpha_O1, sizeof(REALV)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_d_O1, &d_O1, sizeof(REALV)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_alpha_O2, &alpha_O2, sizeof(REALV)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_d_O2, &d_O2, sizeof(REALV)));
	//
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_in_scv_w_O1F2, &in_scv_w_O1F2, sizeof(REALV)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_in_scv_w_O2F2, &in_scv_w_O2F2, sizeof(REALV)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_in_scv_w_O2F3, &in_scv_w_O2F3, sizeof(REALV)));
	//
	for (i = 0; i < K; i++) {
		disp_ex[i] = _disp_ex[i];
		disp_ey[i] = _disp_ey[i];
		disp_ez[i] = _disp_ez[i];
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_disp_ex, &disp_ex, K * sizeof(REALV)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_disp_ey, &disp_ey, K * sizeof(REALV)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_disp_ez, &disp_ez, K * sizeof(REALV)));
	//
	if (in_scv_w_O1F2 == -2) {
		smode = 0;
	}
	if (in_scv_w_O2F3 == -2) {
		smode = 1;
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_smode, &smode, sizeof(int)));
	//
	L  = K / 2;
	L2 = L * 2;
	L4 = L * 4;
	L5 = L * 5;
	L6 = L * 6;
	L7 = L * 7;
	L8 = L * 8;
	L4_1 = L * 4 + 1;
	L8_1 = L * 8 + 1;
	//
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_L, &L, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_L2, &L2, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_L4, &L4, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_L5, &L5, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_L6, &L6, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_L7, &L7, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_L8, &L8, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_L4_1, &L4_1, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_L8_1, &L8_1, sizeof(int)));
	//
	{
		size_t free_mem, total_mem;
		CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
		dsize = free_mem;
	}
	//
	size_t msize_xy;	// size (a slice) of dcv, Offset, RangeTerm, SO1, SO2, Messages
	mx = mesh_x;
	my = mesh_y;
	if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 == -2)) {
		tsize = mesh_x * mesh_y * mesh_z * (num_d + 3 + 3*K + 9*L4_1 + 18*K + 3*K) * sizeof(REALV);
		//msize_xy = mesh_x * mesh_y * (num_d + 3 + 3*K + 9*L4_1 + 18*K + 3*K) * sizeof(REALV);
		msize_xy = mesh_x * mesh_y * max(num_d/4 + 3 + 3*K + 18*K + 3*K, 3 + 3*K + 3*L4_1 + 18*K + 3*K) * sizeof(REALV);
	} else if ((in_scv_w_O1F2 == -2) && (in_scv_w_O2F3 != -2)) {
		tsize = mesh_x * mesh_y * mesh_z * (num_d + 3 + 3*K + 9*L8_1 + 27*K + 3*K) * sizeof(REALV);
		//msize_xy = mesh_x * mesh_y * (num_d + 3 + 3*K + 9*L8_1 + 27*K + 3*K) * sizeof(REALV);
		msize_xy = mesh_x * mesh_y * max(num_d/4 + 3 + 3*K + 27*K + 3*K, 3 + 3*K + 3*L8_1 + 27*K + 3*K) * sizeof(REALV);
	} else if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 != -2)) {
		tsize = mesh_x * mesh_y * mesh_z * (num_d + 3 + 3*K + (9*L4_1 + 18*K) + (9*L8_1 + 27*K) + 3*K) * sizeof(REALV);
		//msize_xy = mesh_x * mesh_y * (num_d + 3 + 3*K + (9*L4_1 + 18*K) + (9*L8_1 + 27*K) + 3*K) * sizeof(REALV);
		msize_xy = mesh_x * mesh_y * max(num_d/4 + 3 + 3*K + 27*K + 3*K, 3 + 3*K + (3*L4_1 + 18*K) + (3*L8_1 + 27*K) + 3*K) * sizeof(REALV);
	}
	//
	tsize += 1000000000;
	//
	mz = min((int)(dsize / msize_xy), mesh_z);
	if (mz < mesh_z) {
		TRACE("mz = %d, mesh_z = %d\n", mz, mesh_z);
		return;
	}
	msize = mx * my * mz;
	if (tsize >= dsize) {
		mmode = 1;
	} else {
		mmode = 0;
	}
	TRACE2("tsize = %u, dsize = %u -> mmode = %d\n", tsize, dsize, mmode);

	/*
	TRACE2("dsize = %d\n", dsize);
	TRACE2("bsize_xy = %d\n", bsize_xy);
	TRACE2("msize_xy = %d\n", msize_xy);
	TRACE2("mz = %d\n", mz);
	TRACE2("msize = %d\n", msize);
	*/


	///////////////////////////////////////////////////////////////////////////////////////
	// Setting grid and block sizes
	///////////////////////////////////////////////////////////////////////////////////////
	threadsInX = 8;
	threadsInY = 8;
	threadsInZ = 1;
	blocksInX   = (mesh_x   + threadsInX-1) / threadsInX;
	blocksInY   = (mesh_y   + threadsInY-1) / threadsInY;
	blocksInZ   = (mz       + threadsInZ-1) / threadsInZ;
	blocksInZ_4 = ((mz / 4) + threadsInZ-1) / threadsInZ;
	Dg   = dim3(blocksInX, blocksInY, blocksInZ);
	Dg_4 = dim3(blocksInX, blocksInY, blocksInZ_4);
	Db   = dim3(threadsInX, threadsInY, threadsInZ);
	invBlocksInY = 1.0f / (float)blocksInY;
	//tsize_e = 6*K + num_d;
	tsize_e = 4*K;
	Ns = threadsInX * threadsInY * threadsInZ * tsize_e * sizeof(REALV);
	//
	//CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferNone));
	//
	if (Ns > 16000) {
		TRACE("error: Ns = %d is larger than 16000\n", Ns);
		return;
	}

	/*
	TRACE2("Db = %d, %d, %d\n", threadsInX, threadsInY, threadsInZ);
	TRACE2("Dg = %d, %d, %d\n", blocksInX, blocksInY*blocksInZ, 1);
	*/
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
#else
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
#endif

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	if (mmode == 0) {
		CUDA_SAFE_CALL(cudaMalloc(&ddcv, msize * num_d * sizeof(REALV)));
	}
	//
	CUDA_SAFE_CALL(cudaMalloc(&dOffset, 3 * sizeof(REALV*)));
	CUDA_SAFE_CALL(cudaMalloc(&dRangeTerm, 3 * sizeof(REALV*)));
	CUDA_SAFE_CALL(cudaMalloc(&dDualMessage, 3 * sizeof(REALV*)));
	//
	CUDA_SAFE_CALL(cudaMalloc(&dSO1, 3 * sizeof(REALV**)));
	CUDA_SAFE_CALL(cudaMalloc(&dSO1F2Message, 3 * sizeof(REALV**)));
	//
	CUDA_SAFE_CALL(cudaMalloc(&dSO2, 3 * sizeof(REALV**)));
	CUDA_SAFE_CALL(cudaMalloc(&dSO2F3Message, 3 * sizeof(REALV**)));
	//
	for (i = 0; i < 3; i++) {
		CUDA_SAFE_CALL(cudaMalloc(&hOffset[i], msize * sizeof(REALV)));
		CUDA_SAFE_CALL(cudaMalloc(&hRangeTerm[i], msize * K * sizeof(REALV)));
		CUDA_SAFE_CALL(cudaMalloc(&hDualMessage[i], msize * K * sizeof(REALV)));
		//
		if (in_scv_w_O1F2 != -2) {
			CUDA_SAFE_CALL(cudaMalloc(&ddSO1[i], 3 * sizeof(REALV*)));
			if (mmode == 0) {
				for (j = 0; j < 3; j++) {
					CUDA_SAFE_CALL(cudaMalloc(&hSO1[i][j], msize * L4_1 * sizeof(REALV)));
				}
				CUDA_SAFE_CALL(cudaMemcpy(ddSO1[i], hSO1[i], 3 * sizeof(REALV*), cudaMemcpyHostToDevice));
			}
			//
			CUDA_SAFE_CALL(cudaMalloc(&ddSO1F2Message[i], 6 * sizeof(REALV*)));
			for (j = 0; j < 6; j++) {
				CUDA_SAFE_CALL(cudaMalloc(&hSO1F2Message[i][j], msize * K * sizeof(REALV)));
			}
			CUDA_SAFE_CALL(cudaMemcpy(ddSO1F2Message[i], hSO1F2Message[i], 6 * sizeof(REALV*), cudaMemcpyHostToDevice));
		}
		if (in_scv_w_O2F3 != -2) {
			CUDA_SAFE_CALL(cudaMalloc(&ddSO2[i], 3 * sizeof(REALV*)));
			if (mmode == 0) {
				for (j = 0; j < 3; j++) {
					CUDA_SAFE_CALL(cudaMalloc(&hSO2[i][j], msize * L8_1 * sizeof(REALV)));
				}
				CUDA_SAFE_CALL(cudaMemcpy(ddSO2[i], hSO2[i], 3 * sizeof(REALV*), cudaMemcpyHostToDevice));
			}
			//
			CUDA_SAFE_CALL(cudaMalloc(&ddSO2F3Message[i], 9 * sizeof(REALV*)));
			for (j = 0; j < 9; j++) {
				CUDA_SAFE_CALL(cudaMalloc(&hSO2F3Message[i][j], msize * K * sizeof(REALV)));
			}
			CUDA_SAFE_CALL(cudaMemcpy(ddSO2F3Message[i], hSO2F3Message[i], 9 * sizeof(REALV*), cudaMemcpyHostToDevice));
		}
	}
	CUDA_SAFE_CALL(cudaMemcpy(dRangeTerm, hRangeTerm, 3 * sizeof(REALV*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dDualMessage, hDualMessage, 3 * sizeof(REALV*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dOffset, hOffset, 3 * sizeof(REALV*), cudaMemcpyHostToDevice));
	if (in_scv_w_O1F2 != -2) {
		CUDA_SAFE_CALL(cudaMemcpy(dSO1F2Message, ddSO1F2Message, 3 * sizeof(REALV**), cudaMemcpyHostToDevice));
	}
	if (in_scv_w_O2F3 != -2) {
		CUDA_SAFE_CALL(cudaMemcpy(dSO2F3Message, ddSO2F3Message, 3 * sizeof(REALV**), cudaMemcpyHostToDevice));
	}
	if (mmode == 0) {
		if (in_scv_w_O1F2 != -2) {
			CUDA_SAFE_CALL(cudaMemcpy(dSO1, ddSO1, 3 * sizeof(REALV**), cudaMemcpyHostToDevice));
		}
		if (in_scv_w_O2F3 != -2) {
			CUDA_SAFE_CALL(cudaMemcpy(dSO2, ddSO2, 3 * sizeof(REALV**), cudaMemcpyHostToDevice));
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	cutStopTimer(timer);
	TRACE2("alloc time = %f\n", cutGetTimerValue(timer));
	cutDeleteTimer(timer);
#else
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	TRACE2("alloc time = %f\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
#endif
}

extern "C"
void cu_BP_Free()
{
	int i, j;

	cudaDeviceSynchronize();

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
#else
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
#endif

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	if (mmode == 0) {
		CUDA_SAFE_CALL(cudaFree(ddcv));
	}
	//
	for (i = 0; i < 3; i++) {
		CUDA_SAFE_CALL(cudaFree(hOffset[i]));
		CUDA_SAFE_CALL(cudaFree(hRangeTerm[i]));
		CUDA_SAFE_CALL(cudaFree(hDualMessage[i]));
		//
		if (in_scv_w_O1F2 != -2) {
			if (mmode == 0) {
				for (j = 0; j < 3; j++) {
					CUDA_SAFE_CALL(cudaFree(hSO1[i][j]));
				}
			}
			for (j = 0; j < 6; j++) {
				CUDA_SAFE_CALL(cudaFree(hSO1F2Message[i][j]));
			}
			CUDA_SAFE_CALL(cudaFree(ddSO1[i]));
			CUDA_SAFE_CALL(cudaFree(ddSO1F2Message[i]));
		}
		if (in_scv_w_O2F3 != -2) {
			if (mmode == 0) {
				for (j = 0; j < 3; j++) {
					CUDA_SAFE_CALL(cudaFree(hSO2[i][j]));
				}
			}
			for (j = 0; j < 9; j++) {
				CUDA_SAFE_CALL(cudaFree(hSO2F3Message[i][j]));
			}
			CUDA_SAFE_CALL(cudaFree(ddSO2[i]));
			CUDA_SAFE_CALL(cudaFree(ddSO2F3Message[i]));
		}
	}
	CUDA_SAFE_CALL(cudaFree(dOffset));
	CUDA_SAFE_CALL(cudaFree(dRangeTerm));
	CUDA_SAFE_CALL(cudaFree(dDualMessage));
	//
	CUDA_SAFE_CALL(cudaFree(dSO1));
	CUDA_SAFE_CALL(cudaFree(dSO1F2Message));
	//
	CUDA_SAFE_CALL(cudaFree(dSO2));
	CUDA_SAFE_CALL(cudaFree(dSO2F3Message));
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	cutStopTimer(timer);
	TRACE2("free time = %f\n", cutGetTimerValue(timer));
	cutDeleteTimer(timer);
#else
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	TRACE2("free time = %f\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
#endif
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
__device__ static void cu_Add2Message(REALV* message, const REALV* other, int nstates)
{
	int i;
	for (i = 0; i < nstates; i++) {
		message[i] += other[i];
	}
}
#if 0
__device__ static REALV cu_FindMin(REALV* message, int nstates)
{
	REALV min;
	int i;
	min = message[0];
	for (i = 1; i < nstates; i++) {
		if (min > message[i]) {
			min = message[i];
		}
	}
	return min;
}
#endif
__device__ static void cu_SubtractMin(REALV* message, int nstates, REALV& min)
{
	int i;
	min = message[0];
	for (i = 1; i < nstates; i++) {
		if (min > message[i]) {
			min = message[i];
		}
	}
	for (i = 0; i < nstates; i++) {
		message[i] -= min;
	}
}
__device__ static void cu_SubtractMin(REALV* message, REALV* message_out, int nstates, REALV& min)
{
	int i;
	min = message[0];
	for (i = 1; i < nstates; i++) {
		if (min > message[i]) {
			min = message[i];
		}
	}
	for (i = 0; i < nstates; i++) {
		message_out[i] = message[i] - min;
	}
}
__device__ static void cu_Add2MessageDual(REALV* message, int x, int y, int z, int nStates,
	size_t idx, REALV* dRangeTerm, REALV** dSO1F2Message, REALV** dSO2F3Message)
{
	// initialize the message using the range term
	memcpy(message, &dRangeTerm[idx*nStates], sizeof(REALV) * nStates);

	// add spatial messages
	if (c_in_scv_w_O1F2 != -2) {
		if (x > 0) {			// add x- -> x+
			cu_Add2Message(message, &dSO1F2Message[0][idx*nStates], nStates);
		}
		if (x < c_mesh_x-1) {	// add x+ -> x-
			cu_Add2Message(message, &dSO1F2Message[1][idx*nStates], nStates);
		}
		if (y > 0) {			// add y- -> y+
			cu_Add2Message(message, &dSO1F2Message[2][idx*nStates], nStates);
		}
		if (y < c_mesh_y-1) {	// add y+ -> y-
			cu_Add2Message(message, &dSO1F2Message[3][idx*nStates], nStates);
		}
		if (z > 0) {			// add z- -> z+
			cu_Add2Message(message, &dSO1F2Message[4][idx*nStates], nStates);
		}
		if (z < c_mesh_z-1) {	// add z+ -> z-
			cu_Add2Message(message, &dSO1F2Message[5][idx*nStates], nStates);
		}
	}
	if (c_in_scv_w_O2F3 != -2) {
		if (x < c_mesh_x-2) {			// f+ -> x
			cu_Add2Message(message, &dSO2F3Message[0][idx*nStates], nStates);
		}
		if (x > 0 && x < c_mesh_x-1) {	// f0 -> x
			cu_Add2Message(message, &dSO2F3Message[1][idx*nStates], nStates);
		}
		if (x > 1) {					// f- -> x
			cu_Add2Message(message, &dSO2F3Message[2][idx*nStates], nStates);
		}
		if (y < c_mesh_y-2) {
			cu_Add2Message(message, &dSO2F3Message[3][idx*nStates], nStates);
		}
		if (y > 0 && y < c_mesh_y-1) {
			cu_Add2Message(message, &dSO2F3Message[4][idx*nStates], nStates);
		}
		if (y > 1) {
			cu_Add2Message(message, &dSO2F3Message[5][idx*nStates], nStates);
		}
		if (z < c_mesh_z-2) {
			cu_Add2Message(message, &dSO2F3Message[6][idx*nStates], nStates);
		}
		if (z > 0 && z < c_mesh_z-1) {
			cu_Add2Message(message, &dSO2F3Message[7][idx*nStates], nStates);
		}
		if (z > 1) {
			cu_Add2Message(message, &dSO2F3Message[8][idx*nStates], nStates);
		}
	}
}
__device__ static void cu_Add2MessageSpatial_O1F2(REALV* message, int x, int y, int z, int direction, int nStates,
	size_t idx, REALV* dRangeTerm, REALV** dSO1F2Message, REALV** dSO2F3Message, REALV* dDualMessage)
{
	// initialize the message from the dual plane
	memcpy(message, &dDualMessage[idx*nStates], sizeof(REALV) * nStates);

	// add the range term
	cu_Add2Message(message, &dRangeTerm[idx*nStates], nStates);

	// add spatial messages
	if (c_in_scv_w_O1F2 != -2) {
		if (x > 0          && direction != 1) {	// add x- -> x+
			cu_Add2Message(message, &dSO1F2Message[0][idx*nStates], nStates);
		}
		if (x < c_mesh_x-1 && direction != 0) {	// add x+ -> x-
			cu_Add2Message(message, &dSO1F2Message[1][idx*nStates], nStates);
		}
		if (y > 0          && direction != 3) {	// add y- -> y+
			cu_Add2Message(message, &dSO1F2Message[2][idx*nStates], nStates);
		}
		if (y < c_mesh_y-1 && direction != 2) {	// add y+ -> y-
			cu_Add2Message(message, &dSO1F2Message[3][idx*nStates], nStates);
		}
		if (z > 0          && direction != 5) {	// add z- -> z+
			cu_Add2Message(message, &dSO1F2Message[4][idx*nStates], nStates);
		}
		if (z < c_mesh_z-1 && direction != 4) {	// add z+ -> z-
			cu_Add2Message(message, &dSO1F2Message[5][idx*nStates], nStates);
		}
	}
	if (c_in_scv_w_O2F3 != -2) {
		if (x < c_mesh_x-2) {			// f+ -> x
			cu_Add2Message(message, &dSO2F3Message[0][idx*nStates], nStates);
		}
		if (x > 0 && x < c_mesh_x-1) {	// f0 -> x
			cu_Add2Message(message, &dSO2F3Message[1][idx*nStates], nStates);
		}
		if (x > 1) {					// f- -> x
			cu_Add2Message(message, &dSO2F3Message[2][idx*nStates], nStates);
		}
		if (y < c_mesh_y-2) {
			cu_Add2Message(message, &dSO2F3Message[3][idx*nStates], nStates);
		}
		if (y > 0 && y < c_mesh_y-1) {
			cu_Add2Message(message, &dSO2F3Message[4][idx*nStates], nStates);
		}
		if (y > 1) {
			cu_Add2Message(message, &dSO2F3Message[5][idx*nStates], nStates);
		}
		if (z < c_mesh_z-2) {
			cu_Add2Message(message, &dSO2F3Message[6][idx*nStates], nStates);
		}
		if (z > 0 && z < c_mesh_z-1) {
			cu_Add2Message(message, &dSO2F3Message[7][idx*nStates], nStates);
		}
		if (z > 1 && direction != 8) {
			cu_Add2Message(message, &dSO2F3Message[8][idx*nStates], nStates);
		}
	}
}
__device__ static void cu_Add2MessageSpatial_O2F3(REALV* message, int x, int y, int z, int direction, int nStates,
	size_t idx, REALV* dRangeTerm, REALV** dSO1F2Message, REALV** dSO2F3Message, REALV* dDualMessage)
{
	// initialize the message from the dual plane
	memcpy(message, &dDualMessage[idx*nStates], sizeof(REALV) * nStates);

	// add the range term
	cu_Add2Message(message, &dRangeTerm[idx*nStates], nStates);

	// add spatial messages
	if (c_in_scv_w_O1F2 != -2) {
		if (x > 0) {			// add x- -> x+
			cu_Add2Message(message, &dSO1F2Message[0][idx*nStates], nStates);
		}
		if (x < c_mesh_x-1) {	// add x+ -> x-
			cu_Add2Message(message, &dSO1F2Message[1][idx*nStates], nStates);
		}
		if (y > 0) {			// add y- -> y+
			cu_Add2Message(message, &dSO1F2Message[2][idx*nStates], nStates);
		}
		if (y < c_mesh_y-1) {	// add y+ -> y-
			cu_Add2Message(message, &dSO1F2Message[3][idx*nStates], nStates);
		}
		if (z > 0) {			// add z- -> z+
			cu_Add2Message(message, &dSO1F2Message[4][idx*nStates], nStates);
		}
		if (z < c_mesh_z-1) {	// add z+ -> z-
			cu_Add2Message(message, &dSO1F2Message[5][idx*nStates], nStates);
		}
	}
	if (c_in_scv_w_O2F3 != -2) {
		if (x < c_mesh_x-2 && direction != 0) {				// f+ -> x
			cu_Add2Message(message, &dSO2F3Message[0][idx*nStates], nStates);
		}
		if (x > 0 && x < c_mesh_x-1 && direction != 1) {	// f0 -> x
			cu_Add2Message(message, &dSO2F3Message[1][idx*nStates], nStates);
		}
		if (x > 1 && direction != 2) {						// f- -> x
			cu_Add2Message(message, &dSO2F3Message[2][idx*nStates], nStates);
		}
		if (y < c_mesh_y-2 && direction != 3) {
			cu_Add2Message(message, &dSO2F3Message[3][idx*nStates], nStates);
		}
		if (y > 0 && y < c_mesh_y-1 && direction != 4) {
			cu_Add2Message(message, &dSO2F3Message[4][idx*nStates], nStates);
		}
		if (y > 1 && direction != 5) {
			cu_Add2Message(message, &dSO2F3Message[5][idx*nStates], nStates);
		}
		if (z < c_mesh_z-2 && direction != 6) {
			cu_Add2Message(message, &dSO2F3Message[6][idx*nStates], nStates);
		}
		if (z > 0 && z < c_mesh_z-1 && direction != 7) {
			cu_Add2Message(message, &dSO2F3Message[7][idx*nStates], nStates);
		}
		if (z > 1 && direction != 8) {
			cu_Add2Message(message, &dSO2F3Message[8][idx*nStates], nStates);
		}
	}
}
__device__ void cu_ComputeSpatialMessageDT(REALV* message, REALV* message_org, REALV* message_buf, int x, int y, int z, REALV d0, int nStates, int wsize, REALV* c_disp_e)
{
#if 0
	//////////////////////////////////////////
	REALV Min;
	ptrdiff_t l;

	if (message_org != message_buf) {
		memcpy(message_buf, message_org, nStates * sizeof(REALV));
	}

	// use distance transform function to impose smoothness compatibility
	Min = cu_FindMin(message_buf, nStates) + c_d;
	for (l = 1; l < nStates; l++) {
		message_buf[l] = min(message_buf[l], message_buf[l-1] + c_alpha);
	}
	for (l = nStates-2; l >= 0; l--) {
		message_buf[l] = min(message_buf[l], message_buf[l+1] + c_alpha);
	}

	// transform the compatibility 
	int shift = -d0;
	if (abs(shift) > wsize+wsize) { // the shift is too big that there is no overlap
		if (x > 0 || y > 0 || z > 0) {
			for (l = 0; l < nStates; l++) {
				message[l] =  l * c_alpha;
			}
		} else {
			for (l = 0; l < nStates; l++) {
				message[l] = -l * c_alpha;
			}
		}
	} else {
		int start = max(-wsize, shift-wsize);
		int end   = min( wsize, shift+wsize);
		for (l = start; l <= end; l++) {
			message[l-shift+wsize] = message_buf[l+wsize];
		}
		if (start-shift+wsize > 0) {
			for (l = start-shift+wsize-1; l >= 0; l--) {
				message[l] = message[l+1] + c_alpha;
			}
		}
		if (end-shift+wsize < nStates) {
			for (l = end-shift+wsize+1; l < nStates; l++) {
				message[l] = message[l-1] + c_alpha;
			}
		}
	}

	// put back the threshold
	for (l = 0; l < nStates; l++) {
		message[l] = min(message[l], Min);
	}
	//////////////////////////////////////////
#endif
	//
#if 1
	//////////////////////////////////////////
	REALV s, T;
	REALV v_fx, delta_1; //delta_f
	int k0, k1;
	//
	s = c_alpha_O1;
	T = c_d_O1;
	//////////////////////////////////////////
	for (k1 = 0; k1 < nStates; k1++) {
		delta_1 = min(s * fabs(d0+c_disp_e[0]-c_disp_e[k1]), T) + message_org[0];
		for (k0 = 0; k0 < nStates; k0++) {
			v_fx = min(s * fabs(d0+c_disp_e[k0]-c_disp_e[k1]), T) + message_org[k0];
			TRUNCATE(delta_1, v_fx);
		}
		message[k1] = delta_1;
	}
	//////////////////////////////////////////
#endif
	//
#if 0
	//////////////////////////////////////////
	REALV s, T;
	REALV _2s, xv;
	REALV fx0_min, fx0_min_T;
	int k, k0, k1;
	REALV zv[MAX_K+2];
	int v[MAX_K];
	REALV v_fx, delta_1;
	//
	s = c_alpha;
	T = c_d;
	_2s = 0.5f / s;
	//////////////////////////////////////////
	fx0_min = message_org[0];
	for (k0 = 1; k0 < nStates; k0++) {
		TRUNCATE(fx0_min, message_org[k0]);
	}
	fx0_min_T = fx0_min + T;
	//////////////////////////////////////////
	// DT
	k = 0;
	v[0] = 0;
	zv[0] = -INFINITE_S;
	zv[1] = INFINITE_S;
	for (k1 = 1; k1 < nStates; k1++) {
		xv = ((message_org[k1] + s*c_disp_e[k1]) - (message_org[v[k]] - s*c_disp_e[v[k]])) * _2s;
		if (xv > c_disp_e[v[k]] && xv < c_disp_e[k1]) {
			if (xv <= zv[k]) {
				k--;
				//
				k1--;
				continue;
			} else {
				k++;
				//
				v[k] = k1;
				zv[k] = xv;
				zv[k+1] = INFINITE_S;
			}
		} else if ((xv == c_disp_e[v[k]]) || (xv == c_disp_e[k1])) {
			if (message_org[k1] < message_org[v[k]]) {
				v[k] = k1;
				zv[k+1] = INFINITE_S;
			}
		} else {
			if (k == 0) {
				if (message_org[k1] < message_org[v[0]]) {
					v[0] = k1;
					zv[0] = -INFINITE_S;
					zv[1] = INFINITE_S;
				}
			} else {
				if (message_org[k1] < message_org[v[k]]) {
					k--;
					//
					k1--;
					continue;
				}
			}
		}
	}
	k = 0;
	for (k1 = 0; k1 < nStates; k1++) {
		while (zv[k+1] < -d0 + c_disp_e[k1]) {
			k++;
		}
		delta_1 = min(s * fabs(d0+c_disp_e[v[k]]-c_disp_e[k1]), T) + message_org[v[k]];
		// apply truncation
		message[k1] = min(delta_1, fx0_min_T);
	}
	//////////////////////////////////////////
#endif
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
// Update Message for BP
///////////////////////////////////////////////////////////////////////////////////////
__device__ static void cu_UpdateSpatialMessageD(int x, int y, int z, int direction,
	int cx, int cy, int cz, int ox, int oy, int oz, int mx, int my, int mz,
	REALV* dOffset, REALV* dRangeTerm, REALV** dSO1F2Message, REALV* dDualMessage)
{
	int x1, y1, z1;
	REALV* message_org;
	//REALV message_org[MAX_K];
	REALV* message;
	size_t idx, idx1;
	REALV d0, Min;
	REALV* c_disp_e;
	//
	extern __shared__ REALV sbuf[];

	// eliminate impossible messages
	if (direction == 0 && x == c_mesh_x-1) { return; }
	if (direction == 1 && x == 0         ) { return; }
	if (direction == 2 && y == c_mesh_y-1) { return; }
	if (direction == 3 && y == 0         ) { return; }
	if (direction == 4 && z == c_mesh_z-1) { return; }
	if (direction == 5 && z == 0         ) { return; }

	x1 = x; y1 = y; z1 = z; // get the destination
	switch (direction) {
	case 0: 
		x1++; 
		c_disp_e = c_disp_ex;
		break;
	case 1: 
		x1--; 
		c_disp_e = c_disp_ex;
		break;
	case 2: 
		y1++; 
		c_disp_e = c_disp_ey;
		break;
	case 3: 
		y1--; 
		c_disp_e = c_disp_ey;
		break;
	case 4: 
		z1++; 
		c_disp_e = c_disp_ez;
		break;
	case 5: 
		z1--; 
		c_disp_e = c_disp_ez;
		break;
	}

	idx  = ((cz   )*my + (cy   ))*mx + (cx   );
	idx1 = ((z1-oz)*my + (y1-oy))*mx + (x1-ox);

	message_org = &sbuf[(threadIdx.y * blockDim.x + threadIdx.x)*c_K];
	message     = &dSO1F2Message[direction][idx1*c_K];

	cu_Add2MessageSpatial_O1F2(message_org, x, y, z, direction, c_K, idx, dRangeTerm, dSO1F2Message, NULL, dDualMessage);

	__syncthreads();

#ifdef O1_USE_OFFSET
	d0 = dOffset[idx] - dOffset[idx1];
#else
	d0 = 0;
#endif

	cu_ComputeSpatialMessageDT(message, message_org, message_org, x, y, z, d0, c_K, c_nL, c_disp_e);
	// normalize the message by subtracting the minimum value
	cu_SubtractMin(message, c_K, Min);
}

__global__ static void cu_UpdateSpatialMessage(unsigned int blocksInY, float invBlocksInY, int bw, int ox, int oy, int oz, int mx, int my, int mz,
	REALV* dOffset, REALV* dRangeTerm, REALV** dSO1F2Message, REALV* dDualMessage)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x, y, z;

	//cuPrintf("\tbx = %3d, by = %3d, bz = %3d, tx = %3d, ty = %3d, tz = %3d, idx = (%3d, %3d, %3d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, cx, cy, cz);

	x = cx + ox;
	y = cy + oy;
	z = cz + oz;

	if (bw == 0) {
		cu_UpdateSpatialMessageD(x, y, z, 0, cx, cy, cz, ox, oy, oz, mx, my, mz, dOffset, dRangeTerm, dSO1F2Message, dDualMessage);

		__syncthreads();

		cu_UpdateSpatialMessageD(x, y, z, 2, cx, cy, cz, ox, oy, oz, mx, my, mz, dOffset, dRangeTerm, dSO1F2Message, dDualMessage);

		__syncthreads();

		cu_UpdateSpatialMessageD(x, y, z, 4, cx, cy, cz, ox, oy, oz, mx, my, mz, dOffset, dRangeTerm, dSO1F2Message, dDualMessage);
	} else {
		cu_UpdateSpatialMessageD(x, y, z, 1, cx, cy, cz, ox, oy, oz, mx, my, mz, dOffset, dRangeTerm, dSO1F2Message, dDualMessage);

		__syncthreads();

		cu_UpdateSpatialMessageD(x, y, z, 3, cx, cy, cz, ox, oy, oz, mx, my, mz, dOffset, dRangeTerm, dSO1F2Message, dDualMessage);

		__syncthreads();

		cu_UpdateSpatialMessageD(x, y, z, 5, cx, cy, cz, ox, oy, oz, mx, my, mz, dOffset, dRangeTerm, dSO1F2Message, dDualMessage);
	}
}

__global__ static void cu_UpdateSpatialMessage_O2F3(unsigned int blocksInY, float invBlocksInY, int dir1, int ox, int oy, int oz, int mx, int my, int mz,
	REALV* dRangeTerm, REALV** dSO2, REALV** dSO2F3Message, REALV* dDualMessage)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x1, y1, z1;
	size_t idx0, idx1, idx2;
	//
	extern __shared__ REALV sbuf[];

	x1 = cx + ox;
	y1 = cy + oy;
	z1 = cz + oz;

	int x0, y0, z0, dir0;
	int x2, y2, z2, dir2;
	REALV *m_fx0, *m_fx1, *m_fx2;
	REALV *m_fx0_u, *m_fx1_u, *m_fx2_u;
	REALV Y0[MAX_L8_1], Y1[MAX_L8_1], Y2[MAX_L8_1]; //Yf[MAX_L8_1]
	REALV *D;
	int y, k0, k1, k2, K, L, L2, L5, L6, L7, L8, L8_1; //yy
	REALV delta_0, delta_1, delta_2, v_fx;

	K = c_K;
	L = c_L;
	L2 = c_L2;
	L5 = c_L5;
	L6 = c_L6;
	L7 = c_L7;
	L8 = c_L8;
	L8_1 = c_L8_1;

	// eliminate impossible messages
	if (dir1 == 1 && (x1 <= 0 || x1 >= c_mesh_x-1)) { return; }
	if (dir1 == 4 && (y1 <= 0 || y1 >= c_mesh_y-1)) { return; }
	if (dir1 == 7 && (z1 <= 0 || z1 >= c_mesh_z-1)) { return; }

	switch (dir1) {
	case 1: 
		x0 = x1-1; y0 = y1  ; z0 = z1  ; dir0 = 0;
		x2 = x1+1; y2 = y1  ; z2 = z1  ; dir2 = 2;
		//
		idx0 = ((cz  )*my + (cy  ))*mx + (cx-1);
		idx1 = ((cz  )*my + (cy  ))*mx + (cx  );
		idx2 = ((cz  )*my + (cy  ))*mx + (cx+1);
		//
		D = &dSO2[0][idx1*L8_1];
		break;
	case 4:
		x0 = x1  ; y0 = y1-1; z0 = z1  ; dir0 = 3;
		x2 = x1  ; y2 = y1+1; z2 = z1  ; dir2 = 5;
		//
		idx0 = ((cz  )*my + (cy-1))*mx + (cx  );
		idx1 = ((cz  )*my + (cy  ))*mx + (cx  );
		idx2 = ((cz  )*my + (cy+1))*mx + (cx  );
		//
		D = &dSO2[1][idx1*L8_1];
		break;
	case 7:
		x0 = x1  ; y0 = y1  ; z0 = z1-1; dir0 = 6;
		x2 = x1  ; y2 = y1  ; z2 = z1+1; dir2 = 8;
		//
		idx0 = ((cz-1)*my + (cy  ))*mx + (cx  );
		idx1 = ((cz  )*my + (cy  ))*mx + (cx  );
		idx2 = ((cz+1)*my + (cy  ))*mx + (cx  );
		//
		D = &dSO2[2][idx1*L8_1];
		break;
	}

	int tidx = threadIdx.y * blockDim.x + threadIdx.x;
	int tsize_e = 3*K;

	m_fx0_u = &sbuf[tidx*tsize_e + 0*K];
	m_fx1_u = &sbuf[tidx*tsize_e + 1*K];
	m_fx2_u = &sbuf[tidx*tsize_e + 2*K];

	cu_Add2MessageSpatial_O2F3(m_fx0_u, x0, y0, z0, dir0, K, idx0, dRangeTerm, NULL, dSO2F3Message, dDualMessage);
	cu_Add2MessageSpatial_O2F3(m_fx1_u, x1, y1, z1, dir1, K, idx1, dRangeTerm, NULL, dSO2F3Message, dDualMessage);
	cu_Add2MessageSpatial_O2F3(m_fx2_u, x2, y2, z2, dir2, K, idx2, dRangeTerm, NULL, dSO2F3Message, dDualMessage);

	__syncthreads();

	m_fx0 = &dSO2F3Message[dir0][idx0*K];
	m_fx1 = &dSO2F3Message[dir1][idx1*K];
	m_fx2 = &dSO2F3Message[dir2][idx2*K];

	//////////////////////////////////////////
	for (y = 0; y <= L8; y++) {
		Y0[y] = INFINITE_S;
		Y1[y] = INFINITE_S;
		Y2[y] = INFINITE_S;
	}

	// make y = L4 when k0 = L, k1 = L, k2 = L

	// y = -2*x1 + x2 = -2*(k1-L) + (k2-L) = 2*k1 - k2 + L + (4*L)
	// y = [L, L7]
	for (k2 = 0; k2 < K; k2++) {
		for (k1 = 0; k1 < K; k1++) {
			y = -2*k1 + k2 + L5;
			delta_0 = m_fx1_u[k1] + m_fx2_u[k2];
			TRUNCATE(Y0[y], delta_0);
		}
	}
	// y = x0 + x2 = (k0-L) + (k2-L) = k0 + k2 - 2*L + (4*L)
	// y = [L2, L6]
	for (k2 = 0; k2 < K; k2++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 + k2 + L2;
			delta_1 = m_fx0_u[k0] + m_fx2_u[k2];
			TRUNCATE(Y1[y], delta_1);
		}
	}
	// y = x0 - 2*x1 = (k0-L) - 2*(k1-L) = k0 - 2*k1 + L + (4*L)
	// y = [L, L7]
	for (k1 = 0; k1 < K; k1++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 - 2*k1 + L5;
			delta_2 = m_fx0_u[k0] + m_fx1_u[k1];
			TRUNCATE(Y2[y], delta_2);
		}
	}
	//////////////////////////////////////////

	//////////////////////////////////////////
	// Calculating messages
	//////////////////////////////////////////
	for (k0 = 0; k0 < K; k0++) {
		delta_0 = D[L+k0-L] + Y0[L];
		for (y = L; y <= L7; y++) {
			v_fx = D[y+k0-L] + Y0[y];
			TRUNCATE(delta_0, v_fx);
		}
		m_fx0_u[k0] = delta_0;
	}
	cu_SubtractMin(m_fx0_u, m_fx0, K, delta_0);
	//////////////////////////////////////////
	for (k1 = 0; k1 < K; k1++) {
		delta_1 = D[L2-2*k1+L2] + Y1[L2];
		for (y = L2; y <= L6; y++) {
			v_fx = D[y-2*k1+L2] + Y1[y];
			TRUNCATE(delta_1, v_fx);
		}
		m_fx1_u[k1] = delta_1;
	}
	cu_SubtractMin(m_fx1_u, m_fx1, K, delta_1);
	//////////////////////////////////////////
	for (k2 = 0; k2 < K; k2++) {
		delta_2 = D[L+k2-L] + Y2[L];
		for (y = L; y <= L7; y++) {
			v_fx = D[y+k2-L] + Y2[y];
			TRUNCATE(delta_2, v_fx);
		}
		m_fx2_u[k2] = delta_2;
	}
	cu_SubtractMin(m_fx2_u, m_fx2, K, delta_2);
	//////////////////////////////////////////
}

__global__ static void cu_UpdateDualMessage(unsigned int blocksInY, float invBlocksInY,
	int ox, int oy, int oz, int mx, int my, int mz,
	REALV* ddcv, REALV** dRangeTerm, REALV*** dSO1F2Message, REALV*** dSO2F3Message, REALV** dDualMessage)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x, y, z;
	size_t cidx, idx;
	//
	REALV *m_fx0_b, *m_fx1_b, *m_fx2_b;
	REALV m_fx0_u[MAX_K], m_fx1_u[MAX_K], m_fx2_u[MAX_K];
	REALV *m_fx0, *m_fx1, *m_fx2;
	REALV v_fx, delta_0, delta_1, delta_2;
	int k0, k1, k2, kk, k2_K_2, k1_K, K, K_2;
	//
	extern __shared__ REALV sbuf[];

	x = cx + ox;
	y = cy + oy;
	z = cz + oz;

	idx  = (z*c_mesh_y + y)*c_mesh_x + x;
	cidx = (cz*my + cy)*mx + cx;

	K = c_K;
	K_2 = c_K * c_K;

	int tidx = threadIdx.y * blockDim.x + threadIdx.x;
	int tsize_e = 3*K;
	m_fx0_b = &sbuf[tidx*tsize_e + 0*K];
	m_fx1_b = &sbuf[tidx*tsize_e + 1*K];
	m_fx2_b = &sbuf[tidx*tsize_e + 2*K];

	//////////////////////////////////////////
	//////////////////////////////////////////
	cu_Add2MessageDual(m_fx0_b, x, y, z, K, idx, dRangeTerm[0], dSO1F2Message[0], dSO2F3Message[0]);
	cu_Add2MessageDual(m_fx1_b, x, y, z, K, idx, dRangeTerm[1], dSO1F2Message[1], dSO2F3Message[1]);
	cu_Add2MessageDual(m_fx2_b, x, y, z, K, idx, dRangeTerm[2], dSO1F2Message[2], dSO2F3Message[2]);
	//////////////////////////////////////////
	//////////////////////////////////////////

	__syncthreads();

	//////////////////////////////////////////
	//////////////////////////////////////////
#if 1
	REALV *Dm;
	Dm = &ddcv[cidx*c_num_d];
#else
	REALV *D;
	REALV *Dm;
	D = &ddcv[idx*num_d];
	Dm = &sbuf[tidx*tsize_e + 6*K];
	memcpy(Dm, D, num_d * sizeof(REALV));
#endif
	m_fx0 = &dDualMessage[0][idx*K];
	m_fx1 = &dDualMessage[1][idx*K];
	m_fx2 = &dDualMessage[2][idx*K];

	for (k0 = 0; k0 < K; k0++) {
		delta_0 = INFINITE_S;
		for (k2 = 0; k2 < K; k2++) {
			kk = k2*K_2 + k0;
			for (k1 = 0; k1 < K; k1++) {
				v_fx = Dm[kk + k1*K] + m_fx1_b[k1] + m_fx2_b[k2];
				TRUNCATE(delta_0, v_fx);
			}
		}
		m_fx0_u[k0] = delta_0;
	}

	for (k1 = 0; k1 < K; k1++) {
		delta_1 = INFINITE_S;
		k1_K = k1 * K;
		for (k2 = 0; k2 < K; k2++) {
			kk = k2*K_2 + k1_K;
			for (k0 = 0; k0 < K; k0++) {
				v_fx = Dm[kk + k0] + m_fx0_b[k0] + m_fx2_b[k2];
				TRUNCATE(delta_1, v_fx);
			}
		}
		m_fx1_u[k1] = delta_1;
	}

	for (k2 = 0; k2 < K; k2++) {
		delta_2 = INFINITE_S;
		k2_K_2 = k2 * K_2;
		for (k1 = 0; k1 < K; k1++) {
			kk = k2_K_2 + k1 * K;
			for (k0 = 0; k0 < K; k0++) {
				v_fx = Dm[kk + k0] + m_fx0_b[k0] + m_fx1_b[k1];
				TRUNCATE(delta_2, v_fx);
			}
		}
		m_fx2_u[k2] = delta_2;
	}
	//////////////////////////////////////////
	//////////////////////////////////////////

	//////////////////////////////////////////
	//////////////////////////////////////////
	//if (plane != 0) {
		cu_SubtractMin(m_fx0_u, m_fx0, K, delta_0);
	//}
	//////////////////////////////////////////
	//if (plane != 1) {
		cu_SubtractMin(m_fx1_u, m_fx1, K, delta_1);
	//}
	//////////////////////////////////////////
	//if (plane != 2) {
		cu_SubtractMin(m_fx2_u, m_fx2, K, delta_2);
	//}
	//////////////////////////////////////////
	//////////////////////////////////////////
}

extern "C"
void cu_BP_S(int iter, REALV**** pdcv, REALV**** pOffset[3], REALV**** pRangeTerm[3], REALV**** pSO1[3][3], REALV**** pSO2[3][3],
	REALV**** pSO1F2Message[3][6], REALV**** pSO2F3Message[3][9], REALV**** pDualMessage[3], int iterPrev)
{
	int i, j, l, it;

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
#else
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
#endif

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	if (iterPrev == 0) {
		if (mmode == 0) {
			cu_VolInit(ddcv, pdcv, 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, num_d);
		}
		for (i = 0; i < 3; i++) {
			cu_VolInit(hOffset[i], pOffset[i], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, 1);
			cu_VolInit(hRangeTerm[i], pRangeTerm[i], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
			if (in_scv_w_O1F2 != -2) {
				if (mmode == 0) {
					for (j = 0; j < 3; j++) {
						cu_VolInit(hSO1[i][j], pSO1[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, L4_1);
					}
				}
				for (j = 0; j < 6; j++) {
					cu_VolInit(hSO1F2Message[i][j], pSO1F2Message[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
				}
			}
			if (in_scv_w_O2F3 != -2) {
				if (mmode == 0) {
					for (j = 0; j < 3; j++) {
						cu_VolInit(hSO2[i][j], pSO2[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, L8_1);
					}
				}
				for (j = 0; j < 9; j++) {
					cu_VolInit(hSO2F3Message[i][j], pSO2F3Message[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
				}
			}
			cu_VolInit(hDualMessage[i], pDualMessage[i], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	cutStopTimer(timer);
	TRACE2("h->d time = %f\n", cutGetTimerValue(timer));
	cutDeleteTimer(timer);

	cutCreateTimer(&timer);
	cutStartTimer(timer);
#else
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	TRACE2("h->d time = %f\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
#endif

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	//cudaPrintfInit(blocksInX*blocksInY*blocksInZ*threadsInX*threadsInY*threadsInZ*256);

	for (it = 0; it < iter; it++) {
		///////////////////////////////////////////////////////////////////////////////////////
		if (in_scv_w_O1F2 != -2) {
			for (l = 0; l < 3; l++) {
				cu_UpdateSpatialMessage<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0, 0, mx, my, mz, hOffset[l], hRangeTerm[l], ddSO1F2Message[l], hDualMessage[l]);
				//
				//FILE* fp = fopen("culog.txt", "w");
				//cudaPrintfDisplay(fp, false);
				//fflush(fp);
				//fclose(fp);
			}
		} else if (in_scv_w_O2F3 != -2) {
			for (l = 0; l < 3; l++) {
				cu_UpdateSpatialMessage_O2F3<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 1, 0, 0, 0, mx, my, mz, hRangeTerm[l], ddSO2[l], ddSO2F3Message[l], hDualMessage[l]);
				cu_UpdateSpatialMessage_O2F3<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 4, 0, 0, 0, mx, my, mz, hRangeTerm[l], ddSO2[l], ddSO2F3Message[l], hDualMessage[l]);
				cu_UpdateSpatialMessage_O2F3<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 7, 0, 0, 0, mx, my, mz, hRangeTerm[l], ddSO2[l], ddSO2F3Message[l], hDualMessage[l]);
			}
		}
		cudaDeviceSynchronize();
		///////////////////////////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////////////////////////
		if (in_scv_w_O1F2 != -2) {
			for (l = 0; l < 3; l++) {
				cu_UpdateSpatialMessage<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 1, 0, 0, 0, mx, my, mz, hOffset[l], hRangeTerm[l], ddSO1F2Message[l], hDualMessage[l]);
			}
		} else if (in_scv_w_O2F3 != -2) {
		}
		cudaDeviceSynchronize();
		///////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////////
		//*
		if (mmode == 0) {
			cu_UpdateDualMessage<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0, mx, my, mz, ddcv, dRangeTerm, dSO1F2Message, dSO2F3Message, dDualMessage);
		} else if (mmode == 1) {
			size_t msize_4 = msize / 4;
			int mz_4  =     mz / 4;
			int mz_42 = 2 * mz / 4;
			int mz_43 = 3 * mz / 4;
			//
			#if 0
			{
				size_t free_mem, total_mem, req_mem;
				CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
				req_mem = msize_4 * num_d * sizeof(REALV);
				if (req_mem > free_mem) {
					TRACE("req_mem = %u, free_mem = %u\n", req_mem, free_mem);
					return;
				}
			}
			#endif
			//
			CUDA_SAFE_CALL(cudaMalloc(&ddcv, msize_4 * num_d * sizeof(REALV)));
			//
			cu_VolInit(ddcv, pdcv, 0, 0, 0    , mx, my, mz_4, mesh_x, mesh_y, mesh_z, num_d);
			cu_UpdateDualMessage<<<Dg_4, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0    , mx, my, mz_4, ddcv, dRangeTerm, dSO1F2Message, dSO2F3Message, dDualMessage);
			//
			cu_VolInit(ddcv, pdcv, 0, 0, mz_4 , mx, my, mz_4, mesh_x, mesh_y, mesh_z, num_d);
			cu_UpdateDualMessage<<<Dg_4, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, mz_4 , mx, my, mz_4, ddcv, dRangeTerm, dSO1F2Message, dSO2F3Message, dDualMessage);
			//
			cu_VolInit(ddcv, pdcv, 0, 0, mz_42, mx, my, mz_4, mesh_x, mesh_y, mesh_z, num_d);
			cu_UpdateDualMessage<<<Dg_4, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, mz_42, mx, my, mz_4, ddcv, dRangeTerm, dSO1F2Message, dSO2F3Message, dDualMessage);
			//
			cu_VolInit(ddcv, pdcv, 0, 0, mz_43, mx, my, mz_4, mesh_x, mesh_y, mesh_z, num_d);
			cu_UpdateDualMessage<<<Dg_4, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, mz_43, mx, my, mz_4, ddcv, dRangeTerm, dSO1F2Message, dSO2F3Message, dDualMessage);
			//
			cudaFree(ddcv);
		}
		//*/
		cudaDeviceSynchronize();
		///////////////////////////////////////////////////////////////////////////////////////

		if (it < iter-1) {
			TRACE2("iter %03d\n", it+iterPrev);
		}
	}

	//cudaPrintfEnd();
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	cutStopTimer(timer);
	TRACE2("avg time = %f\n", cutGetTimerValue(timer) / iter);
	cutDeleteTimer(timer);

	cutCreateTimer(&timer);
	cutStartTimer(timer);
#else
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	TRACE2("avg time = %f\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
#endif

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	for (i = 0; i < 3; i++) {
		if (in_scv_w_O1F2 != -2) {
			for (j = 0; j < 6; j++) {
				cu_VolCopy(hSO1F2Message[i][j], pSO1F2Message[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
			}
		}
		if (in_scv_w_O2F3 != -2) {
			for (j = 0; j < 9; j++) {
				cu_VolCopy(hSO2F3Message[i][j], pSO2F3Message[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
			}
		}
		cu_VolCopy(hDualMessage[i], pDualMessage[i], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
	}
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	cutStopTimer(timer);
	TRACE2("d->h time = %f\n", cutGetTimerValue(timer));
	cutDeleteTimer(timer);
#else
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	TRACE2("d->h time = %f\n", elapsedTime);
   	cudaEventDestroy(start);
	cudaEventDestroy(stop); 
#endif
#endif
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
// Update Message for TRW_S
///////////////////////////////////////////////////////////////////////////////////////
__global__ static void cu_UpdateMessage_TRW_S_FW_O1F2(unsigned int blocksInY, float invBlocksInY, int ox, int oy, int oz, int mx, int my, int mz,
	REALV* dRangeTerm, REALV** dSO1F2Message, REALV* dDualMessage)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x, y, z;
	int K;
	size_t idx, idxK;
	//
	extern __shared__ REALV sbuf[];

	x = cx + ox;
	y = cy + oy;
	z = cz + oz;

	idx  = (cz*my + cy)*mx + cx;
	K = c_K;
	idxK = idx * K;

	int i, k;
	BOOL update_s[6];
	REALV *m_fx;
	REALV *m_fx_b;
	REALV *m_fx_u;
	int ns;
	REALV r, vMin;

	m_fx_b = &sbuf[(threadIdx.y * blockDim.x + threadIdx.x)*2*K + 0*K];
	m_fx_u = &sbuf[(threadIdx.y * blockDim.x + threadIdx.x)*2*K + 1*K];

	// add the range term
	memcpy(m_fx_b, &dRangeTerm[idxK], K * sizeof(REALV));

	// add spatial messages
	for (i = 0; i < 6; i++) {
		update_s[i] = TRUE;
	}
	ns = 7;
	if (x > 0) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[0][idxK], K);
	} else {
		update_s[0] = FALSE; ns--;
	}
	if (x < c_mesh_x-1) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[1][idxK], K);
	} else {
		update_s[1] = FALSE; ns--;
	}
	if (y > 0) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[2][idxK], K);
	} else {
		update_s[2] = FALSE; ns--;
	}
	if (y < c_mesh_y-1) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[3][idxK], K);
	} else {
		update_s[3] = FALSE; ns--;
	}
	if (z > 0) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[4][idxK], K);
	} else {
		update_s[4] = FALSE; ns--;
	}
	if (z < c_mesh_z-1) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[5][idxK], K);
	} else {
		update_s[5] = FALSE; ns--;
	}
	r = 1.0 / ns;

	cu_Add2Message(m_fx_b, &dDualMessage[idxK], K);

	for (i = 0; i < 6; i++) {
		if (!update_s[i]) { continue; }

		m_fx = &dSO1F2Message[i][idxK];

		for (k = 0; k < K; k++) {
			m_fx_u[k] = r * m_fx_b[k] - m_fx[k];
		}
		vMin = m_fx_u[0];
		for (k = 1; k < K; k++) {
			TRUNCATE(vMin, m_fx_u[k]);
		}
		for (k = 0; k < K; k++) {
			m_fx[k] = m_fx_u[k] - vMin;
		}
	}
	{
		m_fx = &dDualMessage[idxK];

		for (k = 0; k < K; k++) {
			m_fx_u[k] = r * m_fx_b[k] - m_fx[k];
		}
		vMin = m_fx_u[0];
		for (k = 1; k < K; k++) {
			TRUNCATE(vMin, m_fx_u[k]);
		}
		for (k = 0; k < K; k++) {
			m_fx[k] = m_fx_u[k] - vMin;
		}
	}
}

__global__ static void cu_UpdateMessage_TRW_S_FW_O2F3(unsigned int blocksInY, float invBlocksInY, int ox, int oy, int oz, int mx, int my, int mz,
	REALV* dRangeTerm, REALV** dSO2F3Message, REALV* dDualMessage)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x, y, z;
	int K;
	size_t idx, idxK;
	//
	extern __shared__ REALV sbuf[];

	x = cx + ox;
	y = cy + oy;
	z = cz + oz;

	idx  = (cz*my + cy)*mx + cx;
	K = c_K;
	idxK = idx * K;

	int i, k;
	BOOL update_s[9];
	REALV *m_fx;
	REALV *m_fx_b;
	REALV *m_fx_u;
	int ns;
	REALV r, vMin;

	m_fx_b = &sbuf[(threadIdx.y * blockDim.x + threadIdx.x)*2*K + 0*K];
	m_fx_u = &sbuf[(threadIdx.y * blockDim.x + threadIdx.x)*2*K + 1*K];

	// add the range term
	memcpy(m_fx_b, &dRangeTerm[idxK], K * sizeof(REALV));

	// add spatial messages
	for (i = 0; i < 9; i++) {
		update_s[i] = TRUE;
	}
	ns = 10;
	if (x < c_mesh_x-2) {				// f+ -> x
		cu_Add2Message(m_fx_b, &dSO2F3Message[0][idxK], K);
	} else {
		update_s[0] = FALSE; ns--;
	}
	if (x > 0 && x < c_mesh_x-1) {	// f0 -> x
		cu_Add2Message(m_fx_b, &dSO2F3Message[1][idxK], K);
	} else {
		update_s[1] = FALSE; ns--;
	}
	if (x > 1) {					// f- -> x
		cu_Add2Message(m_fx_b, &dSO2F3Message[2][idxK], K);
	} else {
		update_s[2] = FALSE; ns--;
	}
	if (y < c_mesh_y-2) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[3][idxK], K);
	} else {
		update_s[3] = FALSE; ns--;
	}
	if (y > 0 && y < c_mesh_y-1) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[4][idxK], K);
	} else {
		update_s[4] = FALSE; ns--;
	}
	if (y > 1) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[5][idxK], K);
	} else {
		update_s[5] = FALSE; ns--;
	}
	if (z < c_mesh_z-2) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[6][idxK], K);
	} else {
		update_s[6] = FALSE; ns--;
	}
	if (z > 0 && z < c_mesh_z-1) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[7][idxK], K);
	} else {
		update_s[7] = FALSE; ns--;
	}
	if (z > 1) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[8][idxK], K);
	} else {
		update_s[8] = FALSE; ns--;
	}

	r = 1.0 / ns;

	cu_Add2Message(m_fx_b, &dDualMessage[idxK], K);

	for (i = 0; i < 9; i++) {
		if (!update_s[i]) { continue; }

		m_fx = &dSO2F3Message[i][idxK];

		for (k = 0; k < K; k++) {
			m_fx_u[k] = r * m_fx_b[k] - m_fx[k];
		}
		vMin = m_fx_u[0];
		for (k = 1; k < K; k++) {
			TRUNCATE(vMin, m_fx_u[k]);
		}
		for (k = 0; k < K; k++) {
			m_fx[k] = m_fx_u[k] - vMin;
		}
	}
	{
		m_fx = &dDualMessage[idxK];

		for (k = 0; k < K; k++) {
			m_fx_u[k] = r * m_fx_b[k] - m_fx[k];
		}
		vMin = m_fx_u[0];
		for (k = 1; k < K; k++) {
			TRUNCATE(vMin, m_fx_u[k]);
		}
		for (k = 0; k < K; k++) {
			m_fx[k] = m_fx_u[k] - vMin;
		}
	}
}

__global__ static void cu_UpdateMessage_TRW_S_FW_O1F2_O2F3(unsigned int blocksInY, float invBlocksInY, int ox, int oy, int oz, int mx, int my, int mz,
	REALV* dRangeTerm, REALV** dSO1F2Message, REALV** dSO2F3Message, REALV* dDualMessage)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x, y, z;
	int K;
	size_t idx, idxK;
	//
	extern __shared__ REALV sbuf[];

	x = cx + ox;
	y = cy + oy;
	z = cz + oz;

	idx  = (cz*my + cy)*mx + cx;
	K = c_K;
	idxK = idx * K;

	int i, k;
	BOOL update_s_O1[9];
	BOOL update_s_O2[9];
	REALV *m_fx;
	REALV *m_fx_b;
	REALV *m_fx_u;
	int ns_O1;
	int ns_O2;
	REALV r, vMin;

	m_fx_b = &sbuf[(threadIdx.y * blockDim.x + threadIdx.x)*2*K + 0*K];
	m_fx_u = &sbuf[(threadIdx.y * blockDim.x + threadIdx.x)*2*K + 1*K];

	// add the range term
	memcpy(m_fx_b, &dRangeTerm[idxK], K * sizeof(REALV));

	// add spatial messages for O1
	for (i = 0; i < 6; i++) {
		update_s_O1[i] = TRUE;
	}
	ns_O1 = 6;
	if (x > 0) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[0][idxK], K);
	} else {
		update_s_O1[0] = FALSE; ns_O1--;
	}
	if (x < c_mesh_x-1) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[1][idxK], K);
	} else {
		update_s_O1[1] = FALSE; ns_O1--;
	}
	if (y > 0) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[2][idxK], K);
	} else {
		update_s_O1[2] = FALSE; ns_O1--;
	}
	if (y < c_mesh_y-1) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[3][idxK], K);
	} else {
		update_s_O1[3] = FALSE; ns_O1--;
	}
	if (z > 0) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[4][idxK], K);
	} else {
		update_s_O1[4] = FALSE; ns_O1--;
	}
	if (z < c_mesh_z-1) {
		cu_Add2Message(m_fx_b, &dSO1F2Message[5][idxK], K);
	} else {
		update_s_O1[5] = FALSE; ns_O1--;
	}

	// add spatial messages for O2
	for (i = 0; i < 9; i++) {
		update_s_O2[i] = TRUE;
	}
	ns_O2 = 9;
	if (x < c_mesh_x-2) {				// f+ -> x
		cu_Add2Message(m_fx_b, &dSO2F3Message[0][idxK], K);
	} else {
		update_s_O2[0] = FALSE; ns_O2--;
	}
	if (x > 0 && x < c_mesh_x-1) {	// f0 -> x
		cu_Add2Message(m_fx_b, &dSO2F3Message[1][idxK], K);
	} else {
		update_s_O2[1] = FALSE; ns_O2--;
	}
	if (x > 1) {					// f- -> x
		cu_Add2Message(m_fx_b, &dSO2F3Message[2][idxK], K);
	} else {
		update_s_O2[2] = FALSE; ns_O2--;
	}
	if (y < c_mesh_y-2) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[3][idxK], K);
	} else {
		update_s_O2[3] = FALSE; ns_O2--;
	}
	if (y > 0 && y < c_mesh_y-1) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[4][idxK], K);
	} else {
		update_s_O2[4] = FALSE; ns_O2--;
	}
	if (y > 1) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[5][idxK], K);
	} else {
		update_s_O2[5] = FALSE; ns_O2--;
	}
	if (z < c_mesh_z-2) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[6][idxK], K);
	} else {
		update_s_O2[6] = FALSE; ns_O2--;
	}
	if (z > 0 && z < c_mesh_z-1) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[7][idxK], K);
	} else {
		update_s_O2[7] = FALSE; ns_O2--;
	}
	if (z > 1) {
		cu_Add2Message(m_fx_b, &dSO2F3Message[8][idxK], K);
	} else {
		update_s_O2[8] = FALSE; ns_O2--;
	}

	r = 1.0 / (ns_O1 + ns_O2 + 1);

	cu_Add2Message(m_fx_b, &dDualMessage[idxK], K);

	for (i = 0; i < 6; i++) {
		if (!update_s_O1[i]) { continue; }

		m_fx = &dSO1F2Message[i][idxK];

		for (k = 0; k < K; k++) {
			m_fx_u[k] = r * m_fx_b[k] - m_fx[k];
		}
		vMin = m_fx_u[0];
		for (k = 1; k < K; k++) {
			TRUNCATE(vMin, m_fx_u[k]);
		}
		for (k = 0; k < K; k++) {
			m_fx[k] = m_fx_u[k] - vMin;
		}
	}
	for (i = 0; i < 9; i++) {
		if (!update_s_O2[i]) { continue; }

		m_fx = &dSO2F3Message[i][idxK];

		for (k = 0; k < K; k++) {
			m_fx_u[k] = r * m_fx_b[k] - m_fx[k];
		}
		vMin = m_fx_u[0];
		for (k = 1; k < K; k++) {
			TRUNCATE(vMin, m_fx_u[k]);
		}
		for (k = 0; k < K; k++) {
			m_fx[k] = m_fx_u[k] - vMin;
		}
	}
	{
		m_fx = &dDualMessage[idxK];

		for (k = 0; k < K; k++) {
			m_fx_u[k] = r * m_fx_b[k] - m_fx[k];
		}
		vMin = m_fx_u[0];
		for (k = 1; k < K; k++) {
			TRUNCATE(vMin, m_fx_u[k]);
		}
		for (k = 0; k < K; k++) {
			m_fx[k] = m_fx_u[k] - vMin;
		}
	}
}

__global__ static void cu_UpdateMessage_TRW_S_BW(unsigned int blocksInY, float invBlocksInY, int ox, int oy, int oz, int mx, int my, int mz,
	REALV* dRangeTerm, REALV** dSO1F2Message, REALV** dSO2F3Message, REALV* dDualMessage)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x, y, z;
	int K;
	size_t idx;
	//
	extern __shared__ REALV sbuf[];

	x = cx + ox;
	y = cy + oy;
	z = cz + oz;

	idx  = (cz*my + cy)*mx + cx;
	K = c_K;

	int k;
	REALV *m_fx_b;
	REALV vMin;

	m_fx_b = &sbuf[(threadIdx.y * blockDim.x + threadIdx.x)*K];

	cu_Add2MessageDual(m_fx_b, x, y, z, K, idx, dRangeTerm, dSO1F2Message, dSO2F3Message);

	cu_Add2Message(m_fx_b, &dDualMessage[idx*K], K);

	vMin = m_fx_b[0];
	for (k = 1; k < K; k++) {
		TRUNCATE(vMin, m_fx_b[k]);
	}

	//dLowerBound += vMin;
	//atomicAdd(&dLowerBound, 100);
	atomicAdd(&dLowerBound, (int)(vMin));

	//cuPrintf("\tbx = %3d, by = %3d, bz = %3d, tx = %3d, ty = %3d, tz = %3d, idx = (%3d, %3d, %3d), %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, cx, cy, cz, dLowerBound);
}

__global__ static void cu_UpdateSpatialMessage_TRW_S_BW_O1F2(unsigned int blocksInY, float invBlocksInY, int direction, int ox, int oy, int oz, int mx, int my, int mz,
	REALV* dOffset, REALV** dSO1, REALV** dSO1F2Message)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x, y, z;
	int K;
	size_t idx0, idx1;
	//
	extern __shared__ REALV sbuf[];

	x = cx + ox;
	y = cy + oy;
	z = cz + oz;

	K = c_K;

	int direction1;
	int x1, y1, z1;
	REALV *m_fx0, *m_fx1;
	REALV *m_buf;
	REALV delta_f, delta_0, delta_1;
	REALV* c_disp_e;

	x1 = x; y1 = y; z1 = z; // destination
	if (direction == 0) {
		if (x == c_mesh_x-1) { return; }
		x1++;
		c_disp_e = c_disp_ex;
		direction1 = 1;
	} else if (direction == 1) {
		if (x == 0       ) { return; }
		x1--;
		c_disp_e = c_disp_ex;
		direction1 = 0;
	} else if (direction == 2) {
		if (y == c_mesh_y-1) { return; }
		y1++;
		c_disp_e = c_disp_ey;
		direction1 = 3;
	} else if (direction == 3) {
		if (y == 0       ) { return; }
		y1--;
		c_disp_e = c_disp_ey;
		direction1 = 2;
	} else if (direction == 4) {
		if (z == c_mesh_z-1) { return; }
		z1++;
		c_disp_e = c_disp_ez;
		direction1 = 5;
	} else if (direction == 5) {
		if (z == 0       ) { return; }
		z1--;
		c_disp_e = c_disp_ez;
		direction1 = 4;
	}

	idx0 = (z *c_mesh_y + y )*c_mesh_x + x ;
	idx1 = (z1*c_mesh_y + y1)*c_mesh_x + x1;

	//////////////////////////////////////////
	//////////////////////////////////////////
	REALV d0;
#if 0
	REALV s, T;
#endif
	REALV *Y0, *Y1;
	REALV r_fx0, r_fx1;
	int k0, k1;

	m_fx0 = &dSO1F2Message[direction1][idx0*K];
	m_fx1 = &dSO1F2Message[direction ][idx1*K];
	r_fx0 = r_fx1 = 0.5;
#ifdef O1_USE_OFFSET
	d0 = dOffset[idx0] - dOffset[idx1];
#else
	d0 = 0;
#endif
#if 0
	s = c_alpha_O1;
	T = c_d_O1;
#endif

	//////////////////////////////////////////
#if 0
	delta_f = INFINITE_S;
	for (k1 = 0; k1 < K; k1++) {
		for (k0 = 0; k0 < K; k0++) {
			v_fx = min(s * fabs(d0+disp_e[k0]-disp_e[k1]), T) + m_fx0[k0] + m_fx1[k1];
			TRUNCATE(delta_f, v_fx);
		}
	}
#else
	delta_f = 0;
#endif
	//////////////////////////////////////////

	//////////////////////////////////////////
	int tidx = threadIdx.y * blockDim.x + threadIdx.x;
	int tsize_e = 3*K;

	m_buf = &sbuf[tidx*tsize_e + 0*K];
	Y0    = &sbuf[tidx*tsize_e + 1*K];
	Y1    = &sbuf[tidx*tsize_e + 2*K];
	//////////////////////////////////////////

#if 1
	//////////////////////////////////////////
	cu_ComputeSpatialMessageDT(Y0, m_fx1, m_buf, x, y, z, d0, K, c_nL, c_disp_e);
	for (k0 = 0; k0 < K; k0++) {
		Y0[k0] = r_fx0 * (Y0[k0]-delta_f) + (r_fx0-1)*m_fx0[k0];
	}
	delta_0 = Y0[0];
	for (k0 = 1; k0 < K; k0++) { 
		TRUNCATE(delta_0, Y0[k0]);
	}
	//
	cu_ComputeSpatialMessageDT(Y1, m_fx0, m_buf, x1, y1, z1, -d0, K, c_nL, c_disp_e);
	for (k1 = 0; k1 < K; k1++) {
		Y1[k1] = r_fx1 * (Y1[k1]-delta_f) + (r_fx1-1)*m_fx1[k1];
	}
	delta_1 = Y1[0];
	for (k1 = 1; k1 < K; k1++) { 
		TRUNCATE(delta_1, Y1[k1]);
	}
	//////////////////////////////////////////
#else
	//////////////////////////////////////////
	for (k0 = 0; k0 < K; k0++) {
		delta_0 = min(s * fabs(-d0+disp_e[k0]-disp_e[0]), T) + m_fx1[0];
		for (k1 = 0; k1 < K; k1++) {
			v_fx = min(s * fabs(-d0+disp_e[k0]-disp_e[k1]), T) + m_fx1[k1];
			TRUNCATE(delta_0, v_fx);
		}
		Y0[k0] = r_fx0 * (delta_0-delta_f) + (r_fx0-1)*m_fx0[k0];
	}
	delta_0 = Y0[0];
	for (k0 = 1; k0 < K; k0++) { 
		TRUNCATE(delta_0, Y0[k0]);
	}
	//
	for (k1 = 0; k1 < K; k1++) {
		delta_1 = min(s * fabs(-d0+disp_e[0]-disp_e[k1]), T) + m_fx0[0];
		for (k0 = 0; k0 < K; k0++) {
			v_fx = min(s * fabs(-d0+disp_e[k0]-disp_e[k1]), T) + m_fx0[k0];
			TRUNCATE(delta_1, v_fx);
		}
		Y1[k1] = r_fx1 * (delta_1-delta_f) + (r_fx1-1)*m_fx1[k1];
	}
	delta_1 = Y1[0];
	for (k1 = 1; k1 < K; k1++) { 
		TRUNCATE(delta_1, Y1[k1]);
	}
	//////////////////////////////////////////
#endif

	//////////////////////////////////////////
	for (k0 = 0; k0 < K; k0++) { 
		m_fx0[k0] = Y0[k0] - delta_0;
	}
	for (k1 = 0; k1 < K; k1++) { 
		m_fx1[k1] = Y1[k1] - delta_1;
	}
	//////////////////////////////////////////
	
	//////////////////////////////////////////
	//lowerBound += delta_f + delta_0 + delta_1;
	atomicAdd(&dLowerBound, (int)(delta_f + delta_0 + delta_1));
	//////////////////////////////////////////

	//////////////////////////////////////////
	//////////////////////////////////////////
}

__global__ static void cu_UpdateSpatialMessage_TRW_S_BW_O2F3(unsigned int blocksInY, float invBlocksInY, int dir1, int ox, int oy, int oz, int mx, int my, int mz,
	REALV** dSO2, REALV** dSO2F3Message)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x1, y1, z1;
	size_t idx0, idx1, idx2;
	//
	extern __shared__ REALV sbuf[];

	x1 = cx + ox;
	y1 = cy + oy;
	z1 = cz + oz;

#if 0
	int x0, y0, z0, dir0;
	int x2, y2, z2, dir2;
#else
	int dir0;
	int dir2;
#endif
	REALV *m_fx0, *m_fx1, *m_fx2;
	REALV *m_fx0_u, *m_fx1_u, *m_fx2_u;
	REALV Y0[MAX_L8_1], Y1[MAX_L8_1], Y2[MAX_L8_1], Yf[MAX_L8_1];
	REALV *D;
	int y, yy, k, k0, k1, k2, K, L, L2, L5, L6, L7, L8, L8_1;
	REALV delta_0, delta_1, delta_2, delta_f, v_fx;
	REALV r_fx0, r_fx1, r_fx2;

	K = c_K;
	L = c_L;
	L2 = c_L2;
	L5 = c_L5;
	L6 = c_L6;
	L7 = c_L7;
	L8 = c_L8;
	L8_1 = c_L8_1;

	// eliminate impossible messages
	if (dir1 == 1 && (x1 <= 0 || x1 >= c_mesh_x-1)) { return; }
	if (dir1 == 4 && (y1 <= 0 || y1 >= c_mesh_y-1)) { return; }
	if (dir1 == 7 && (z1 <= 0 || z1 >= c_mesh_z-1)) { return; }

	switch (dir1) {
	case 1: 
#if 0
		x0 = x1-1; y0 = y1  ; z0 = z1  ; dir0 = 0;
		x2 = x1+1; y2 = y1  ; z2 = z1  ; dir2 = 2;
#else
		dir0 = 0;
		dir2 = 2;
#endif
		//
		idx0 = ((cz  )*my + (cy  ))*mx + (cx-1);
		idx1 = ((cz  )*my + (cy  ))*mx + (cx  );
		idx2 = ((cz  )*my + (cy  ))*mx + (cx+1);
		//
		D = &dSO2[0][idx1*L8_1];
		break;
	case 4:
#if 0
		x0 = x1  ; y0 = y1-1; z0 = z1  ; dir0 = 3;
		x2 = x1  ; y2 = y1+1; z2 = z1  ; dir2 = 5;
#else
		dir0 = 3;
		dir2 = 5;
#endif
		//
		idx0 = ((cz  )*my + (cy-1))*mx + (cx  );
		idx1 = ((cz  )*my + (cy  ))*mx + (cx  );
		idx2 = ((cz  )*my + (cy+1))*mx + (cx  );
		//
		D = &dSO2[1][idx1*L8_1];
		break;
	case 7:
#if 0
		x0 = x1  ; y0 = y1  ; z0 = z1-1; dir0 = 6;
		x2 = x1  ; y2 = y1  ; z2 = z1+1; dir2 = 8;
#else
		dir0 = 6;
		dir2 = 8;
#endif
		//
		idx0 = ((cz-1)*my + (cy  ))*mx + (cx  );
		idx1 = ((cz  )*my + (cy  ))*mx + (cx  );
		idx2 = ((cz+1)*my + (cy  ))*mx + (cx  );
		//
		D = &dSO2[2][idx1*L8_1];
		break;
	}

	int tidx = threadIdx.y * blockDim.x + threadIdx.x;
	int tsize_e = 3*K;

	m_fx0_u = &sbuf[tidx*tsize_e + 0*K];
	m_fx1_u = &sbuf[tidx*tsize_e + 1*K];
	m_fx2_u = &sbuf[tidx*tsize_e + 2*K];

	m_fx0 = &dSO2F3Message[dir0][idx0*K];
	m_fx1 = &dSO2F3Message[dir1][idx1*K];
	m_fx2 = &dSO2F3Message[dir2][idx2*K];
	r_fx0 = r_fx1 = r_fx2 = 1.0 / 3.0;

	//////////////////////////////////////////
	for (y = 0; y <= L8; y++) {
		Y0[y] = INFINITE_S;
		Y1[y] = INFINITE_S;
		Y2[y] = INFINITE_S;
		Yf[y] = INFINITE_S;
	}

	memcpy(m_fx0_u, m_fx0, K * sizeof(REALV));
	memcpy(m_fx1_u, m_fx1, K * sizeof(REALV));
	memcpy(m_fx2_u, m_fx2, K * sizeof(REALV));

	// make y = L4 when k0 = L, k1 = L, k2 = L

	// y = -2*x1 + x2 = -2*(k1-L) + (k2-L) = 2*k1 - k2 + L + (4*L)
	// y = [L, L7]
	for (k2 = 0; k2 < K; k2++) {
		for (k1 = 0; k1 < K; k1++) {
			y = -2*k1 + k2 + L5;
			//delta_0 = m_fx1[k1] + m_fx2[k2];
			delta_0 = m_fx1_u[k1] + m_fx2_u[k2];
			TRUNCATE(Y0[y], delta_0);
		}
	}
	// y = x0 + x2 = (k0-L) + (k2-L) = k0 + k2 - 2*L + (4*L)
	// y = [L2, L6]
	for (k2 = 0; k2 < K; k2++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 + k2 + L2;
			//delta_1 = m_fx0[k0] + m_fx2[k2];
			delta_1 = m_fx0_u[k0] + m_fx2_u[k2];
			TRUNCATE(Y1[y], delta_1);
		}
	}
	// y = x0 - 2*x1 = (k0-L) - 2*(k1-L) = k0 - 2*k1 + L + (4*L)
	// y = [L, L7]
	for (k1 = 0; k1 < K; k1++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 - 2*k1 + L5;
			//delta_2 = m_fx0[k0] + m_fx1[k1];
			delta_2 = m_fx0_u[k0] + m_fx1_u[k1];
			TRUNCATE(Y2[y], delta_2);
		}
	}
	//
	// y = k0 - 2*k1 + k2 + (4*L) = k0 + y0 (= -2*k2 + k3 + 5*L) - L
	// y = [0, L8]
	for (k0 = 0; k0 < K; k0++) {
		for (y = L; y <= L7; y++) {
			yy = y + k0 - L;
			//delta_f = m_fx0[k0] + Y0[y];
			delta_f = m_fx0_u[k0] + Y0[y];
			TRUNCATE(Yf[yy], delta_f);
		}
	}
	//
    delta_f = Yf[0] + D[0];
	for (k = 1; k <= L8; k++) {
		Yf[k] += D[k];
		TRUNCATE(delta_f, Yf[k]);
	}
	//for (k = 0; k < L8_1; k++) { 
	//	Yf[k] -= delta_f;
	//}
	//////////////////////////////////////////

	//////////////////////////////////////////
	// Calculating messages
	//////////////////////////////////////////
	for (k0 = 0; k0 < K; k0++) {
		delta_0 = D[L+k0-L] + Y0[L];
		for (y = L; y <= L7; y++) {
			v_fx = D[y+k0-L] + Y0[y];
			TRUNCATE(delta_0, v_fx);
		}
		m_fx0_u[k0] = r_fx0 * (delta_0-delta_f) + (r_fx0-1)*m_fx0[k0];
	}
	cu_SubtractMin(m_fx0_u, m_fx0, K, delta_0);
	//////////////////////////////////////////
	for (k1 = 0; k1 < K; k1++) {
		delta_1 = D[L2-2*k1+L2] + Y1[L2];
		for (y = L2; y <= L6; y++) {
			v_fx = D[y-2*k1+L2] + Y1[y];
			TRUNCATE(delta_1, v_fx);
		}
		m_fx1_u[k1] = r_fx1 * (delta_1-delta_f) + (r_fx1-1)*m_fx1[k1];
	}
	cu_SubtractMin(m_fx1_u, m_fx1, K, delta_1);
	//////////////////////////////////////////
	for (k2 = 0; k2 < K; k2++) {
		delta_2 = D[L+k2-L] + Y2[L];
		for (y = L; y <= L7; y++) {
			v_fx = D[y+k2-L] + Y2[y];
			TRUNCATE(delta_2, v_fx);
		}
		m_fx2_u[k2] = r_fx2 * (delta_2-delta_f) + (r_fx2-1)*m_fx2[k2];
	}
	cu_SubtractMin(m_fx2_u, m_fx2, K, delta_2);
	//////////////////////////////////////////

	//////////////////////////////////////////
	// Updating lower bound
	//////////////////////////////////////////
	//lowerBound += delta_f + delta_0 + delta_1 + delta_2;
	atomicAdd(&dLowerBound, (int)(delta_f + delta_0 + delta_1 + delta_2));
	//////////////////////////////////////////
}

__global__ static void cu_UpdateDualMessage_TRW_S_BW(unsigned int blocksInY, float invBlocksInY,
	int ox, int oy, int oz, int mx, int my, int mz,
	REALV* ddcv, REALV** dDualMessage)
{
	int cx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
#ifdef CU_USE_3D_BLOCK
	int cy = __umul24(blockIdx.y , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdx.z , blockDim.z) + threadIdx.z;
#else
	int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	int blockIdxy = blockIdx.y - __umul24(blockIdxz, blocksInY);
	int cy = __umul24(blockIdxy , blockDim.y) + threadIdx.y;
	int cz = __umul24(blockIdxz , blockDim.z) + threadIdx.z;
#endif
	int x, y, z;
	size_t cidx, idx;
	//
	REALV *m_fx0_u, *m_fx1_u, *m_fx2_u;
	REALV m_fx0_b[MAX_K], m_fx1_b[MAX_K], m_fx2_b[MAX_K], *m_fx_buf;
	REALV *m_fx0, *m_fx1, *m_fx2;
	REALV r_fx0, r_fx1, r_fx2;
	REALV *Dm;
	REALV m_fx2_k2, fxfx, v_fx, delta_f, delta_0, delta_1, delta_2;
	int k0, k1, k2, kk, k2_K_2, k1_K, K, K_2;
	//
	extern __shared__ REALV sbuf[];

	x = cx + ox;
	y = cy + oy;
	z = cz + oz;

	idx  = (z*c_mesh_y + y)*c_mesh_x + x;
	cidx = (cz*my + cy)*mx + cx;

	K = c_K;
	K_2 = c_K * c_K;

	int tidx = threadIdx.y * blockDim.x + threadIdx.x;
	int tsize_e = 4*K;
	m_fx0_u  = &sbuf[tidx*tsize_e + 0*K];
	m_fx1_u  = &sbuf[tidx*tsize_e + 1*K];
	m_fx2_u  = &sbuf[tidx*tsize_e + 2*K];
	m_fx_buf = &sbuf[tidx*tsize_e + 3*K];

	//////////////////////////////////////////
	//////////////////////////////////////////
	Dm = &ddcv[cidx*c_num_d];
	m_fx0 = &dDualMessage[0][idx*K];
	m_fx1 = &dDualMessage[1][idx*K];
	m_fx2 = &dDualMessage[2][idx*K];
	r_fx0 = r_fx1 = r_fx2 = 1.0 / 3.0;

	memcpy(m_fx0_u, m_fx0, K * sizeof(REALV));
	memcpy(m_fx1_u, m_fx1, K * sizeof(REALV));
	memcpy(m_fx2_u, m_fx2, K * sizeof(REALV));

	delta_f = INFINITE_S;
	for (k2 = 0; k2 < K; k2++) {
		k2_K_2 = k2*K_2;
		//m_fx2_k2 = m_fx2[k2];
		m_fx2_k2 = m_fx2_u[k2];
		for (k1 = 0; k1 < K; k1++) {
			kk = k2_K_2 + k1*K;
			//fxfx = m_fx1[k1] + m_fx2_k2;
			fxfx = m_fx1_u[k1] + m_fx2_k2;
			for (k0 = 0; k0 < K; k0++) {
				//v_fx = Dm[kk + k0] + m_fx0[k0] + fxfx;
				v_fx = Dm[kk + k0] + m_fx0_u[k0] + fxfx;
				TRUNCATE(delta_f, v_fx);
				//
				//Dt[kk + k0] = v_fx;
			}
		}
	}

	for (k0 = 0; k0 < K; k0++) {
		delta_0 = INFINITE_S;
		for (k2 = 0; k2 < K; k2++) {
			kk = k2*K_2 + k0;
			for (k1 = 0; k1 < K; k1++) {
				//v_fx = Dt[kk + k1*K];
				//v_fx = Dm[kk + k1*K] + m_fx1[k1] + m_fx2[k2];
				v_fx = Dm[kk + k1*K] + m_fx1_u[k1] + m_fx2_u[k2];
				TRUNCATE(delta_0, v_fx);
			}
		}
		//m_fx0_u[k0] = r_fx0 * (delta_0-delta_f) - m_fx0[k0];
		//m_fx0_u[k0] = r_fx0 * (delta_0-delta_f) + (r_fx0-1) * m_fx0[k0];
		m_fx_buf[k0] = r_fx0 * (delta_0-delta_f) + (r_fx0-1) * m_fx0_u[k0];
	}
	memcpy(m_fx0_b, m_fx_buf, K * sizeof(REALV));

	for (k1 = 0; k1 < K; k1++) {
		delta_1 = INFINITE_S;
		k1_K = k1 * K;
		for (k2 = 0; k2 < K; k2++) {
			kk = k2*K_2 + k1_K;
			for (k0 = 0; k0 < K; k0++) {
				//v_fx = Dt[kk + k0];
				//v_fx = Dm[kk + k0] + m_fx0[k0] + m_fx2[k2];
				v_fx = Dm[kk + k0] + m_fx0_u[k0] + m_fx2_u[k2];
				TRUNCATE(delta_1, v_fx);
			}
		}
		//m_fx1_u[k1] = r_fx1 * (delta_1-delta_f) - m_fx1[k1];
		//m_fx1_u[k1] = r_fx1 * (delta_1-delta_f) + (r_fx1-1) * m_fx1[k1];
		m_fx_buf[k1] = r_fx1 * (delta_1-delta_f) + (r_fx1-1) * m_fx1_u[k1];
	}
	memcpy(m_fx1_b, m_fx_buf, K * sizeof(REALV));

	for (k2 = 0; k2 < K; k2++) {
		delta_2 = INFINITE_S;
		k2_K_2 = k2 * K_2;
		for (k1 = 0; k1 < K; k1++) {
			kk = k2_K_2 + k1 * K;
			for (k0 = 0; k0 < K; k0++) {
				//v_fx = Dt[kk + k0];
				//v_fx = Dm[kk + k0] + m_fx0[k0] + m_fx1[k1];
				v_fx = Dm[kk + k0] + m_fx0_u[k0] + m_fx1_u[k1];
				TRUNCATE(delta_2, v_fx);
			}
		}
		//m_fx2_u[k2] = r_fx2 * (delta_2-delta_f) - m_fx2[k2];
		//m_fx2_u[k2] = r_fx2 * (delta_2-delta_f) + (r_fx2-1) * m_fx2[k2];
		m_fx_buf[k2] = r_fx2 * (delta_2-delta_f) + (r_fx2-1) * m_fx2_u[k2];
	}
	memcpy(m_fx2_b, m_fx_buf, K * sizeof(REALV));
	//////////////////////////////////////////
	//////////////////////////////////////////

	//////////////////////////////////////////
	//////////////////////////////////////////
	memcpy(m_fx0_u, m_fx0_b, K * sizeof(REALV));
	memcpy(m_fx1_u, m_fx1_b, K * sizeof(REALV));
	memcpy(m_fx2_u, m_fx2_b, K * sizeof(REALV));

	cu_SubtractMin(m_fx0_u, m_fx0, K, delta_0);
	cu_SubtractMin(m_fx1_u, m_fx1, K, delta_1);
	cu_SubtractMin(m_fx2_u, m_fx2, K, delta_2);
	//////////////////////////////////////////
	//////////////////////////////////////////

	//////////////////////////////////////////
	//////////////////////////////////////////
	//lowerBound += delta_f + delta_0 + delta_1 + delta_2;
	atomicAdd(&dLowerBound, (int)(delta_f + delta_0 + delta_1 + delta_2));
	//////////////////////////////////////////
	//////////////////////////////////////////
}

extern "C"
void cu_TRW_S(int iter, REALV**** pdcv, REALV**** pOffset[3], REALV**** pRangeTerm[3], REALV**** pSO1[3][3], REALV**** pSO2[3][3],
	REALV**** pSO1F2Message[3][6], REALV**** pSO2F3Message[3][9], REALV**** pDualMessage[3], double* pLowerBound, double* pLowerBoundPrev, int iterPrev)
{
	int i, j, l, it;

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
#else
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
#endif

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	if (iterPrev == 0) {
		if (mmode == 0) {
			cu_VolInit(ddcv, pdcv, 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, num_d);
		}
		for (i = 0; i < 3; i++) {
			cu_VolInit(hOffset[i], pOffset[i], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, 1);
			cu_VolInit(hRangeTerm[i], pRangeTerm[i], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
			if (in_scv_w_O1F2 != -2) {
				if (mmode == 0) {
					for (j = 0; j < 3; j++) {
						cu_VolInit(hSO1[i][j], pSO1[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, L4_1);
					}
				}
				for (j = 0; j < 6; j++) {
					cu_VolInit(hSO1F2Message[i][j], pSO1F2Message[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
				}
			}
			if (in_scv_w_O2F3 != -2) {
				if (mmode == 0) {
					for (j = 0; j < 3; j++) {
						cu_VolInit(hSO2[i][j], pSO2[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, L8_1);
					}
				}
				for (j = 0; j < 9; j++) {
					cu_VolInit(hSO2F3Message[i][j], pSO2F3Message[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
				}
			}
			cu_VolInit(hDualMessage[i], pDualMessage[i], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	cutStopTimer(timer);
	TRACE2("h->d time = %f\n", cutGetTimerValue(timer));
	cutDeleteTimer(timer);

	cutCreateTimer(&timer);
	cutStartTimer(timer);
#else
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	TRACE2("h->d time = %f\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
#endif

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	//cudaPrintfInit(blocksInX*blocksInY*blocksInZ*threadsInX*threadsInY*threadsInZ*256);

	for (it = 0; it < iter; it++) {
		iLowerBound = 0;
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(dLowerBound, &iLowerBound, sizeof(unsigned long long int)));

		///////////////////////////////////////////////////////////////////////////////////////
		// forward update
		///////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////////
		if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 == -2)) {
			for (l = 0; l < 3; l++) {
				cu_UpdateMessage_TRW_S_FW_O1F2<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0, mx, my, mz, hRangeTerm[l], ddSO1F2Message[l], hDualMessage[l]);
			}
		} else if ((in_scv_w_O1F2 == -2) && (in_scv_w_O2F3 != -2)) {
			for (l = 0; l < 3; l++) {
				cu_UpdateMessage_TRW_S_FW_O2F3<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0, mx, my, mz, hRangeTerm[l], ddSO2F3Message[l], hDualMessage[l]);
			}
		} else if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 != -2)) {
			for (l = 0; l < 3; l++) {
				cu_UpdateMessage_TRW_S_FW_O1F2_O2F3<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0, mx, my, mz, hRangeTerm[l], ddSO1F2Message[l], ddSO2F3Message[l], hDualMessage[l]);
			}
		}
		cudaDeviceSynchronize();
		///////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////////
		// backward update
		///////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////////
		if (in_scv_w_O1F2 != -2) {
			for (l = 2; l >= 0; l--) {
				if (mmode == 1) {
					for (j = 0; j < 3; j++) {
						CUDA_SAFE_CALL(cudaMalloc(&hSO1[l][j], msize * L4_1 * sizeof(REALV)));
						cu_VolInit(hSO1[l][j], pSO1[l][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, L4_1);
					}
					CUDA_SAFE_CALL(cudaMemcpy(ddSO1[l], hSO1[l], 3 * sizeof(REALV*), cudaMemcpyHostToDevice));
				}
				//
				cu_UpdateSpatialMessage_TRW_S_BW_O1F2<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 4, 0, 0, 0, mx, my, mz, hOffset[l], ddSO1[l], ddSO1F2Message[l]);
				cu_UpdateSpatialMessage_TRW_S_BW_O1F2<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 2, 0, 0, 0, mx, my, mz, hOffset[l], ddSO1[l], ddSO1F2Message[l]);
				cu_UpdateSpatialMessage_TRW_S_BW_O1F2<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0, 0, mx, my, mz, hOffset[l], ddSO1[l], ddSO1F2Message[l]);
				//
				if (mmode == 1) {
					for (j = 0; j < 3; j++) {
						CUDA_SAFE_CALL(cudaFree(hSO1[l][j]));
					}
				}
			}
		}
		if (in_scv_w_O2F3 != -2) {
			for (l = 2; l >= 0; l--) {
				if (mmode == 1) {
					for (j = 0; j < 3; j++) {
						CUDA_SAFE_CALL(cudaMalloc(&hSO2[l][j], msize * L8_1 * sizeof(REALV)));
						cu_VolInit(hSO2[l][j], pSO2[l][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, L8_1);
					}
					CUDA_SAFE_CALL(cudaMemcpy(ddSO2[l], hSO2[l], 3 * sizeof(REALV*), cudaMemcpyHostToDevice));
				}
				//
				cu_UpdateSpatialMessage_TRW_S_BW_O2F3<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 7, 0, 0, 0, mx, my, mz, ddSO2[l], ddSO2F3Message[l]);
				cu_UpdateSpatialMessage_TRW_S_BW_O2F3<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 4, 0, 0, 0, mx, my, mz, ddSO2[l], ddSO2F3Message[l]);
				cu_UpdateSpatialMessage_TRW_S_BW_O2F3<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 1, 0, 0, 0, mx, my, mz, ddSO2[l], ddSO2F3Message[l]);
				//
				if (mmode == 1) {
					for (j = 0; j < 3; j++) {
						CUDA_SAFE_CALL(cudaFree(hSO2[l][j]));
					}
				}
			}
		}
		cudaDeviceSynchronize();
		///////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////////
		//*
		if (mmode == 0) {
			cu_UpdateDualMessage_TRW_S_BW<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0, mx, my, mz, ddcv, dDualMessage);
		} else if (mmode == 1) {
			size_t msize_4 = msize / 4;
			int mz_4  =     mz / 4;
			int mz_42 = 2 * mz / 4;
			int mz_43 = 3 * mz / 4;
			//
			#if 0
			{
				size_t free_mem, total_mem, req_mem;
				CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
				req_mem = msize_4 * num_d * sizeof(REALV);
				if (req_mem > free_mem) {
					TRACE("req_mem = %u, free_mem = %u\n", req_mem, free_mem);
					return;
				}
			}
			#endif
			//
			CUDA_SAFE_CALL(cudaMalloc(&ddcv, msize_4 * num_d * sizeof(REALV)));
			//
			cu_VolInit(ddcv, pdcv, 0, 0, 0    , mx, my, mz_4, mesh_x, mesh_y, mesh_z, num_d);
			cu_UpdateDualMessage_TRW_S_BW<<<Dg_4, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0    , mx, my, mz_4, ddcv, dDualMessage);
			//
			cu_VolInit(ddcv, pdcv, 0, 0, mz_4 , mx, my, mz_4, mesh_x, mesh_y, mesh_z, num_d);
			cu_UpdateDualMessage_TRW_S_BW<<<Dg_4, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, mz_4 , mx, my, mz_4, ddcv, dDualMessage);
			//
			cu_VolInit(ddcv, pdcv, 0, 0, mz_42, mx, my, mz_4, mesh_x, mesh_y, mesh_z, num_d);
			cu_UpdateDualMessage_TRW_S_BW<<<Dg_4, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, mz_42, mx, my, mz_4, ddcv, dDualMessage);
			//
			cu_VolInit(ddcv, pdcv, 0, 0, mz_43, mx, my, mz_4, mesh_x, mesh_y, mesh_z, num_d);
			cu_UpdateDualMessage_TRW_S_BW<<<Dg_4, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, mz_43, mx, my, mz_4, ddcv, dDualMessage);
			//
			cudaFree(ddcv);
		}
		//*/
		cudaDeviceSynchronize();
		///////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////////
		for (l = 2; l >= 0; l--) {
			cu_UpdateMessage_TRW_S_BW<<<Dg, Db, Ns>>>(blocksInY, invBlocksInY, 0, 0, 0, mx, my, mz, hRangeTerm[l], ddSO1F2Message[l], ddSO2F3Message[l], hDualMessage[l]);
			//
			//FILE* fp = fopen("culog.txt", "w");
			//cudaPrintfDisplay(fp, false);
			//fflush(fp);
			//fclose(fp);
		}
		cudaDeviceSynchronize();
		///////////////////////////////////////////////////////////////////////////////////////

		CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&iLowerBound, dLowerBound, sizeof(unsigned long long int)));
		*pLowerBound = (double)iLowerBound;

		if (it < iter-1) {
			TRACE2("iter %03d, lb = %f, lb_delta = %f\n", it+iterPrev, *pLowerBound, *pLowerBound-*pLowerBoundPrev);
			//
			*pLowerBoundPrev = *pLowerBound;
		}
	}

	//cudaPrintfEnd();
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	cutStopTimer(timer);
	TRACE2("avg time = %f\n", cutGetTimerValue(timer) / iter);
	cutDeleteTimer(timer);

	cutCreateTimer(&timer);
	cutStartTimer(timer);
#else
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	TRACE2("avg time = %f\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
#endif

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	for (i = 0; i < 3; i++) {
		if (in_scv_w_O1F2 != -2) {
			for (j = 0; j < 6; j++) {
				cu_VolCopy(hSO1F2Message[i][j], pSO1F2Message[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
			}
		}
		if (in_scv_w_O2F3 != -2) {
			for (j = 0; j < 9; j++) {
				cu_VolCopy(hSO2F3Message[i][j], pSO2F3Message[i][j], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
			}
		}
		cu_VolCopy(hDualMessage[i], pDualMessage[i], 0, 0, 0, mx, my, mz, mesh_x, mesh_y, mesh_z, K);
	}
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

#ifdef CU_USE_TIMER
#ifdef CU_USE_CUTIL
	cutStopTimer(timer);
	TRACE2("d->h time = %f\n", cutGetTimerValue(timer));
	cutDeleteTimer(timer);
#else
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	TRACE2("d->h time = %f\n", elapsedTime);
   	cudaEventDestroy(start);
	cudaEventDestroy(stop); 
#endif
#endif
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
