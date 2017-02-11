///////////////////////////////////////////////////////////////////////////////////////
// VolumeBP.cpp
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

#include "stdafx.h"
#include <math.h>
#include "VolumeBP.h"


//#define _USE_CUDA
//#define O1_USE_OFFSET
//#define USE_TIME_CHECK


#define MIN(a,b)  (((a) < (b)) ? (a) : (b))
#define MAX(a,b)  (((a) > (b)) ? (a) : (b))
#define TRUNCATE_MIN(a,b) { if ((a) > (b)) (a) = (b); }
#define TRUNCATE_MAX(a,b) { if ((a) < (b)) (a) = (b); }
#define TRUNCATE TRUNCATE_MIN

#define INFINITE_S 1e10


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
VolumeBP::VolumeBP()
{
	dcv = NULL;
	Offset[0] = Offset[1] = Offset[2] = NULL;
	OffsetCS[0] = OffsetCS[1] = OffsetCS[2] = NULL;
}
VolumeBP::~VolumeBP()
{
	int i;
	for (i = 0; i < 3; i++) {
		if (Offset[i] != NULL) {
			delete Offset[i];
		}
		if (OffsetCS[i] != NULL) {
			delete OffsetCS[i];
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
BOOL VolumeBP::init(RVolume* _dcv, int _wsize, REALV _label_sx, REALV _label_sy, REALV _label_sz, int _lmode, REALV _alpha_O1, REALV _d_O1, REALV _alpha_O2, REALV _d_O2,
	RVolume* _xx, RVolume* _yy, RVolume* _zz, int _mesh_ex, int _mesh_ey, int _mesh_ez, REALV _in_scv_w_O1F2, REALV _in_scv_w_O2F2, REALV _in_scv_w_O2F3, RVolume* _sc)
{
	int i, j, k, y, k0, k1, k2, plane;
	//
	dcv = _dcv;
	nL = _wsize;
	label_sx = _label_sx;
	label_sy = _label_sy;
	label_sz = _label_sz;
	lmode = _lmode;
	alpha_O1 = _alpha_O1;
	d_O1 = _d_O1;
	alpha_O2 = _alpha_O2;
	d_O2 = _d_O2;
	//
	mesh_x = dcv->m_vd_x;
	mesh_y = dcv->m_vd_y;
	mesh_z = dcv->m_vd_z;
	mesh_ex = _mesh_ex;
	mesh_ey = _mesh_ey;
	mesh_ez = _mesh_ez;
	//
	//Offset[0] = _xx;
	//Offset[1] = _yy;
	//Offset[2] = _zz;
	Offset[0] = new RVolume();
	Offset[1] = new RVolume();
	Offset[2] = new RVolume();
	Offset[0]->allocate(mesh_x, mesh_y, mesh_z);
	Offset[1]->allocate(mesh_x, mesh_y, mesh_z);
	Offset[2]->allocate(mesh_x, mesh_y, mesh_z);
	for (k = 0; k < mesh_z; k++) {
		for (j = 0; j < mesh_y; j++) {
			for (i = 0; i < mesh_x; i++) {
				if (_xx != NULL && _yy != NULL && _zz != NULL) {
					Offset[0]->m_pData[k][j][i][0] = _xx->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
					Offset[1]->m_pData[k][j][i][0] = _yy->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
					Offset[2]->m_pData[k][j][i][0] = _zz->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
				} else {
					Offset[0]->m_pData[k][j][i][0] = 0;
					Offset[1]->m_pData[k][j][i][0] = 0;
					Offset[2]->m_pData[k][j][i][0] = 0;
				}
			}
		}
	}
	in_scv_w_O1F2 = _in_scv_w_O1F2;
	in_scv_w_O2F2 = _in_scv_w_O2F2;
	in_scv_w_O2F3 = _in_scv_w_O2F3;
	//
	K = 2 * nL + 1;
	K_2 = K * K;
	//
	if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 == -2)) {
		scv_w_O1F2	= in_scv_w_O1F2;
		O1_v_alpha	= alpha_O1;
		O1_v_T		= d_O1;
		scv_w_O2F3	= in_scv_w_O2F3;
		O2_v_alpha	= 0;
		O2_v_T		= 0;
	} else if ((in_scv_w_O1F2 == -2) && (in_scv_w_O2F3 != -2)) {
		scv_w_O1F2	= in_scv_w_O1F2;
		O1_v_alpha	= 0;
		O1_v_T		= 0;
		scv_w_O2F3	= in_scv_w_O2F3;
		O2_v_alpha	= alpha_O2;
		O2_v_T		= d_O2;
	} else if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 != -2)) {
		scv_w_O1F2	= in_scv_w_O1F2;
		O1_v_alpha	= alpha_O1;
		O1_v_T		= d_O1;
		scv_w_O2F3	= in_scv_w_O2F3;
		O2_v_alpha	= alpha_O2;
		O2_v_T		= d_O2;
	}
	//
	opt_l[0].allocate(mesh_x, mesh_y, mesh_z);
	opt_l[1].allocate(mesh_x, mesh_y, mesh_z);
	opt_l[2].allocate(mesh_x, mesh_y, mesh_z);
	//
	if (lmode == 0) {
		num_d = K * K * K;
		//
		for (k = 0; k < K; k++) {
			for (j = 0; j < K; j++) {
				for (i = 0; i < K; i++) {
					disp_x[k * K_2 + j * K + i] = label_sx * (i - nL);
					disp_y[k * K_2 + j * K + i] = label_sy * (j - nL);
					disp_z[k * K_2 + j * K + i] = label_sz * (k - nL);
				}
			}
		}
		for (i = 0; i < K; i++) {
			disp_ex[i] = label_sx * (i - nL);
			disp_ey[i] = label_sy * (i - nL);
			disp_ez[i] = label_sz * (i - nL);
		}
		//
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					opt_l[0].m_pData[k][j][i][0] = nL;
					opt_l[1].m_pData[k][j][i][0] = nL;
					opt_l[2].m_pData[k][j][i][0] = nL;
				}
			}
		}
	} else {
		return FALSE;
	}
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
	if (in_scv_w_O1F2 != -2) {
		for (plane = 0; plane < 3; plane++) {
			for (i = 0; i < 3; i++) {
				SO1[plane][i].allocate(mesh_x, mesh_y, mesh_z, L4_1);
			}
		}
		//
		for (k1 = 0; k1 < K; k1++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 - k1 + L2;
			O1F2k0[y] = k0;
			O1F2k1[y] = k1;
		}}
		//
		///////////////////////////////////////////////////
		for (plane = 0; plane < 3; plane++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						REALV d0, d1, d00, d10, dd;
						REALV s0, s1, sc;

						if (i <= mesh_x-2) {
							d00 = Offset[plane]->m_pData[k][j][i  ][0];
							d10 = Offset[plane]->m_pData[k][j][i+1][0];
							if (_sc == NULL) {
								sc = 1.0f;
							} else {
								s0 = _sc->m_pData[k * mesh_ez][j * mesh_ey][(i  ) * mesh_ex][0];
								s1 = _sc->m_pData[k * mesh_ez][j * mesh_ey][(i+1) * mesh_ex][0];
								sc = min(s0, s1);
							}
							for (y = 0; y <= L4; y++) {
								d0 = disp_ex[O1F2k0[y]];
								d1 = disp_ex[O1F2k1[y]];
#ifdef O1_USE_OFFSET
								dd = ((d00+d0) - (d10+d1));
#else
								dd = (d0 - d1);
#endif
								SO1[plane][0].m_pData[k][j][i][y] = scv_w_O1F2 * MIN(O1_v_alpha*fabs(dd), O1_v_T) * sc;
							}
						}
						if (j <= mesh_y-2) {
							d00 = Offset[plane]->m_pData[k][j  ][i][0];
							d10 = Offset[plane]->m_pData[k][j+1][i][0];
							if (_sc == NULL) {
								sc = 1.0f;
							} else {
								s0 = _sc->m_pData[k * mesh_ez][(j  ) * mesh_ey][i * mesh_ex][0];
								s1 = _sc->m_pData[k * mesh_ez][(j+1) * mesh_ey][i * mesh_ex][0];
								sc = min(s0, s1);
							}
							for (y = 0; y <= L4; y++) {
								d0 = disp_ey[O1F2k0[y]];
								d1 = disp_ey[O1F2k1[y]];
#ifdef O1_USE_OFFSET
								dd = ((d00+d0) - (d10+d1));
#else
								dd = (d0 - d1);
#endif
								SO1[plane][1].m_pData[k][j][i][y] = scv_w_O1F2 * MIN(O1_v_alpha*fabs(dd), O1_v_T) * sc;
							}
						}
						if (k <= mesh_z-2) {
							d00 = Offset[plane]->m_pData[k  ][j][i][0];
							d10 = Offset[plane]->m_pData[k+1][j][i][0];
							if (_sc == NULL) {
								sc = 1.0f;
							} else {
								s0 = _sc->m_pData[(k  ) * mesh_ez][j * mesh_ey][i * mesh_ex][0];
								s1 = _sc->m_pData[(k+1) * mesh_ez][j * mesh_ey][i * mesh_ex][0];
								sc = min(s0, s1);
							}
							for (y = 0; y <= L4; y++) {
								d0 = disp_ez[O1F2k0[y]];
								d1 = disp_ez[O1F2k1[y]];
#ifdef O1_USE_OFFSET
								dd = ((d00+d0) - (d10+d1));
#else
								dd = (d0 - d1);
#endif
								SO1[plane][2].m_pData[k][j][i][y] = scv_w_O1F2 * MIN(O1_v_alpha*fabs(dd), O1_v_T) * sc;
							}
						}
					}
				}
			}
		}
		///////////////////////////////////////////////////
	}
	//
	if (in_scv_w_O2F3 != -2) {
		for (plane = 0; plane < 3; plane++) {
			for (i = 0; i < 3; i++) {
				SO2[plane][i].allocate(mesh_x, mesh_y, mesh_z, L8_1);
			}
		}
		//
		for (k2 = 0; k2 < K; k2++) {
		for (k1 = 0; k1 < K; k1++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 - 2*k1 + k2 + L4;
			O2F3k0[y] = k0;
			O2F3k1[y] = k1;
			O2F3k2[y] = k2;
		}}}
		//
		///////////////////////////////////////////////////
		for (plane = 0; plane < 3; plane++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						REALV d0, d1, d2, d00, d10, d20, dd;
						REALV s0, s1, s2, sc;

						if ((i >= 1) && (i <= mesh_x-2)) {
							d00 = Offset[plane]->m_pData[k][j][i-1][0];
							d10 = Offset[plane]->m_pData[k][j][i  ][0];
							d20 = Offset[plane]->m_pData[k][j][i+1][0];
							if (_sc == NULL) {
								sc = 1.0f;
							} else {
								s0 = _sc->m_pData[k * mesh_ez][j * mesh_ey][(i-1) * mesh_ex][0];
								s1 = _sc->m_pData[k * mesh_ez][j * mesh_ey][(i  ) * mesh_ex][0];
								s2 = _sc->m_pData[k * mesh_ez][j * mesh_ey][(i+1) * mesh_ex][0];
								sc = min(s0, min(s1, s2));
							}
							for (y = 0; y <= L8; y++) {
								d0 = d00 + disp_ex[O2F3k0[y]];
								d1 = d10 + disp_ex[O2F3k1[y]];
								d2 = d20 + disp_ex[O2F3k2[y]];
								dd = (d0 - 2*d1 + d2);
								//SO2[plane][0].m_pData[k][j][i][y] = scv_w_O2F3 * O2_v_alpha * dd*dd * sc;
								SO2[plane][0].m_pData[k][j][i][y] = scv_w_O2F3 * MIN(O2_v_alpha*fabs(dd), O2_v_T) * sc;
							}
						}
						if ((j >= 1) && (j <= mesh_y-2)) {
							d00 = Offset[plane]->m_pData[k][j-1][i][0];
							d10 = Offset[plane]->m_pData[k][j  ][i][0];
							d20 = Offset[plane]->m_pData[k][j+1][i][0];
							if (_sc == NULL) {
								sc = 1.0f;
							} else {
								s0 = _sc->m_pData[k * mesh_ez][(j-1) * mesh_ey][i * mesh_ex][0];
								s1 = _sc->m_pData[k * mesh_ez][(j  ) * mesh_ey][i * mesh_ex][0];
								s2 = _sc->m_pData[k * mesh_ez][(j+1) * mesh_ey][i * mesh_ex][0];
								sc = min(s0, min(s1, s2));
							}
							for (y = 0; y <= L8; y++) {
								d0 = d00 + disp_ey[O2F3k0[y]];
								d1 = d10 + disp_ey[O2F3k1[y]];
								d2 = d20 + disp_ey[O2F3k2[y]];
								dd = (d0 - 2*d1 + d2);
								//SO2[plane][1].m_pData[k][j][i][y] = scv_w_O2F3 * O2_v_alpha * dd*dd * sc;
								SO2[plane][1].m_pData[k][j][i][y] = scv_w_O2F3 * MIN(O2_v_alpha*fabs(dd), O2_v_T) * sc;
							}
						}
						if ((k >= 1) && (k <= mesh_z-2)) {
							d00 = Offset[plane]->m_pData[k-1][j][i][0];
							d10 = Offset[plane]->m_pData[k  ][j][i][0];
							d20 = Offset[plane]->m_pData[k+1][j][i][0];
							if (_sc == NULL) {
								sc = 1.0f;
							} else {
								s0 = _sc->m_pData[(k-1) * mesh_ez][j * mesh_ey][i * mesh_ex][0];
								s1 = _sc->m_pData[(k  ) * mesh_ez][j * mesh_ey][i * mesh_ex][0];
								s2 = _sc->m_pData[(k+1) * mesh_ez][j * mesh_ey][i * mesh_ex][0];
								sc = min(s0, min(s1, s2));
							}
							for (y = 0; y <= L8; y++) {
								d0 = d00 + disp_ez[O2F3k0[y]];
								d1 = d10 + disp_ez[O2F3k1[y]];
								d2 = d20 + disp_ez[O2F3k2[y]];
								dd = (d0 - 2*d1 + d2);
								//SO2[plane][2].m_pData[k][j][i][y] = scv_w_O2F3 * O2_v_alpha * dd*dd * sc;
								SO2[plane][2].m_pData[k][j][i][y] = scv_w_O2F3 * MIN(O2_v_alpha*fabs(dd), O2_v_T) * sc;
							}
						}
					}
				}
			}
		}
		///////////////////////////////////////////////////
	}

	return TRUE;
}

BOOL VolumeBP::initSyD(RVolume* _dcv, int _wsize, REALV _label_sx, REALV _label_sy, REALV _label_sz, int _lmode, REALV _alpha_O1, REALV _d_O1, REALV _alpha_O2, REALV _d_O2,
	RVolume* _xx_cs, RVolume* _yy_cs, RVolume* _zz_cs, RVolume* _xx_ct, RVolume* _yy_ct, RVolume* _zz_ct, 
	int _mesh_ex, int _mesh_ey, int _mesh_ez, REALV _in_scv_w_O1F2, REALV _in_scv_w_O2F2, REALV _in_scv_w_O2F3)
{
	int i, j, k, y, k0, k1, k2, plane;
	//
	dcv = _dcv;
	nL = _wsize;
	label_sx = _label_sx;
	label_sy = _label_sy;
	label_sz = _label_sz;
	lmode = _lmode;
	alpha_O1 = _alpha_O1;
	d_O1 = _d_O1;
	alpha_O2 = _alpha_O2;
	d_O2 = _d_O2;
	//
	mesh_x = dcv->m_vd_x;
	mesh_y = dcv->m_vd_y;
	mesh_z = dcv->m_vd_z;
	mesh_ex = _mesh_ex;
	mesh_ey = _mesh_ey;
	mesh_ez = _mesh_ez;
	//
	Offset[0] = new RVolume();
	Offset[1] = new RVolume();
	Offset[2] = new RVolume();
	OffsetCS[0] = new RVolume();
	OffsetCS[1] = new RVolume();
	OffsetCS[2] = new RVolume();
	Offset[0]->allocate(mesh_x, mesh_y, mesh_z);
	Offset[1]->allocate(mesh_x, mesh_y, mesh_z);
	Offset[2]->allocate(mesh_x, mesh_y, mesh_z);
	OffsetCS[0]->allocate(mesh_x, mesh_y, mesh_z);
	OffsetCS[1]->allocate(mesh_x, mesh_y, mesh_z);
	OffsetCS[2]->allocate(mesh_x, mesh_y, mesh_z);
	for (k = 0; k < mesh_z; k++) {
		for (j = 0; j < mesh_y; j++) {
			for (i = 0; i < mesh_x; i++) {
				if (_xx_cs != NULL && _yy_cs != NULL && _zz_cs != NULL) {
					OffsetCS[0]->m_pData[k][j][i][0] = _xx_cs->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
					OffsetCS[1]->m_pData[k][j][i][0] = _yy_cs->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
					OffsetCS[2]->m_pData[k][j][i][0] = _zz_cs->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
				} else {
					OffsetCS[0]->m_pData[k][j][i][0] = 0;
					OffsetCS[1]->m_pData[k][j][i][0] = 0;
					OffsetCS[2]->m_pData[k][j][i][0] = 0;
				}
				if (_xx_ct != NULL && _yy_ct != NULL && _zz_ct != NULL) {
					Offset[0]->m_pData[k][j][i][0] = _xx_ct->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
					Offset[1]->m_pData[k][j][i][0] = _yy_ct->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
					Offset[2]->m_pData[k][j][i][0] = _zz_ct->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
				} else {
					Offset[0]->m_pData[k][j][i][0] = 0;
					Offset[1]->m_pData[k][j][i][0] = 0;
					Offset[2]->m_pData[k][j][i][0] = 0;
				}
			}
		}
	}
	in_scv_w_O1F2 = _in_scv_w_O1F2;
	in_scv_w_O2F2 = _in_scv_w_O2F2;
	in_scv_w_O2F3 = _in_scv_w_O2F3;
	//
	K = 2 * nL + 1;
	K_2 = K * K;
	//
	if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 == -2)) {
		scv_w_O1F2	= in_scv_w_O1F2;
		O1_v_alpha	= alpha_O1;
		O1_v_T		= d_O1;
		scv_w_O2F3	= in_scv_w_O2F3;
		O2_v_alpha	= 0;
		O2_v_T		= 0;
	} else if ((in_scv_w_O1F2 == -2) && (in_scv_w_O2F3 != -2)) {
		scv_w_O1F2	= in_scv_w_O1F2;
		O1_v_alpha	= 0;
		O1_v_T		= 0;
		scv_w_O2F3	= in_scv_w_O2F3;
		O2_v_alpha	= alpha_O2;
		O2_v_T		= d_O2;
	} else if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 != -2)) {
		scv_w_O1F2	= in_scv_w_O1F2;
		O1_v_alpha	= alpha_O1;
		O1_v_T		= d_O1;
		scv_w_O2F3	= in_scv_w_O2F3;
		O2_v_alpha	= alpha_O2;
		O2_v_T		= d_O2;
	}
	//
	opt_l[0].allocate(mesh_x, mesh_y, mesh_z);
	opt_l[1].allocate(mesh_x, mesh_y, mesh_z);
	opt_l[2].allocate(mesh_x, mesh_y, mesh_z);
	//
	if (lmode == 0) {
		num_d = K * K * K;
		//
		for (k = 0; k < K; k++) {
			for (j = 0; j < K; j++) {
				for (i = 0; i < K; i++) {
					disp_x[k * K_2 + j * K + i] = label_sx * (i - nL);
					disp_y[k * K_2 + j * K + i] = label_sy * (j - nL);
					disp_z[k * K_2 + j * K + i] = label_sz * (k - nL);
				}
			}
		}
		for (i = 0; i < K; i++) {
			disp_ex[i] = label_sx * (i - nL);
			disp_ey[i] = label_sy * (i - nL);
			disp_ez[i] = label_sz * (i - nL);
		}
		//
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					opt_l[0].m_pData[k][j][i][0] = nL;
					opt_l[1].m_pData[k][j][i][0] = nL;
					opt_l[2].m_pData[k][j][i][0] = nL;
				}
			}
		}
	} else {
		return FALSE;
	}
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
	if (in_scv_w_O1F2 != -2) {
		for (plane = 0; plane < 3; plane++) {
			for (i = 0; i < 3; i++) {
				SO1[plane][i].allocate(mesh_x, mesh_y, mesh_z, L4_1);
			}
		}
		//
		for (k1 = 0; k1 < K; k1++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 - k1 + L2;
			O1F2k0[y] = k0;
			O1F2k1[y] = k1;
		}}
		//
		///////////////////////////////////////////////////
		for (plane = 0; plane < 3; plane++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						REALV d0, d1, dd;

						if (i <= mesh_x-2) {
							for (y = 0; y <= L4; y++) {
								d0 = disp_ex[O1F2k0[y]];
								d1 = disp_ex[O1F2k1[y]];
								dd = (d0 - d1);
								SO1[plane][0].m_pData[k][j][i][y] = scv_w_O1F2 * MIN(O1_v_alpha*fabs(dd), O1_v_T);
							}
							//
							for (y = 0; y <= L4; y++) {
								d0 = -disp_ex[O1F2k0[y]];
								d1 = -disp_ex[O1F2k1[y]];
								dd = (d0 - d1);
								SO1[plane][0].m_pData[k][j][i][y] += scv_w_O1F2 * MIN(O1_v_alpha*fabs(dd), O1_v_T);
							}
						}
						if (j <= mesh_y-2) {
							for (y = 0; y <= L4; y++) {
								d0 = disp_ey[O1F2k0[y]];
								d1 = disp_ey[O1F2k1[y]];
								dd = (d0 - d1);
								SO1[plane][1].m_pData[k][j][i][y] = scv_w_O1F2 * MIN(O1_v_alpha*fabs(dd), O1_v_T);
							}
							//
							for (y = 0; y <= L4; y++) {
								d0 = -disp_ey[O1F2k0[y]];
								d1 = -disp_ey[O1F2k1[y]];
								dd = (d0 - d1);
								SO1[plane][1].m_pData[k][j][i][y] += scv_w_O1F2 * MIN(O1_v_alpha*fabs(dd), O1_v_T);
							}
						}
						if (k <= mesh_z-2) {
							for (y = 0; y <= L4; y++) {
								d0 = disp_ez[O1F2k0[y]];
								d1 = disp_ez[O1F2k1[y]];
								dd = (d0 - d1);
								SO1[plane][2].m_pData[k][j][i][y] = scv_w_O1F2 * MIN(O1_v_alpha*fabs(dd), O1_v_T);
							}
							//
							for (y = 0; y <= L4; y++) {
								d0 = -disp_ez[O1F2k0[y]];
								d1 = -disp_ez[O1F2k1[y]];
								dd = (d0 - d1);
								SO1[plane][2].m_pData[k][j][i][y] += scv_w_O1F2 * MIN(O1_v_alpha*fabs(dd), O1_v_T);
							}
						}
					}
				}
			}
		}
		///////////////////////////////////////////////////
	}
	//
	if (in_scv_w_O2F3 != -2) {
		for (plane = 0; plane < 3; plane++) {
			for (i = 0; i < 3; i++) {
				SO2[plane][i].allocate(mesh_x, mesh_y, mesh_z, L8_1);
			}
		}
		//
		for (k2 = 0; k2 < K; k2++) {
		for (k1 = 0; k1 < K; k1++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 - 2*k1 + k2 + L4;
			O2F3k0[y] = k0;
			O2F3k1[y] = k1;
			O2F3k2[y] = k2;
		}}}
		//
		///////////////////////////////////////////////////
		for (plane = 0; plane < 3; plane++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						REALV d0, d1, d2, d00, d10, d20, dd;

						if ((i >= 1) && (i <= mesh_x-2)) {
							d00 = Offset[plane]->m_pData[k][j][i-1][0];
							d10 = Offset[plane]->m_pData[k][j][i  ][0];
							d20 = Offset[plane]->m_pData[k][j][i+1][0];
							for (y = 0; y <= L8; y++) {
								d0 = d00 + disp_ex[O2F3k0[y]];
								d1 = d10 + disp_ex[O2F3k1[y]];
								d2 = d20 + disp_ex[O2F3k2[y]];
								dd = (d0 - 2*d1 + d2);
								//SO2[plane][0].m_pData[k][j][i][y] = scv_w_O2F3 * O2_v_alpha * dd*dd;
								SO2[plane][0].m_pData[k][j][i][y] = scv_w_O2F3 * MIN(O2_v_alpha*fabs(dd), O2_v_T);
							}
							//
							d00 = OffsetCS[plane]->m_pData[k][j][i-1][0];
							d10 = OffsetCS[plane]->m_pData[k][j][i  ][0];
							d20 = OffsetCS[plane]->m_pData[k][j][i+1][0];
							for (y = 0; y <= L8; y++) {
								d0 = d00 - disp_ex[O2F3k0[y]];
								d1 = d10 - disp_ex[O2F3k1[y]];
								d2 = d20 - disp_ex[O2F3k2[y]];
								dd = (d0 - 2*d1 + d2);
								//SO2[plane][0].m_pData[k][j][i][y] += scv_w_O2F3 * O2_v_alpha * dd*dd;
								SO2[plane][0].m_pData[k][j][i][y] += scv_w_O2F3 * MIN(O2_v_alpha*fabs(dd), O2_v_T);
							}
						}
						if ((j >= 1) && (j <= mesh_y-2)) {
							d00 = Offset[plane]->m_pData[k][j-1][i][0];
							d10 = Offset[plane]->m_pData[k][j  ][i][0];
							d20 = Offset[plane]->m_pData[k][j+1][i][0];
							for (y = 0; y <= L8; y++) {
								d0 = d00 + disp_ey[O2F3k0[y]];
								d1 = d10 + disp_ey[O2F3k1[y]];
								d2 = d20 + disp_ey[O2F3k2[y]];
								dd = (d0 - 2*d1 + d2);
								//SO2[plane][1].m_pData[k][j][i][y] = scv_w_O2F3 * O2_v_alpha * dd*dd;
								SO2[plane][1].m_pData[k][j][i][y] = scv_w_O2F3 * MIN(O2_v_alpha*fabs(dd), O2_v_T);
							}
							//
							d00 = OffsetCS[plane]->m_pData[k][j-1][i][0];
							d10 = OffsetCS[plane]->m_pData[k][j  ][i][0];
							d20 = OffsetCS[plane]->m_pData[k][j+1][i][0];
							for (y = 0; y <= L8; y++) {
								d0 = d00 - disp_ey[O2F3k0[y]];
								d1 = d10 - disp_ey[O2F3k1[y]];
								d2 = d20 - disp_ey[O2F3k2[y]];
								dd = (d0 - 2*d1 + d2);
								//SO2[plane][1].m_pData[k][j][i][y] += scv_w_O2F3 * O2_v_alpha * dd*dd;
								SO2[plane][1].m_pData[k][j][i][y] += scv_w_O2F3 * MIN(O2_v_alpha*fabs(dd), O2_v_T);
							}
						}
						if ((k >= 1) && (k <= mesh_z-2)) {
							d00 = Offset[plane]->m_pData[k-1][j][i][0];
							d10 = Offset[plane]->m_pData[k  ][j][i][0];
							d20 = Offset[plane]->m_pData[k+1][j][i][0];
							for (y = 0; y <= L8; y++) {
								d0 = d00 + disp_ez[O2F3k0[y]];
								d1 = d10 + disp_ez[O2F3k1[y]];
								d2 = d20 + disp_ez[O2F3k2[y]];
								dd = (d0 - 2*d1 + d2);
								//SO2[plane][2].m_pData[k][j][i][y] = scv_w_O2F3 * O2_v_alpha * dd*dd;
								SO2[plane][2].m_pData[k][j][i][y] = scv_w_O2F3 * MIN(O2_v_alpha*fabs(dd), O2_v_T);
							}
							//
							d00 = OffsetCS[plane]->m_pData[k-1][j][i][0];
							d10 = OffsetCS[plane]->m_pData[k  ][j][i][0];
							d20 = OffsetCS[plane]->m_pData[k+1][j][i][0];
							for (y = 0; y <= L8; y++) {
								d0 = d00 - disp_ez[O2F3k0[y]];
								d1 = d10 - disp_ez[O2F3k1[y]];
								d2 = d20 - disp_ez[O2F3k2[y]];
								dd = (d0 - 2*d1 + d2);
								//SO2[plane][2].m_pData[k][j][i][y] += scv_w_O2F3 * O2_v_alpha * dd*dd;
								SO2[plane][2].m_pData[k][j][i][y] += scv_w_O2F3 * MIN(O2_v_alpha*fabs(dd), O2_v_T);
							}
						}
					}
				}
			}
		}
		///////////////////////////////////////////////////
	}

	return TRUE;
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
// function to compute range term
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::ComputeRangeTerm(REALV _gamma)
{
	int i, j, k, l;
	//
	gamma = _gamma;
	//
	for (i = 0; i < 3; i++) {
		RangeTerm[i].allocate(mesh_x, mesh_y, mesh_z, K);
	}
	for (k = 0; k < mesh_z; k++) {
		for (j = 0; j < mesh_y; j++) {
			for (i = 0; i < mesh_x; i++) {
				for (l = 0; l < K; l++) {
					if (Offset[0] != NULL) {
						RangeTerm[0].m_pData[k][j][i][l] = gamma * fabs(Offset[0]->m_pData[k][j][i][0] + disp_ex[l]);
						RangeTerm[1].m_pData[k][j][i][l] = gamma * fabs(Offset[1]->m_pData[k][j][i][0] + disp_ey[l]);
						RangeTerm[2].m_pData[k][j][i][l] = gamma * fabs(Offset[2]->m_pData[k][j][i][0] + disp_ez[l]);
					}
					//
					if (OffsetCS[0] != NULL) {
						RangeTerm[0].m_pData[k][j][i][l] += gamma * fabs(OffsetCS[0]->m_pData[k][j][i][0] - disp_ex[l]);
						RangeTerm[1].m_pData[k][j][i][l] += gamma * fabs(OffsetCS[1]->m_pData[k][j][i][0] - disp_ey[l]);
						RangeTerm[2].m_pData[k][j][i][l] += gamma * fabs(OffsetCS[2]->m_pData[k][j][i][0] - disp_ez[l]);
					}
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::AllocateMessage()
{
	int i, j;
	for (i = 0; i < 3; i++) {
		if (in_scv_w_O1F2 != -2) {
			for (j = 0; j < 6; j++) {
				SO1F2Message[i][j].allocate(mesh_x, mesh_y, mesh_z, K);
			}
		}
		if (in_scv_w_O2F3 != -2) {
			for (j = 0; j < 9; j++) {
				SO2F3Message[i][j].allocate(mesh_x, mesh_y, mesh_z, K);
			}
		}
		DualMessage[i].allocate(mesh_x, mesh_y, mesh_z, K);
		Belief[i].allocate(mesh_x, mesh_y, mesh_z, K);
	}
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
// Update the message from (x, y, z, plane) to the neighbors on the same plane
//
// The encoding of the direction
//  0: -x -> +x, 1: +x -> -x
//  2: -y -> +y, 3: +y -> -y
//  4: -z -> +z, 5: +z -> -z
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::Add2Message(REALV* message, const REALV* other, int nstates)
{
	int i;
	for (i = 0; i < nstates; i++) {
		message[i] += other[i];
	}
}
void VolumeBP::Add2Message(REALV* message, const REALV* other, int nstates, double ctrw)
{
	int i;
	for (i = 0; i < nstates; i++) {
		message[i] += other[i] * ctrw;
	}
}
REALV VolumeBP::FindMin(REALV* message, int nstates)
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
void VolumeBP::SubtractMin(REALV* message, int nstates, REALV& min)
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
void VolumeBP::SubtractMin(REALV* message, REALV* message_out, int nstates, REALV& min)
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
void VolumeBP::Sub2Message(REALV* message, const REALV* other, int nstates)
{
	int i;
	for (i = 0; i < nstates; i++) {
		message[i] -= other[i];
	}
}
void VolumeBP::Sub2Message(REALV* message_out, REALV* message, const REALV* other, int nstates)
{
	int i;
	for (i = 0; i < nstates; i++) {
		message_out[i] = message[i] - other[i];
	}
}

REALV VolumeBP::ComputeSpatialMessageDT(REALV* message, REALV* message_org, REALV* message_buf, int x, int y, int z, REALV d0, int plane, int nStates, int wsize)
{
	REALV Min;
	ptrdiff_t l;

	if (message_org != message_buf) {
		memcpy(message_buf, message_org, nStates * sizeof(REALV));
	}

	// use distance transform function to impose smoothness compatibility
	Min = FindMin(message_buf, nStates) + d_O1;
	for (l = 1; l < nStates; l++) {
		message_buf[l] = MIN(message_buf[l], message_buf[l-1] + alpha_O1);
	}
	for (l = nStates-2; l >= 0; l--) {
		message_buf[l] = MIN(message_buf[l], message_buf[l+1] + alpha_O1);
	}

	// transform the compatibility 
	int shift = (int)d0;
	if (abs(shift) > wsize+wsize) { // the shift is too big that there is no overlap
		if (x > 0 || y > 0 || z > 0) {
			for (l = 0; l < nStates; l++) {
				message[l] =  l * alpha_O1;
			}
		} else {
			for (l = 0; l < nStates; l++) {
				message[l] = -l * alpha_O1;
			}
		}
	} else {
		int start = MAX(-wsize, shift-wsize);
		int end   = MIN( wsize, shift+wsize);
		for (l = start; l <= end; l++) {
			message[l-shift+wsize] = message_buf[l+wsize];
		}
		if (start-shift+wsize > 0) {
			for (l = start-shift+wsize-1; l >= 0; l--) {
				message[l] = message[l+1] + alpha_O1;
			}
		}
		if (end-shift+wsize < nStates) {
			for (l = end-shift+wsize+1; l < nStates; l++) {
				message[l] = message[l-1] + alpha_O1;
			}
		}
	}

	// put back the threshold
	for (l = 0; l < nStates; l++) {
		message[l] = MIN(message[l], Min);
	}

	return Min;
}

void VolumeBP::Add2MessageDual(REALV* message, int x, int y, int z, int plane, int nStates)
{
	// initialize the message using the range term
	memcpy(message, RangeTerm[plane].m_pData[z][y][x], sizeof(REALV) * nStates);

	// add spatial messages
	if (in_scv_w_O1F2 != -2) {
		if (x > 0) {		// add x- -> x+
			Add2Message(message, SO1F2Message[plane][0].m_pData[z][y][x], nStates);
		}
		if (x < mesh_x-1) {	// add x+ -> x-
			Add2Message(message, SO1F2Message[plane][1].m_pData[z][y][x], nStates);
		}
		if (y > 0) {		// add y- -> y+
			Add2Message(message, SO1F2Message[plane][2].m_pData[z][y][x], nStates);
		}
		if (y < mesh_y-1) {	// add y+ -> y-
			Add2Message(message, SO1F2Message[plane][3].m_pData[z][y][x], nStates);
		}
		if (z > 0) {		// add z- -> z+
			Add2Message(message, SO1F2Message[plane][4].m_pData[z][y][x], nStates);
		}
		if (z < mesh_z-1) {	// add z+ -> z-
			Add2Message(message, SO1F2Message[plane][5].m_pData[z][y][x], nStates);
		}
	}
	if (in_scv_w_O2F3 != -2) {
		if (x < mesh_x-2) {				// f+ -> x
			Add2Message(message, SO2F3Message[plane][0].m_pData[z][y][x], nStates);
		}
		if (x > 0 && x < mesh_x-1) {	// f0 -> x
			Add2Message(message, SO2F3Message[plane][1].m_pData[z][y][x], nStates);
		}
		if (x > 1) {					// f- -> x
			Add2Message(message, SO2F3Message[plane][2].m_pData[z][y][x], nStates);
		}
		if (y < mesh_y-2) {
			Add2Message(message, SO2F3Message[plane][3].m_pData[z][y][x], nStates);
		}
		if (y > 0 && y < mesh_y-1) {
			Add2Message(message, SO2F3Message[plane][4].m_pData[z][y][x], nStates);
		}
		if (y > 1) {
			Add2Message(message, SO2F3Message[plane][5].m_pData[z][y][x], nStates);
		}
		if (z < mesh_z-2) {
			Add2Message(message, SO2F3Message[plane][6].m_pData[z][y][x], nStates);
		}
		if (z > 0 && z < mesh_z-1) {
			Add2Message(message, SO2F3Message[plane][7].m_pData[z][y][x], nStates);
		}
		if (z > 1) {
			Add2Message(message, SO2F3Message[plane][8].m_pData[z][y][x], nStates);
		}
	}
}

void VolumeBP::Add2MessageSpatial_O1F2(REALV* message, int x, int y, int z, int plane, int direction, int nStates)
{
	// initialize the message from the dual plane
	memcpy(message, DualMessage[plane].m_pData[z][y][x], sizeof(REALV) * nStates);

	// add the range term
	Add2Message(message, RangeTerm[plane].m_pData[z][y][x], nStates);

	// add spatial messages
	if (in_scv_w_O1F2 != -2) {
		if (x > 0        && direction != 1) {	// add x- -> x+
			Add2Message(message, SO1F2Message[plane][0].m_pData[z][y][x], nStates);
		}
		if (x < mesh_x-1 && direction != 0) {	// add x+ -> x-
			Add2Message(message, SO1F2Message[plane][1].m_pData[z][y][x], nStates);
		}
		if (y > 0        && direction != 3) {	// add y- -> y+
			Add2Message(message, SO1F2Message[plane][2].m_pData[z][y][x], nStates);
		}
		if (y < mesh_y-1 && direction != 2) {	// add y+ -> y-
			Add2Message(message, SO1F2Message[plane][3].m_pData[z][y][x], nStates);
		}
		if (z > 0        && direction != 5) {	// add z- -> z+
			Add2Message(message, SO1F2Message[plane][4].m_pData[z][y][x], nStates);
		}
		if (z < mesh_z-1 && direction != 4) {	// add z+ -> z-
			Add2Message(message, SO1F2Message[plane][5].m_pData[z][y][x], nStates);
		}
	}
	if (in_scv_w_O2F3 != -2) {
		if (x < mesh_x-2) {				// f+ -> x
			Add2Message(message, SO2F3Message[plane][0].m_pData[z][y][x], nStates);
		}
		if (x > 0 && x < mesh_x-1) {	// f0 -> x
			Add2Message(message, SO2F3Message[plane][1].m_pData[z][y][x], nStates);
		}
		if (x > 1) {					// f- -> x
			Add2Message(message, SO2F3Message[plane][2].m_pData[z][y][x], nStates);
		}
		if (y < mesh_y-2) {
			Add2Message(message, SO2F3Message[plane][3].m_pData[z][y][x], nStates);
		}
		if (y > 0 && y < mesh_y-1) {
			Add2Message(message, SO2F3Message[plane][4].m_pData[z][y][x], nStates);
		}
		if (y > 1) {
			Add2Message(message, SO2F3Message[plane][5].m_pData[z][y][x], nStates);
		}
		if (z < mesh_z-2) {
			Add2Message(message, SO2F3Message[plane][6].m_pData[z][y][x], nStates);
		}
		if (z > 0 && z < mesh_z-1) {
			Add2Message(message, SO2F3Message[plane][7].m_pData[z][y][x], nStates);
		}
		if (z > 1) {
			Add2Message(message, SO2F3Message[plane][8].m_pData[z][y][x], nStates);
		}
	}
}

void VolumeBP::Add2MessageSpatial_O2F3(REALV* message, int x, int y, int z, int plane, int direction, int nStates)
{
	// initialize the message from the dual plane
	memcpy(message, DualMessage[plane].m_pData[z][y][x], sizeof(REALV) * nStates);

	// add the range term
	Add2Message(message, RangeTerm[plane].m_pData[z][y][x], nStates);

	// add spatial messages
	if (in_scv_w_O1F2 != -2) {
		if (x > 0) {		// add x- -> x+
			Add2Message(message, SO1F2Message[plane][0].m_pData[z][y][x], nStates);
		}
		if (x < mesh_x-1) {	// add x+ -> x-
			Add2Message(message, SO1F2Message[plane][1].m_pData[z][y][x], nStates);
		}
		if (y > 0) {		// add y- -> y+
			Add2Message(message, SO1F2Message[plane][2].m_pData[z][y][x], nStates);
		}
		if (y < mesh_y-1) {	// add y+ -> y-
			Add2Message(message, SO1F2Message[plane][3].m_pData[z][y][x], nStates);
		}
		if (z > 0) {		// add z- -> z+
			Add2Message(message, SO1F2Message[plane][4].m_pData[z][y][x], nStates);
		}
		if (z < mesh_z-1) {	// add z+ -> z-
			Add2Message(message, SO1F2Message[plane][5].m_pData[z][y][x], nStates);
		}
	}
	if (in_scv_w_O2F3 != -2) {
		if (x < mesh_x-2 && direction != 0) {			// f+ -> x
			Add2Message(message, SO2F3Message[plane][0].m_pData[z][y][x], nStates);
		}
		if (x > 0 && x < mesh_x-1 && direction != 1) {	// f0 -> x
			Add2Message(message, SO2F3Message[plane][1].m_pData[z][y][x], nStates);
		}
		if (x > 1 && direction != 2) {					// f- -> x
			Add2Message(message, SO2F3Message[plane][2].m_pData[z][y][x], nStates);
		}
		if (y < mesh_y-2 && direction != 3) {
			Add2Message(message, SO2F3Message[plane][3].m_pData[z][y][x], nStates);
		}
		if (y > 0 && y < mesh_y-1 && direction != 4) {
			Add2Message(message, SO2F3Message[plane][4].m_pData[z][y][x], nStates);
		}
		if (y > 1 && direction != 5) {
			Add2Message(message, SO2F3Message[plane][5].m_pData[z][y][x], nStates);
		}
		if (z < mesh_z-2 && direction != 6) {
			Add2Message(message, SO2F3Message[plane][6].m_pData[z][y][x], nStates);
		}
		if (z > 0 && z < mesh_z-1 && direction != 7) {
			Add2Message(message, SO2F3Message[plane][7].m_pData[z][y][x], nStates);
		}
		if (z > 1 && direction != 8) {
			Add2Message(message, SO2F3Message[plane][8].m_pData[z][y][x], nStates);
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
// Update Message for BP
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::UpdateSpatialMessage(int x, int y, int z, int plane, int direction)
{
	int x1, y1, z1;
	REALV* message_org;
	REALV* message;
	REALV* disp_e = NULL;

	// eliminate impossible messages
	if (direction == 0 && x == mesh_x-1) { return; }
	if (direction == 1 && x == 0       ) { return; }
	if (direction == 2 && y == mesh_y-1) { return; }
	if (direction == 3 && y == 0       ) { return; }
	if (direction == 4 && z == mesh_z-1) { return; }
	if (direction == 5 && z == 0       ) { return; }

	x1 = x; y1 = y; z1 = z; // get the destination
	switch (direction) {
	case 0: 
		x1++;
		disp_e = disp_ex;
		break;
	case 1: 
		x1--; 
		disp_e = disp_ex;
		break;
	case 2: 
		y1++; 
		disp_e = disp_ey;
		break;
	case 3: 
		y1--; 
		disp_e = disp_ey;
		break;
	case 4: 
		z1++; 
		disp_e = disp_ez;
		break;
	case 5: 
		z1--; 
		disp_e = disp_ez;
		break;
	}

	message_org = (REALV*)malloc(K * sizeof(REALV));
	message     = SO1F2Message[plane][direction].m_pData[z1][y1][x1];

	//*
	Add2MessageSpatial_O1F2(message_org, x, y, z, plane, direction, K);
	/*/
	// initialize the message from the dual plane
	memcpy(message_org, DualMessage[plane].m_pData[z][y][x], sizeof(REALV) * K);

	// add the range term
	Add2Message(message_org, RangeTerm[plane].m_pData[z][y][x], K);
	
	// add spatial messages
	if (x > 0        && direction != 1) {	// add x- -> x+
		Add2Message(message_org, SO1F2Message[plane][0].m_pData[z][y][x], K);
	}
	if (x < mesh_x-1 && direction != 0) {	// add x+ -> x-
		Add2Message(message_org, SO1F2Message[plane][1].m_pData[z][y][x], K);
	}
	if (y > 0        && direction != 3) {	// add y- -> y+
		Add2Message(message_org, SO1F2Message[plane][2].m_pData[z][y][x], K);
	}
	if (y < mesh_y-1 && direction != 2) {	// add y+ -> y-
		Add2Message(message_org, SO1F2Message[plane][3].m_pData[z][y][x], K);
	}
	if (z > 0        && direction != 5) {	// add z- -> z+
		Add2Message(message_org, SO1F2Message[plane][4].m_pData[z][y][x], K);
	}
	if (z < mesh_z-1 && direction != 4) {	// add z+ -> z-
		Add2Message(message_org, SO1F2Message[plane][5].m_pData[z][y][x], K);
	}
	//*/

#if 0
	//////////////////////////////////////////
	REALV d0, Min;
#ifdef O1_USE_OFFSET
	d0 = Offset[plane]->m_pData[z1][y1][x1][0] - Offset[plane]->m_pData[z][y][x][0];
#else
	d0 = 0;
#endif
	Min = ComputeSpatialMessageDT(message, message_org, message_org, x, y, z, d0, plane, K, nL);
	// normalize the message by subtracting the minimum value
	SubtractMin(message, K, Min);
	//////////////////////////////////////////
#else
	//////////////////////////////////////////
	REALV d0, s, T;
	REALV _2s, xv;
	REALV *m_fx0, *m_fx1;
	REALV r_fx1;
	REALV fx0_min, fx0_min_T;
	int k, k0, k1;
	REALV* zv;
	int* v;
	REALV delta_1, delta_fx1;
	REALV *Y1, *Y3;
	//
#ifdef O1_USE_OFFSET
	d0 = Offset[plane]->m_pData[z][y][x][0] - Offset[plane]->m_pData[z1][y1][x1][0];
#else
	d0 = 0;
#endif
	s = alpha_O1;
	T = d_O1;
	_2s = 0.5f / s;
	m_fx0 = message_org;
	m_fx1 = message;
	r_fx1 = 1;
	//r_fx1 = CTRW;
	//
	zv = (REALV*)malloc((K+2) * sizeof(REALV));
	v = (int*)malloc(K * sizeof(int));
	Y1 = (REALV*)malloc(K * sizeof(REALV));
	Y3 = (REALV*)malloc(K * sizeof(REALV));
	//
#if 0
	//////////////////////////////////////////
	for (k1 = 0; k1 < K; k1++) {
		delta_1 = min(s * fabs(d0+disp_e[0]-disp_e[k1]), T) + m_fx0[0];
		for (k0 = 0; k0 < K; k0++) {
			v_fx = min(s * fabs(d0+disp_e[k0]-disp_e[k1]), T) + m_fx0[k0];
			TRUNCATE(delta_1, v_fx);
		}
		Y1[k1] = r_fx1 * delta_1 + (r_fx1-1)*m_fx1[k1];
	}
	//////////////////////////////////////////
#else
//#define TEST_MC
#ifdef TEST_MC
	//////////////////////////////////////////
	for (k1 = 0; k1 < K; k1++) {
		delta_1 = min(s * fabs(d0+disp_e[0]-disp_e[k1]), T) + m_fx0[0];
		for (k0 = 0; k0 < K; k0++) {
			v_fx = min(s * fabs(d0+disp_e[k0]-disp_e[k1]), T) + m_fx0[k0];
			TRUNCATE(delta_1, v_fx);
		}
		Y1[k1] = r_fx1 * delta_1 + (r_fx1-1)*m_fx1[k1];
	}
	//////////////////////////////////////////
#endif
	//
	//////////////////////////////////////////
	fx0_min = m_fx0[0];
	for (k0 = 1; k0 < K; k0++) {
		TRUNCATE(fx0_min, m_fx0[k0]);
	}
	fx0_min_T = fx0_min + T;
	//////////////////////////////////////////
	// DT
	k = 0;
	v[0] = 0;
	zv[0] = -INFINITE_S;
	zv[1] = INFINITE_S;
	for (k1 = 1; k1 < K; k1++) {
		xv = ((m_fx0[k1] + s*disp_e[k1]) - (m_fx0[v[k]] - s*disp_e[v[k]])) * _2s;
		if (xv > disp_e[v[k]] && xv < disp_e[k1]) {
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
		} else if ((xv == disp_e[v[k]]) || (xv == disp_e[k1])) {
			if (m_fx0[k1] < m_fx0[v[k]]) {
				v[k] = k1;
				zv[k+1] = INFINITE_S;
			}
		} else {
			if (k == 0) {
				if (m_fx0[k1] < m_fx0[v[0]]) {
					v[0] = k1;
					zv[0] = -INFINITE_S;
					zv[1] = INFINITE_S;
				}
			} else {
				if (m_fx0[k1] < m_fx0[v[k]]) {
					k--;
					//
					k1--;
					continue;
				}
			}
		}
	}
	k = 0;
	delta_fx1 = INFINITE_S;
	for (k1 = 0; k1 < K; k1++) {
		while (zv[k+1] < -d0 + disp_e[k1]) {
			k++;
		}
		delta_1 = MIN(s * fabs(d0+disp_e[v[k]]-disp_e[k1]), T) + m_fx0[v[k]];
		//delta_1 = D[v[k]-k1+L2] + m_fx0[v[k]];
		// apply truncation
		delta_1 = MIN(delta_1, fx0_min_T);
		//
		/*
		Y3[k1] = r_fx1 * (delta_1-delta_f) + (r_fx1-1)*m_fx1[k1];
		if (abs(Y1[k1] - Y3[k1]) > 0.00001) {
			Y1[k1] = Y1[k1];
		}
		//*/
		Y3[k1] = delta_1;
	}
	//////////////////////////////////////////
	for (k1 = 0; k1 < K; k1++) { 
#ifndef TEST_MC
		Y1[k1] = r_fx1 * Y3[k1] + (r_fx1-1)*m_fx1[k1];
#else
		Y3[k1] = r_fx1 * Y3[k1] + (r_fx1-1)*m_fx1[k1];
		if (abs(Y1[k1] - Y3[k1]) > 0.001) {
			TRACE("O1: %f != %f\r\n", Y1[k1], Y3[k1]);
		}
#endif
	}
	//////////////////////////////////////////
#endif

	//////////////////////////////////////////
	delta_1 = Y1[0];
	for (k1 = 1; k1 < K; k1++) { 
		TRUNCATE(delta_1, Y1[k1]);
	}
	//////////////////////////////////////////
	for (k1 = 0; k1 < K; k1++) { 
		m_fx1[k1] = Y1[k1] - delta_1;
	}
	//////////////////////////////////////////

	//////////////////////////////////////////
	// Updating lower bound
	//////////////////////////////////////////
	//lowerBound += delta_1;
	//////////////////////////////////////////

	free(zv);
	free(v);
	free(Y1);
	free(Y3);
#endif

	free(message_org);
}

void VolumeBP::UpdateSpatialMessage_O2F3(int x1, int y1, int z1, int plane, int dir1)
{
	int x0, y0, z0, dir0;
	int x2, y2, z2, dir2;
	REALV *m_fx0, *m_fx1, *m_fx2;
	REALV *m_fx0_u, *m_fx1_u, *m_fx2_u;
	REALV *Y0, *Y1, *Y2;
	REALV *D = NULL; // to prevent compiler warning
	int y, k0, k1, k2;
	REALV delta_0, delta_1, delta_2, v_fx;

	x0 = y0 = z0 = dir0 = 0; // to prevent compiler warning
	x2 = y2 = z2 = dir2 = 0; // to prevent compiler warning

	// eliminate impossible messages
	if (dir1 == 1 && (x1 <= 0 || x1 >= mesh_x-1)) { return; }
	if (dir1 == 4 && (y1 <= 0 || y1 >= mesh_y-1)) { return; }
	if (dir1 == 7 && (z1 <= 0 || z1 >= mesh_z-1)) { return; }

	switch (dir1) {
	case 1: 
		x0 = x1-1; y0 = y1  ; z0 = z1  ; dir0 = 0;
		x2 = x1+1; y2 = y1  ; z2 = z1  ; dir2 = 2;
		D = SO2[plane][0].m_pData[z1][y1][x1];
		break;
	case 4:
		x0 = x1  ; y0 = y1-1; z0 = z1  ; dir0 = 3;
		x2 = x1  ; y2 = y1+1; z2 = z1  ; dir2 = 5;
		D = SO2[plane][1].m_pData[z1][y1][x1];
		break;
	case 7:
		x0 = x1  ; y0 = y1  ; z0 = z1-1; dir0 = 6;
		x2 = x1  ; y2 = y1  ; z2 = z1+1; dir2 = 8;
		D = SO2[plane][2].m_pData[z1][y1][x1];
		break;
	}

	m_fx0_u = (REALV*)malloc(K * sizeof(REALV));
	m_fx1_u = (REALV*)malloc(K * sizeof(REALV));
	m_fx2_u = (REALV*)malloc(K * sizeof(REALV));
	Y0 = (REALV*)malloc(L8_1 * sizeof(REALV));
	Y1 = (REALV*)malloc(L8_1 * sizeof(REALV));
	Y2 = (REALV*)malloc(L8_1 * sizeof(REALV));

	Add2MessageSpatial_O2F3(m_fx0_u, x0, y0, z0, plane, dir0, K);
	Add2MessageSpatial_O2F3(m_fx1_u, x1, y1, z1, plane, dir1, K);
	Add2MessageSpatial_O2F3(m_fx2_u, x2, y2, z2, plane, dir2, K);

	m_fx0 = SO2F3Message[plane][dir0].m_pData[z0][y0][x0];
	m_fx1 = SO2F3Message[plane][dir1].m_pData[z1][y1][x1];
	m_fx2 = SO2F3Message[plane][dir2].m_pData[z2][y2][x2];

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
	SubtractMin(m_fx0_u, m_fx0, K, delta_0);
	//////////////////////////////////////////
	for (k1 = 0; k1 < K; k1++) {
		delta_1 = D[L2-2*k1+L2] + Y1[L2];
		for (y = L2; y <= L6; y++) {
			v_fx = D[y-2*k1+L2] + Y1[y];
			TRUNCATE(delta_1, v_fx);
		}
		m_fx1_u[k1] = delta_1;
	}
	SubtractMin(m_fx1_u, m_fx1, K, delta_1);
	//////////////////////////////////////////
	for (k2 = 0; k2 < K; k2++) {
		delta_2 = D[L+k2-L] + Y2[L];
		for (y = L; y <= L7; y++) {
			v_fx = D[y+k2-L] + Y2[y];
			TRUNCATE(delta_2, v_fx);
		}
		m_fx2_u[k2] = delta_2;
	}
	SubtractMin(m_fx2_u, m_fx2, K, delta_2);
	//////////////////////////////////////////

	free(m_fx0_u);
	free(m_fx1_u);
	free(m_fx2_u);
	free(Y0);
	free(Y1);
	free(Y2);
}

void VolumeBP::UpdateDualMessage(int x, int y, int z, int plane)
{
	REALV *m_fx0_b, *m_fx1_b, *m_fx2_b;
	REALV *m_fx0_u, *m_fx1_u, *m_fx2_u;
	REALV *m_fx0, *m_fx1, *m_fx2;
	REALV *Dm;
	REALV v_fx, delta_0, delta_1, delta_2;
	int k0, k1, k2, kk, k2_K_2, k1_K;

	m_fx0_b = (REALV*)malloc(K * sizeof(REALV));
	m_fx1_b = (REALV*)malloc(K * sizeof(REALV));
	m_fx2_b = (REALV*)malloc(K * sizeof(REALV));
	m_fx0_u = (REALV*)malloc(K * sizeof(REALV));
	m_fx1_u = (REALV*)malloc(K * sizeof(REALV));
	m_fx2_u = (REALV*)malloc(K * sizeof(REALV));

	//////////////////////////////////////////
	//////////////////////////////////////////
	Add2MessageDual(m_fx0_b, x, y, z, 0, K);
	Add2MessageDual(m_fx1_b, x, y, z, 1, K);
	Add2MessageDual(m_fx2_b, x, y, z, 2, K);
	//////////////////////////////////////////
	//////////////////////////////////////////

	//////////////////////////////////////////
	//////////////////////////////////////////
	Dm = dcv->m_pData[z][y][x];
	m_fx0 = DualMessage[0].m_pData[z][y][x];
	m_fx1 = DualMessage[1].m_pData[z][y][x];
	m_fx2 = DualMessage[2].m_pData[z][y][x];

	//*
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
	/*/
	REALV *Dt;

	r_fx0 = 1;
	r_fx1 = 1;
	r_fx2 = 1;

	Dt = (REALV*)malloc(num_d * sizeof(REALV));

	delta_f = INFINITE_S;
	for (k2 = 0; k2 < K; k2++) {
		k2_K_2 = k2*K_2;
		m_fx2_k2 = m_fx2_b[k2];
		for (k1 = 0; k1 < K; k1++) {
			kk = k2_K_2 + k1*K;
			fxfx = m_fx1_b[k1] + m_fx2_k2;
			for (k0 = 0; k0 < K; k0++) {
				v_fx = Dm[kk + k0] + m_fx0_b[k0] + fxfx;
				TRUNCATE(delta_f, v_fx);
				//
				Dt[kk + k0] = v_fx;
			}
		}
	}

	for (k0 = 0; k0 < K; k0++) {
		delta_0 = INFINITE_S;
		for (k2 = 0; k2 < K; k2++) {
			kk = k2*K_2 + k0;
			for (k1 = 0; k1 < K; k1++) {
				v_fx = Dt[kk + k1*K];
				TRUNCATE(delta_0, v_fx);
			}
		}
		m_fx0_u[k0] = r_fx0 * (delta_0-delta_f) - m_fx0_b[k0];
	}

	for (k1 = 0; k1 < K; k1++) {
		delta_1 = INFINITE_S;
		k1_K = k1 * K;
		for (k2 = 0; k2 < K; k2++) {
			kk = k2*K_2 + k1_K;
			for (k0 = 0; k0 < K; k0++) {
				v_fx = Dt[kk + k0];
				TRUNCATE(delta_1, v_fx);
			}
		}
		m_fx1_u[k1] = r_fx1 * (delta_1-delta_f) - m_fx1_b[k1];
	}

	for (k2 = 0; k2 < K; k2++) {
		delta_2 = INFINITE_S;
		k2_K_2 = k2 * K_2;
		for (k1 = 0; k1 < K; k1++) {
			kk = k2_K_2 + k1 * K;
			for (k0 = 0; k0 < K; k0++) {
				v_fx = Dt[kk + k0];
				TRUNCATE(delta_2, v_fx);
			}
		}
		m_fx2_u[k2] = r_fx2 * (delta_2-delta_f) - m_fx2_b[k2];
	}

	free(Dt);
	//*/
	//////////////////////////////////////////
	//////////////////////////////////////////

	//////////////////////////////////////////
	//////////////////////////////////////////
	//if (plane != 0) {
		SubtractMin(m_fx0_u, m_fx0, K, delta_0);
	//}
	//////////////////////////////////////////
	//if (plane != 1) {
		SubtractMin(m_fx1_u, m_fx1, K, delta_1);
	//}
	//////////////////////////////////////////
	//if (plane != 2) {
		SubtractMin(m_fx2_u, m_fx2, K, delta_2);
	//}
	//////////////////////////////////////////
	//////////////////////////////////////////

	free(m_fx0_b);
	free(m_fx1_b);
	free(m_fx2_b);
	free(m_fx0_u);
	free(m_fx1_u);
	free(m_fx2_u);
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
// Update Message for TRW_S
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::UpdateMessage_TRW_S_FW_O1F2(int x, int y, int z, int plane)
{
	int i, k;
	BOOL update_s[6];
	REALV *m_fx;
	REALV *m_fx_b;
	REALV *m_fx_u;
	int ns;
	REALV r, vMin;

	m_fx_b = (REALV*)malloc(K * sizeof(REALV));
	m_fx_u = (REALV*)malloc(K * sizeof(REALV));

	// add the range term
	memcpy(m_fx_b, RangeTerm[plane].m_pData[z][y][x], K * sizeof(REALV));

	// add spatial messages
	for (i = 0; i < 6; i++) {
		update_s[i] = TRUE;
	}
	ns = 7;
	if (x > 0) {
		Add2Message(m_fx_b, SO1F2Message[plane][0].m_pData[z][y][x], K);
	} else {
		update_s[0] = FALSE; ns--;
	}
	if (x < mesh_x-1) {
		Add2Message(m_fx_b, SO1F2Message[plane][1].m_pData[z][y][x], K);
	} else {
		update_s[1] = FALSE; ns--;
	}
	if (y > 0) {
		Add2Message(m_fx_b, SO1F2Message[plane][2].m_pData[z][y][x], K);
	} else {
		update_s[2] = FALSE; ns--;
	}
	if (y < mesh_y-1) {
		Add2Message(m_fx_b, SO1F2Message[plane][3].m_pData[z][y][x], K);
	} else {
		update_s[3] = FALSE; ns--;
	}
	if (z > 0) {
		Add2Message(m_fx_b, SO1F2Message[plane][4].m_pData[z][y][x], K);
	} else {
		update_s[4] = FALSE; ns--;
	}
	if (z < mesh_z-1) {
		Add2Message(m_fx_b, SO1F2Message[plane][5].m_pData[z][y][x], K);
	} else {
		update_s[5] = FALSE; ns--;
	}
	r = 1.0 / ns;

	Add2Message(m_fx_b, DualMessage[plane].m_pData[z][y][x], K);

	for (i = 0; i < 6; i++) {
		if (!update_s[i]) { continue; }

		m_fx = SO1F2Message[plane][i].m_pData[z][y][x];

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
		m_fx = DualMessage[plane].m_pData[z][y][x];

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

	free(m_fx_b);
	free(m_fx_u);
}

void VolumeBP::UpdateMessage_TRW_S_FW_O2F3(int x, int y, int z, int plane)
{
	int i, k;
	BOOL update_s[9];
	REALV *m_fx;
	REALV *m_fx_b;
	REALV *m_fx_u;
	int ns;
	REALV r, vMin;

	m_fx_b = (REALV*)malloc(K * sizeof(REALV));
	m_fx_u = (REALV*)malloc(K * sizeof(REALV));

	// add the range term
	memcpy(m_fx_b, RangeTerm[plane].m_pData[z][y][x], K * sizeof(REALV));

	// add spatial messages
	for (i = 0; i < 9; i++) {
		update_s[i] = TRUE;
	}
	ns = 10;
	if (x < mesh_x-2) {				// f+ -> x
		Add2Message(m_fx_b, SO2F3Message[plane][0].m_pData[z][y][x], K);
	} else {
		update_s[0] = FALSE; ns--;
	}
	if (x > 0 && x < mesh_x-1) {	// f0 -> x
		Add2Message(m_fx_b, SO2F3Message[plane][1].m_pData[z][y][x], K);
	} else {
		update_s[1] = FALSE; ns--;
	}
	if (x > 1) {					// f- -> x
		Add2Message(m_fx_b, SO2F3Message[plane][2].m_pData[z][y][x], K);
	} else {
		update_s[2] = FALSE; ns--;
	}
	if (y < mesh_y-2) {
		Add2Message(m_fx_b, SO2F3Message[plane][3].m_pData[z][y][x], K);
	} else {
		update_s[3] = FALSE; ns--;
	}
	if (y > 0 && y < mesh_y-1) {
		Add2Message(m_fx_b, SO2F3Message[plane][4].m_pData[z][y][x], K);
	} else {
		update_s[4] = FALSE; ns--;
	}
	if (y > 1) {
		Add2Message(m_fx_b, SO2F3Message[plane][5].m_pData[z][y][x], K);
	} else {
		update_s[5] = FALSE; ns--;
	}
	if (z < mesh_z-2) {
		Add2Message(m_fx_b, SO2F3Message[plane][6].m_pData[z][y][x], K);
	} else {
		update_s[6] = FALSE; ns--;
	}
	if (z > 0 && z < mesh_z-1) {
		Add2Message(m_fx_b, SO2F3Message[plane][7].m_pData[z][y][x], K);
	} else {
		update_s[7] = FALSE; ns--;
	}
	if (z > 1) {
		Add2Message(m_fx_b, SO2F3Message[plane][8].m_pData[z][y][x], K);
	} else {
		update_s[8] = FALSE; ns--;
	}

	r = 1.0 / ns;

	Add2Message(m_fx_b, DualMessage[plane].m_pData[z][y][x], K);

	for (i = 0; i < 9; i++) {
		if (!update_s[i]) { continue; }

		m_fx = SO2F3Message[plane][i].m_pData[z][y][x];

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
		m_fx = DualMessage[plane].m_pData[z][y][x];

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

	free(m_fx_b);
	free(m_fx_u);
}

void VolumeBP::UpdateMessage_TRW_S_FW_O1F2_O2F3(int x, int y, int z, int plane)
{
	int i, k;
	BOOL update_s_O1[9];
	BOOL update_s_O2[9];
	REALV *m_fx;
	REALV *m_fx_b;
	REALV *m_fx_u;
	int ns_O1;
	int ns_O2;
	REALV r, vMin;

	m_fx_b = (REALV*)malloc(K * sizeof(REALV));
	m_fx_u = (REALV*)malloc(K * sizeof(REALV));

	// add the range term
	memcpy(m_fx_b, RangeTerm[plane].m_pData[z][y][x], K * sizeof(REALV));

	// add spatial messages for O1F2
	for (i = 0; i < 6; i++) {
		update_s_O1[i] = TRUE;
	}
	ns_O1 = 6;
	if (x > 0) {
		Add2Message(m_fx_b, SO1F2Message[plane][0].m_pData[z][y][x], K);
	} else {
		update_s_O1[0] = FALSE; ns_O1--;
	}
	if (x < mesh_x-1) {
		Add2Message(m_fx_b, SO1F2Message[plane][1].m_pData[z][y][x], K);
	} else {
		update_s_O1[1] = FALSE; ns_O1--;
	}
	if (y > 0) {
		Add2Message(m_fx_b, SO1F2Message[plane][2].m_pData[z][y][x], K);
	} else {
		update_s_O1[2] = FALSE; ns_O1--;
	}
	if (y < mesh_y-1) {
		Add2Message(m_fx_b, SO1F2Message[plane][3].m_pData[z][y][x], K);
	} else {
		update_s_O1[3] = FALSE; ns_O1--;
	}
	if (z > 0) {
		Add2Message(m_fx_b, SO1F2Message[plane][4].m_pData[z][y][x], K);
	} else {
		update_s_O1[4] = FALSE; ns_O1--;
	}
	if (z < mesh_z-1) {
		Add2Message(m_fx_b, SO1F2Message[plane][5].m_pData[z][y][x], K);
	} else {
		update_s_O1[5] = FALSE; ns_O1--;
	}

	// add spatial messages for O2F3
	for (i = 0; i < 9; i++) {
		update_s_O2[i] = TRUE;
	}
	ns_O2 = 9;
	if (x < mesh_x-2) {				// f+ -> x
		Add2Message(m_fx_b, SO2F3Message[plane][0].m_pData[z][y][x], K);
	} else {
		update_s_O2[0] = FALSE; ns_O2--;
	}
	if (x > 0 && x < mesh_x-1) {	// f0 -> x
		Add2Message(m_fx_b, SO2F3Message[plane][1].m_pData[z][y][x], K);
	} else {
		update_s_O2[1] = FALSE; ns_O2--;
	}
	if (x > 1) {					// f- -> x
		Add2Message(m_fx_b, SO2F3Message[plane][2].m_pData[z][y][x], K);
	} else {
		update_s_O2[2] = FALSE; ns_O2--;
	}
	if (y < mesh_y-2) {
		Add2Message(m_fx_b, SO2F3Message[plane][3].m_pData[z][y][x], K);
	} else {
		update_s_O2[3] = FALSE; ns_O2--;
	}
	if (y > 0 && y < mesh_y-1) {
		Add2Message(m_fx_b, SO2F3Message[plane][4].m_pData[z][y][x], K);
	} else {
		update_s_O2[4] = FALSE; ns_O2--;
	}
	if (y > 1) {
		Add2Message(m_fx_b, SO2F3Message[plane][5].m_pData[z][y][x], K);
	} else {
		update_s_O2[5] = FALSE; ns_O2--;
	}
	if (z < mesh_z-2) {
		Add2Message(m_fx_b, SO2F3Message[plane][6].m_pData[z][y][x], K);
	} else {
		update_s_O2[6] = FALSE; ns_O2--;
	}
	if (z > 0 && z < mesh_z-1) {
		Add2Message(m_fx_b, SO2F3Message[plane][7].m_pData[z][y][x], K);
	} else {
		update_s_O2[7] = FALSE; ns_O2--;
	}
	if (z > 1) {
		Add2Message(m_fx_b, SO2F3Message[plane][8].m_pData[z][y][x], K);
	} else {
		update_s_O2[8] = FALSE; ns_O2--;
	}

	r = 1.0 / (ns_O1 + ns_O2 + 1);

	Add2Message(m_fx_b, DualMessage[plane].m_pData[z][y][x], K);

	for (i = 0; i < 6; i++) {
		if (!update_s_O1[i]) { continue; }

		m_fx = SO1F2Message[plane][i].m_pData[z][y][x];

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

		m_fx = SO2F3Message[plane][i].m_pData[z][y][x];

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
		m_fx = DualMessage[plane].m_pData[z][y][x];

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

	free(m_fx_b);
	free(m_fx_u);
}

void VolumeBP::UpdateMessage_TRW_S_BW(int x, int y, int z, int plane)
{
	int k;
	REALV *m_fx_b;
	REALV vMin;

	m_fx_b = (REALV*)malloc(K * sizeof(REALV));

	/*
	// add the range term
	memcpy(m_fx_b, RangeTerm[plane].m_pData[z][y][x], K * sizeof(REALV));

	// add spatial messages
	if (x > 0) {
		Add2Message(m_fx_b, SO1F2Message[plane][0].m_pData[z][y][x], K);
	}
	if (x < mesh_x-1) {
		Add2Message(m_fx_b, SO1F2Message[plane][1].m_pData[z][y][x], K);
	}
	if (y > 0) {
		Add2Message(m_fx_b, SO1F2Message[plane][2].m_pData[z][y][x], K);
	}
	if (y < mesh_y-1) {
		Add2Message(m_fx_b, SO1F2Message[plane][3].m_pData[z][y][x], K);
	}
	if (z > 0) {
		Add2Message(m_fx_b, SO1F2Message[plane][4].m_pData[z][y][x], K);
	}
	if (z < mesh_z-1) {
		Add2Message(m_fx_b, SO1F2Message[plane][5].m_pData[z][y][x], K);
	}
	/*/
	Add2MessageDual(m_fx_b, x, y, z, plane, K);
	//*/

	Add2Message(m_fx_b, DualMessage[plane].m_pData[z][y][x], K);

	vMin = m_fx_b[0];
	for (k = 1; k < K; k++) {
		TRUNCATE(vMin, m_fx_b[k]);
	}

	lowerBound += vMin;

	free(m_fx_b);
}

void VolumeBP::UpdateSpatialMessage_TRW_S_BW_O1F2(int x, int y, int z, int plane, int direction)
{
	int direction1 = 0;
	int x1, y1, z1;
	REALV *m_fx0, *m_fx1;
	REALV *m_buf;
	REALV delta_f, delta_0, delta_1;
	REALV *disp_e = NULL;

	x1 = x; y1 = y; z1 = z; // destination
	if (direction == 0) {
		if (x == mesh_x-1) { return; }
		x1++;
		disp_e = disp_ex;
		direction1 = 1;
	} else if (direction == 1) {
		if (x == 0       ) { return; }
		x1--;
		disp_e = disp_ex;
		direction1 = 0;
	} else if (direction == 2) {
		if (y == mesh_y-1) { return; }
		y1++;
		disp_e = disp_ey;
		direction1 = 3;
	} else if (direction == 3) {
		if (y == 0       ) { return; }
		y1--;
		disp_e = disp_ey;
		direction1 = 2;
	} else if (direction == 4) {
		if (z == mesh_z-1) { return; }
		z1++;
		disp_e = disp_ez;
		direction1 = 5;
	} else if (direction == 5) {
		if (z == 0       ) { return; }
		z1--;
		disp_e = disp_ez;
		direction1 = 4;
	}

	//////////////////////////////////////////
	//////////////////////////////////////////
	REALV d0, s, T;
	REALV v_fx;
	REALV *Y0, *Y1;
	REALV r_fx0, r_fx1;
	int k0, k1;

	m_fx0 = SO1F2Message[plane][direction1].m_pData[z ][y ][x ];
	m_fx1 = SO1F2Message[plane][direction ].m_pData[z1][y1][x1];
	r_fx0 = r_fx1 = 0.5;
#ifdef O1_USE_OFFSET
	d0 = Offset[plane]->m_pData[z1][y1][x1][0] - Offset[plane]->m_pData[z][y][x][0];
#else
	d0 = 0;
#endif
	s = alpha_O1;
	T = d_O1;

	//////////////////////////////////////////
	/*
	delta_f = INFINITE_S;
	for (k1 = 0; k1 < K; k1++) {
		for (k0 = 0; k0 < K; k0++) {
			v_fx = min(s * fabs(d0+disp_e[k0]-disp_e[k1]), T) + m_fx0[k0] + m_fx1[k1];
			TRUNCATE(delta_f, v_fx);
		}
	}
	/*/
	delta_f = 0;
	//*/
	//////////////////////////////////////////

	//////////////////////////////////////////
	m_buf = (REALV*)malloc(K * sizeof(REALV));
	Y0 = (REALV*)malloc(K * sizeof(REALV));
	Y1 = (REALV*)malloc(K * sizeof(REALV));
	//////////////////////////////////////////

	/*
	//////////////////////////////////////////
	ComputeSpatialMessageDT(Y0, m_fx1, m_buf, x, y, z, d0, plane, K, nL);
	for (k0 = 0; k0 < K; k0++) {
		Y0[k0] = r_fx0 * (Y0[k0]-delta_f) + (r_fx0-1)*m_fx0[k0];
	}
	delta_0 = Y0[0];
	for (k0 = 1; k0 < K; k0++) { 
		TRUNCATE(delta_0, Y0[k0]);
	}
	//
	ComputeSpatialMessageDT(Y1, m_fx0, m_buf, x1, y1, z1, -d0, plane, K, nL);
	for (k1 = 0; k1 < K; k1++) {
		Y1[k1] = r_fx1 * (Y1[k1]-delta_f) + (r_fx1-1)*m_fx1[k1];
	}
	delta_1 = Y1[0];
	for (k1 = 1; k1 < K; k1++) { 
		TRUNCATE(delta_1, Y1[k1]);
	}
	//////////////////////////////////////////
	/*/
	//////////////////////////////////////////
	for (k0 = 0; k0 < K; k0++) {
		delta_0 = MIN(s * fabs(-d0+disp_e[k0]-disp_e[0]), T) + m_fx1[0];
		for (k1 = 0; k1 < K; k1++) {
			v_fx = MIN(s * fabs(-d0+disp_e[k0]-disp_e[k1]), T) + m_fx1[k1];
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
		delta_1 = MIN(s * fabs(-d0+disp_e[0]-disp_e[k1]), T) + m_fx0[0];
		for (k0 = 0; k0 < K; k0++) {
			v_fx = MIN(s * fabs(-d0+disp_e[k0]-disp_e[k1]), T) + m_fx0[k0];
			TRUNCATE(delta_1, v_fx);
		}
		Y1[k1] = r_fx1 * (delta_1-delta_f) + (r_fx1-1)*m_fx1[k1];
	}
	delta_1 = Y1[0];
	for (k1 = 1; k1 < K; k1++) { 
		TRUNCATE(delta_1, Y1[k1]);
	}
	//////////////////////////////////////////
	//*/

	//////////////////////////////////////////
	for (k0 = 0; k0 < K; k0++) { 
		m_fx0[k0] = Y0[k0] - delta_0;
	}
	for (k1 = 0; k1 < K; k1++) { 
		m_fx1[k1] = Y1[k1] - delta_1;
	}
	//////////////////////////////////////////
	
	//////////////////////////////////////////
	lowerBound += delta_f + delta_0 + delta_1;
	//////////////////////////////////////////

	free(m_buf);
	free(Y0);
	free(Y1);
	//////////////////////////////////////////
	//////////////////////////////////////////
}

void VolumeBP::UpdateSpatialMessage_TRW_S_BW_O2F3(int x1, int y1, int z1, int plane, int dir1)
{
	int x0, y0, z0, dir0;
	int x2, y2, z2, dir2;
	REALV *m_fx0, *m_fx1, *m_fx2;
	REALV *m_fx0_u, *m_fx1_u, *m_fx2_u;
	REALV *Y0, *Y1, *Y2, *Yf;
	REALV *D = NULL; // to prevent compiler warning
	int y, yy, k, k0, k1, k2;
	REALV delta_0, delta_1, delta_2, delta_f, v_fx;
	REALV r_fx0, r_fx1, r_fx2;

	x0 = y0 = z0 = dir0 = 0; // to prevent compiler warning
	x2 = y2 = z2 = dir2 = 0; // to prevent compiler warning

	// eliminate impossible messages
	if (dir1 == 1 && (x1 <= 0 || x1 >= mesh_x-1)) { return; }
	if (dir1 == 4 && (y1 <= 0 || y1 >= mesh_y-1)) { return; }
	if (dir1 == 7 && (z1 <= 0 || z1 >= mesh_z-1)) { return; }

	switch (dir1) {
	case 1: 
		x0 = x1-1; y0 = y1  ; z0 = z1  ; dir0 = 0;
		x2 = x1+1; y2 = y1  ; z2 = z1  ; dir2 = 2;
		D = SO2[plane][0].m_pData[z1][y1][x1];
		break;
	case 4:
		x0 = x1  ; y0 = y1-1; z0 = z1  ; dir0 = 3;
		x2 = x1  ; y2 = y1+1; z2 = z1  ; dir2 = 5;
		D = SO2[plane][1].m_pData[z1][y1][x1];
		break;
	case 7:
		x0 = x1  ; y0 = y1  ; z0 = z1-1; dir0 = 6;
		x2 = x1  ; y2 = y1  ; z2 = z1+1; dir2 = 8;
		D = SO2[plane][2].m_pData[z1][y1][x1];
		break;
	}

	m_fx0_u = (REALV*)malloc(K * sizeof(REALV));
	m_fx1_u = (REALV*)malloc(K * sizeof(REALV));
	m_fx2_u = (REALV*)malloc(K * sizeof(REALV));
	Y0 = (REALV*)malloc(L8_1 * sizeof(REALV));
	Y1 = (REALV*)malloc(L8_1 * sizeof(REALV));
	Y2 = (REALV*)malloc(L8_1 * sizeof(REALV));
	Yf = (REALV*)malloc(L8_1 * sizeof(REALV));

	m_fx0 = SO2F3Message[plane][dir0].m_pData[z0][y0][x0];
	m_fx1 = SO2F3Message[plane][dir1].m_pData[z1][y1][x1];
	m_fx2 = SO2F3Message[plane][dir2].m_pData[z2][y2][x2];
	r_fx0 = r_fx1 = r_fx2 = (REALV)(1.0 / 3.0);

	//////////////////////////////////////////
	for (y = 0; y <= L8; y++) {
		Y0[y] = INFINITE_S;
		Y1[y] = INFINITE_S;
		Y2[y] = INFINITE_S;
		Yf[y] = INFINITE_S;
	}

	// make y = L4 when k0 = L, k1 = L, k2 = L

	// y = -2*x1 + x2 = -2*(k1-L) + (k2-L) = 2*k1 - k2 + L + (4*L)
	// y = [L, L7]
	for (k2 = 0; k2 < K; k2++) {
		for (k1 = 0; k1 < K; k1++) {
			y = -2*k1 + k2 + L5;
			delta_0 = m_fx1[k1] + m_fx2[k2];
			TRUNCATE(Y0[y], delta_0);
		}
	}
	// y = x0 + x2 = (k0-L) + (k2-L) = k0 + k2 - 2*L + (4*L)
	// y = [L2, L6]
	for (k2 = 0; k2 < K; k2++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 + k2 + L2;
			delta_1 = m_fx0[k0] + m_fx2[k2];
			TRUNCATE(Y1[y], delta_1);
		}
	}
	// y = x0 - 2*x1 = (k0-L) - 2*(k1-L) = k0 - 2*k1 + L + (4*L)
	// y = [L, L7]
	for (k1 = 0; k1 < K; k1++) {
		for (k0 = 0; k0 < K; k0++) {
			y = k0 - 2*k1 + L5;
			delta_2 = m_fx0[k0] + m_fx1[k1];
			TRUNCATE(Y2[y], delta_2);
		}
	}
	//
	// y = k0 - 2*k1 + k2 + (4*L) = k0 + y0 (= -2*k2 + k3 + 5*L) - L
	// y = [0, L8]
	for (k0 = 0; k0 < K; k0++) {
		for (y = L; y <= L7; y++) {
			yy = y + k0 - L;
			delta_f = m_fx0[k0] + Y0[y];
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
	SubtractMin(m_fx0_u, m_fx0, K, delta_0);
	//////////////////////////////////////////
	for (k1 = 0; k1 < K; k1++) {
		delta_1 = D[L2-2*k1+L2] + Y1[L2];
		for (y = L2; y <= L6; y++) {
			v_fx = D[y-2*k1+L2] + Y1[y];
			TRUNCATE(delta_1, v_fx);
		}
		m_fx1_u[k1] = r_fx1 * (delta_1-delta_f) + (r_fx1-1)*m_fx1[k1];
	}
	SubtractMin(m_fx1_u, m_fx1, K, delta_1);
	//////////////////////////////////////////
	for (k2 = 0; k2 < K; k2++) {
		delta_2 = D[L+k2-L] + Y2[L];
		for (y = L; y <= L7; y++) {
			v_fx = D[y+k2-L] + Y2[y];
			TRUNCATE(delta_2, v_fx);
		}
		m_fx2_u[k2] = r_fx2 * (delta_2-delta_f) + (r_fx2-1)*m_fx2[k2];
	}
	SubtractMin(m_fx2_u, m_fx2, K, delta_2);
	//////////////////////////////////////////

	//////////////////////////////////////////
	// Updating lower bound
	//////////////////////////////////////////
	lowerBound += delta_f + delta_0 + delta_1 + delta_2;
	//////////////////////////////////////////

	free(m_fx0_u);
	free(m_fx1_u);
	free(m_fx2_u);
	free(Y0);
	free(Y1);
	free(Y2);
	free(Yf);
}

void VolumeBP::UpdateDualMessage_TRW_S_BW(int x, int y, int z)
{
	REALV *m_fx0_u, *m_fx1_u, *m_fx2_u;
	REALV *m_fx0, *m_fx1, *m_fx2;
	REALV r_fx0, r_fx1, r_fx2;
	REALV *Dm, *Dt;
	REALV m_fx2_k2, fxfx, v_fx, delta_f, delta_0, delta_1, delta_2;
	int k0, k1, k2, kk, k2_K_2, k1_K;

	m_fx0_u = (REALV*)malloc(K * sizeof(REALV));
	m_fx1_u = (REALV*)malloc(K * sizeof(REALV));
	m_fx2_u = (REALV*)malloc(K * sizeof(REALV));
	Dt = (REALV*)malloc(num_d * sizeof(REALV));

	//////////////////////////////////////////
	//////////////////////////////////////////
	Dm = dcv->m_pData[z][y][x];
	m_fx0 = DualMessage[0].m_pData[z][y][x];
	m_fx1 = DualMessage[1].m_pData[z][y][x];
	m_fx2 = DualMessage[2].m_pData[z][y][x];
	r_fx0 = r_fx1 = r_fx2 = (REALV)(1.0 / 3.0);

	delta_f = INFINITE_S;
	for (k2 = 0; k2 < K; k2++) {
		k2_K_2 = k2*K_2;
		m_fx2_k2 = m_fx2[k2];
		for (k1 = 0; k1 < K; k1++) {
			kk = k2_K_2 + k1*K;
			fxfx = m_fx1[k1] + m_fx2_k2;
			for (k0 = 0; k0 < K; k0++) {
				v_fx = Dm[kk + k0] + m_fx0[k0] + fxfx;
				TRUNCATE(delta_f, v_fx);
				//
				Dt[kk + k0] = v_fx;
			}
		}
	}

	for (k0 = 0; k0 < K; k0++) {
		delta_0 = INFINITE_S;
		for (k2 = 0; k2 < K; k2++) {
			kk = k2*K_2 + k0;
			for (k1 = 0; k1 < K; k1++) {
				v_fx = Dt[kk + k1*K];
				TRUNCATE(delta_0, v_fx);
			}
		}
		m_fx0_u[k0] = r_fx0 * (delta_0-delta_f) - m_fx0[k0];
	}

	for (k1 = 0; k1 < K; k1++) {
		delta_1 = INFINITE_S;
		k1_K = k1 * K;
		for (k2 = 0; k2 < K; k2++) {
			kk = k2*K_2 + k1_K;
			for (k0 = 0; k0 < K; k0++) {
				v_fx = Dt[kk + k0];
				TRUNCATE(delta_1, v_fx);
			}
		}
		m_fx1_u[k1] = r_fx1 * (delta_1-delta_f) - m_fx1[k1];
	}

	for (k2 = 0; k2 < K; k2++) {
		delta_2 = INFINITE_S;
		k2_K_2 = k2 * K_2;
		for (k1 = 0; k1 < K; k1++) {
			kk = k2_K_2 + k1 * K;
			for (k0 = 0; k0 < K; k0++) {
				v_fx = Dt[kk + k0];
				TRUNCATE(delta_2, v_fx);
			}
		}
		m_fx2_u[k2] = r_fx2 * (delta_2-delta_f) - m_fx2[k2];
	}
	//////////////////////////////////////////
	//////////////////////////////////////////

	//////////////////////////////////////////
	//////////////////////////////////////////
	SubtractMin(m_fx0_u, m_fx0, K, delta_0);
	SubtractMin(m_fx1_u, m_fx1, K, delta_1);
	SubtractMin(m_fx2_u, m_fx2, K, delta_2);
	//////////////////////////////////////////
	//////////////////////////////////////////

	//////////////////////////////////////////
	//////////////////////////////////////////
	lowerBound += delta_f + delta_0 + delta_1 + delta_2;
	//////////////////////////////////////////
	//////////////////////////////////////////

	free(m_fx0_u);
	free(m_fx1_u);
	free(m_fx2_u);
	free(Dt);
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::BP_S(int count)
{
	int i, j, k;
	int l;
	//
#if 0
	l = count % 3;
	if (count % 6 < 3) {	// forward update
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					UpdateSpatialMessage(i, j, k, l, 0);
					UpdateSpatialMessage(i, j, k, l, 2);
					UpdateSpatialMessage(i, j, k, l, 4);
					if (count % 12 < 6) {
						UpdateDualMessage(i, j, k, l);
					}
				}
			}
		}
	} else {				// backward upate
		for (k = mesh_z-1; k >= 0; k--) {
			for (j = mesh_y-1; j >= 0; j--) {
				for (i = mesh_x-1; i >= 0; i--) {
					UpdateSpatialMessage(i, j, k, l, 1);
					UpdateSpatialMessage(i, j, k, l, 3);
					UpdateSpatialMessage(i, j, k, l, 5);
					if (count % 12 < 6) {
						UpdateDualMessage(i, j, k, l);
					}
				}
			}
		}
	}
#endif
#if 1
	if (in_scv_w_O1F2 != -2) {
		// forward update
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						UpdateSpatialMessage(i, j, k, l, 0);
						UpdateSpatialMessage(i, j, k, l, 2);
						UpdateSpatialMessage(i, j, k, l, 4);
					}
				}
			}
		}
		// backward upate
		for (l = 0; l < 3; l++) {
			for (k = mesh_z-1; k >= 0; k--) {
				for (j = mesh_y-1; j >= 0; j--) {
					for (i = mesh_x-1; i >= 0; i--) {
						UpdateSpatialMessage(i, j, k, l, 1);
						UpdateSpatialMessage(i, j, k, l, 3);
						UpdateSpatialMessage(i, j, k, l, 5);
					}
				}
			}
		}
		// dual update
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					UpdateDualMessage(i, j, k, 0);
				}
			}
		}
	} else if (in_scv_w_O2F3 != -2) {
		// spatial update
		//*
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 1);
						UpdateSpatialMessage_O2F3(i, j, k, l, 4);
						UpdateSpatialMessage_O2F3(i, j, k, l, 7);
					}
				}
			}
		}
		/*/
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i+=3) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 1);
					}
				}
			}
		}
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j+=3) {
					for (i = 0; i < mesh_x; i++) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 4);
					}
				}
			}
		}
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k+=3) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 7);
					}
				}
			}
		}
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 1; i < mesh_x; i+=3) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 1);
					}
				}
			}
		}
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 1; j < mesh_y; j+=3) {
					for (i = 0; i < mesh_x; i++) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 4);
					}
				}
			}
		}
		for (l = 0; l < 3; l++) {
			for (k = 1; k < mesh_z; k+=3) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 7);
					}
				}
			}
		}
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 2; i < mesh_x; i+=3) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 1);
					}
				}
			}
		}
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 2; j < mesh_y; j+=3) {
					for (i = 0; i < mesh_x; i++) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 4);
					}
				}
			}
		}
		for (l = 0; l < 3; l++) {
			for (k = 2; k < mesh_z; k+=3) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						UpdateSpatialMessage_O2F3(i, j, k, l, 7);
					}
				}
			}
		}
		//*/
		// dual update
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					UpdateDualMessage(i, j, k, 0);
				}
			}
		}
	}
#endif
}
void VolumeBP::TRW_S(int count)
{
	int i, j, k, l;
	//
	if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 == -2)) {
		// forward pass
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						UpdateMessage_TRW_S_FW_O1F2(i, j, k, l);
					}
				}
			}
		}
		// backward pass
		for (l = 2; l >= 0; l--) {
			for (k = mesh_z-1; k >= 0; k--) {
				for (j = mesh_y-1; j >= 0; j--) {
					for (i = mesh_x-1; i >= 0; i--) {
						UpdateSpatialMessage_TRW_S_BW_O1F2(i, j, k, l, 4);
						UpdateSpatialMessage_TRW_S_BW_O1F2(i, j, k, l, 2);
						UpdateSpatialMessage_TRW_S_BW_O1F2(i, j, k, l, 0);
					}
				}
			}
		}
		for (k = mesh_z-1; k >= 0; k--) {
			for (j = mesh_y-1; j >= 0; j--) {
				for (i = mesh_x-1; i >= 0; i--) {
					UpdateDualMessage_TRW_S_BW(i, j, k);
				}
			}
		}
		for (l = 2; l >= 0; l--) {
			for (k = mesh_z-1; k >= 0; k--) {
				for (j = mesh_y-1; j >= 0; j--) {
					for (i = mesh_x-1; i >= 0; i--) {
						UpdateMessage_TRW_S_BW(i, j, k, l);
					}
				}
			}
		}
	}
	if ((in_scv_w_O1F2 == -2) && (in_scv_w_O2F3 != -2)) {
		// forward pass
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						UpdateMessage_TRW_S_FW_O2F3(i, j, k, l);
					}
				}
			}
		}
		// backward pass
		for (l = 2; l >= 0; l--) {
			for (k = mesh_z-1; k >= 0; k--) {
				for (j = mesh_y-1; j >= 0; j--) {
					for (i = mesh_x-1; i >= 0; i--) {
						UpdateSpatialMessage_TRW_S_BW_O2F3(i, j, k, l, 7);
						UpdateSpatialMessage_TRW_S_BW_O2F3(i, j, k, l, 4);
						UpdateSpatialMessage_TRW_S_BW_O2F3(i, j, k, l, 1);
					}
				}
			}
		}
		for (k = mesh_z-1; k >= 0; k--) {
			for (j = mesh_y-1; j >= 0; j--) {
				for (i = mesh_x-1; i >= 0; i--) {
					UpdateDualMessage_TRW_S_BW(i, j, k);
				}
			}
		}
		for (l = 2; l >= 0; l--) {
			for (k = mesh_z-1; k >= 0; k--) {
				for (j = mesh_y-1; j >= 0; j--) {
					for (i = mesh_x-1; i >= 0; i--) {
						UpdateMessage_TRW_S_BW(i, j, k, l);
					}
				}
			}
		}
	}
	if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 != -2)) {
		// forward pass
		for (l = 0; l < 3; l++) {
			for (k = 0; k < mesh_z; k++) {
				for (j = 0; j < mesh_y; j++) {
					for (i = 0; i < mesh_x; i++) {
						UpdateMessage_TRW_S_FW_O1F2_O2F3(i, j, k, l);
					}
				}
			}
		}
		// backward pass
		for (l = 2; l >= 0; l--) {
			for (k = mesh_z-1; k >= 0; k--) {
				for (j = mesh_y-1; j >= 0; j--) {
					for (i = mesh_x-1; i >= 0; i--) {
						UpdateSpatialMessage_TRW_S_BW_O1F2(i, j, k, l, 4);
						UpdateSpatialMessage_TRW_S_BW_O1F2(i, j, k, l, 2);
						UpdateSpatialMessage_TRW_S_BW_O1F2(i, j, k, l, 0);
					}
				}
			}
		}
		for (l = 2; l >= 0; l--) {
			for (k = mesh_z-1; k >= 0; k--) {
				for (j = mesh_y-1; j >= 0; j--) {
					for (i = mesh_x-1; i >= 0; i--) {
						UpdateSpatialMessage_TRW_S_BW_O2F3(i, j, k, l, 7);
						UpdateSpatialMessage_TRW_S_BW_O2F3(i, j, k, l, 4);
						UpdateSpatialMessage_TRW_S_BW_O2F3(i, j, k, l, 1);
					}
				}
			}
		}
		for (k = mesh_z-1; k >= 0; k--) {
			for (j = mesh_y-1; j >= 0; j--) {
				for (i = mesh_x-1; i >= 0; i--) {
					UpdateDualMessage_TRW_S_BW(i, j, k);
				}
			}
		}
		for (l = 2; l >= 0; l--) {
			for (k = mesh_z-1; k >= 0; k--) {
				for (j = mesh_y-1; j >= 0; j--) {
					for (i = mesh_x-1; i >= 0; i--) {
						UpdateMessage_TRW_S_BW(i, j, k, l);
					}
				}
			}
		}
	}
	//
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
// Compute belief
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::ComputeBelief()
{
	int plane, i, j, k;
	for (plane = 0; plane < 3; plane++) {
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					REALV* belief = Belief[plane].m_pData[k][j][i];

					Add2MessageDual(belief, i, j, k, plane, K);

					// add message from the dual layer
					Add2Message(belief, DualMessage[plane].m_pData[k][j][i], K);
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::FindOptimalSolution()
{
	int plane, i, j, k, l;
	double Min;
	for (plane = 0; plane < 3; plane++) {
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					REALV* belief = Belief[plane].m_pData[k][j][i];
					int index = 0;
					Min = belief[0];
					for (l = 1; l < K; l++) {
						if (Min > belief[l]) {
							Min = belief[l];
							index = l;
						}
					}
					opt_l[plane].m_pData[k][j][i][0] = index;
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
double VolumeBP::GetEnergy()
{
	int plane, i, j, k, y;
	int lx, ly, lz;
	double energy;

	energy = 0;
	if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 == -2)) {
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					for (plane = 0; plane < 3; plane++) {
						energy += RangeTerm[plane].m_pData[k][j][i][opt_l[plane].m_pData[k][j][i][0]];
						//
						if (i < mesh_x-1) {
							//energy += __min((double)fabs((disp_e[opt_l[plane].m_pData[k][j][i  ][0]] + Offset[plane]->m_pData[k][j][i  ][0]) - 
							//							 (disp_e[opt_l[plane].m_pData[k][j][i+1][0]] + Offset[plane]->m_pData[k][j][i+1][0])) * alpha, d);
							y = opt_l[plane].m_pData[k][j][i][0] - opt_l[plane].m_pData[k][j][i+1][0] + (K-1);
							energy += SO1[plane][0].m_pData[k][j][i][y];
						}
						if (j < mesh_y-1) {
							//energy += __min((double)fabs((disp_e[opt_l[plane].m_pData[k][j  ][i][0]] + Offset[plane]->m_pData[k][j  ][i][0]) - 
							//							 (disp_e[opt_l[plane].m_pData[k][j+1][i][0]] + Offset[plane]->m_pData[k][j+1][i][0])) * alpha, d);
							y = opt_l[plane].m_pData[k][j][i][0] - opt_l[plane].m_pData[k][j+1][i][0] + (K-1);
							energy += SO1[plane][1].m_pData[k][j][i][y];
						}
						if (k < mesh_z-1) {
							//energy += __min((double)fabs((disp_e[opt_l[plane].m_pData[k  ][j][i][0]] + Offset[plane]->m_pData[k  ][j][i][0]) - 
							//							 (disp_e[opt_l[plane].m_pData[k+1][j][i][0]] + Offset[plane]->m_pData[k+1][j][i][0])) * alpha, d);
							y = opt_l[plane].m_pData[k][j][i][0] - opt_l[plane].m_pData[k+1][j][i][0] + (K-1);
							energy += SO1[plane][2].m_pData[k][j][i][y];
						}
					}
					//
					lx = opt_l[0].m_pData[k][j][i][0];
					ly = opt_l[1].m_pData[k][j][i][0];
					lz = opt_l[2].m_pData[k][j][i][0];
					energy += dcv->m_pData[k][j][i][lz * K_2 + ly * K + lx];
				}
			}
		}
	} else if ((in_scv_w_O1F2 == -2) && (in_scv_w_O2F3 != -2)) {
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					for (plane = 0; plane < 3; plane++) {
						energy += RangeTerm[plane].m_pData[k][j][i][opt_l[plane].m_pData[k][j][i][0]];
						//
						if (i > 0 && i < mesh_x-1) {
							y = opt_l[plane].m_pData[k][j][i-1][0] -2*opt_l[plane].m_pData[k][j][i][0] + opt_l[plane].m_pData[k][j][i+1][0] + 2*(K-1);
							energy += SO2[plane][0].m_pData[k][j][i][y];
						}
						if (j > 0 && j < mesh_y-1) {
							y = opt_l[plane].m_pData[k][j-1][i][0] -2*opt_l[plane].m_pData[k][j][i][0] + opt_l[plane].m_pData[k][j+1][i][0] + 2*(K-1);
							energy += SO2[plane][1].m_pData[k][j][i][y];
						}
						if (k > 0 && k < mesh_z-1) {
							y = opt_l[plane].m_pData[k-1][j][i][0] -2*opt_l[plane].m_pData[k][j][i][0] + opt_l[plane].m_pData[k+1][j][i][0] + 2*(K-1);
							energy += SO2[plane][2].m_pData[k][j][i][y];
						}
					}
					//
					lx = opt_l[0].m_pData[k][j][i][0];
					ly = opt_l[1].m_pData[k][j][i][0];
					lz = opt_l[2].m_pData[k][j][i][0];
					energy += dcv->m_pData[k][j][i][lz * K_2 + ly * K + lx];
				}
			}
		}
	} else if ((in_scv_w_O1F2 != -2) && (in_scv_w_O2F3 != -2)) {
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					for (plane = 0; plane < 3; plane++) {
						energy += RangeTerm[plane].m_pData[k][j][i][opt_l[plane].m_pData[k][j][i][0]];
						//
						if (i < mesh_x-1) {
							y = opt_l[plane].m_pData[k][j][i][0] - opt_l[plane].m_pData[k][j][i+1][0] + (K-1);
							energy += SO1[plane][0].m_pData[k][j][i][y];
						}
						if (j < mesh_y-1) {
							y = opt_l[plane].m_pData[k][j][i][0] - opt_l[plane].m_pData[k][j+1][i][0] + (K-1);
							energy += SO1[plane][1].m_pData[k][j][i][y];
						}
						if (k < mesh_z-1) {
							y = opt_l[plane].m_pData[k][j][i][0] - opt_l[plane].m_pData[k+1][j][i][0] + (K-1);
							energy += SO1[plane][2].m_pData[k][j][i][y];
						}
						//
						if (i > 0 && i < mesh_x-1) {
							y = opt_l[plane].m_pData[k][j][i-1][0] -2*opt_l[plane].m_pData[k][j][i][0] + opt_l[plane].m_pData[k][j][i+1][0] + 2*(K-1);
							energy += SO2[plane][0].m_pData[k][j][i][y];
						}
						if (j > 0 && j < mesh_y-1) {
							y = opt_l[plane].m_pData[k][j-1][i][0] -2*opt_l[plane].m_pData[k][j][i][0] + opt_l[plane].m_pData[k][j+1][i][0] + 2*(K-1);
							energy += SO2[plane][1].m_pData[k][j][i][y];
						}
						if (k > 0 && k < mesh_z-1) {
							y = opt_l[plane].m_pData[k-1][j][i][0] -2*opt_l[plane].m_pData[k][j][i][0] + opt_l[plane].m_pData[k+1][j][i][0] + 2*(K-1);
							energy += SO2[plane][2].m_pData[k][j][i][y];
						}
					}
					//
					lx = opt_l[0].m_pData[k][j][i][0];
					ly = opt_l[1].m_pData[k][j][i][0];
					lz = opt_l[2].m_pData[k][j][i][0];
					energy += dcv->m_pData[k][j][i][lz * K_2 + ly * K + lx];
				}
			}
		}
	}

	return energy;
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
#ifdef _USE_CUDA
extern "C"
BOOL cu_BP_Check();
extern "C"
void cu_BP_Allocate(int _mesh_x, int _mesh_y, int _mesh_z, int _mesh_ex, int _mesh_ey, int _mesh_ez,
	int _nL, int _K, int _num_d, REALV _alpha_O1, REALV _d_O1, REALV _alpha_O2, REALV _d_O2, REALV _gamma, REALV* _disp_ex, REALV* _disp_ey, REALV* _disp_ez,
	REALV _in_scv_w_O1F2, REALV _in_scv_w_O2F2, REALV _in_scv_w_O2F3);
extern "C"
void cu_BP_Free();
extern "C"
void cu_BP_S(int iter, REALV**** pdcv, REALV**** Offset[3], REALV**** pRangeTerm[3], REALV**** pSO1[3][3], REALV**** pSO2[3][3],
	REALV**** pSO1F2Message[3][6], REALV**** pSO2F3Message[3][9], REALV**** pDualMessage[3], int iterPrev);
extern "C"
void cu_TRW_S(int iter, REALV**** pdcv, REALV**** pOffset[3], REALV**** pRangeTerm[3], REALV**** pSO1[3][3], REALV**** pSO2[3][3],
	REALV**** pSO1F2Message[3][6], REALV**** pSO2F3Message[3][9], REALV**** pDualMessage[3], double* pLowerBound, double* pLowerBoundPrev, int iterPrev);
#endif
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::MessagePassing(int method, int nIterations, int nHierarchy, double* pEnergy, double* pLowerBound)
{
	double energy, lowerBoundPrev;
	int count;
#ifdef _USE_CUDA
	BOOL bCUDA = FALSE;
#endif

	TRACE2("AllocateMessage\n");
	AllocateMessage();

	/*
	if (method == 0 && nHierarchy > 0) {
		TRACE2("=== hierarchy %03d start ===\r\n", nHierarchy);

		VolumeBP* vbp;
		RVolume _dcv, _xx, _yy, _zz;

		///////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////
		vbp = new VolumeBP();

		dcv->ReduceVolume(_dcv, false);
		Offset[0]->ReduceVolume(_xx, true);
		Offset[1]->ReduceVolume(_yy, true);
		Offset[2]->ReduceVolume(_zz, true);

		vbp->init(&_dcv, nL, label_s, lmode, alpha, d, &_xx, &_yy, &_zz, in_scv_w_O1F2, in_scv_w_O2F2, in_scv_w_O2F3);
		vbp->ComputeRangeTerm(gamma / 2);

		vbp->MessagePassing(method, 20, nHierarchy-1);

		vbp->Propagate(*this);

		delete vbp;
		///////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////

		TRACE2("=== hierarchy %03d end ===\r\n", nHierarchy);
	}
	//*/
#ifdef _USE_CUDA
	bCUDA = cu_BP_Check();
#endif

#ifdef _USE_CUDA
	int i, j;
	REALV ****pdcv, ****pOffset[3], ****pRangeTerm[3], ****pSO1[3][3], ****pSO2[3][3];
	REALV ****pSO1F2Message[3][6], ****pSO2F3Message[3][9], ****pDualMessage[3];

	if (bCUDA) {
		pdcv = dcv->m_pData;
		for (i = 0; i < 3; i++) {
			pOffset[i] = Offset[i]->m_pData;
			pRangeTerm[i] = RangeTerm[i].m_pData;
			for (j = 0; j < 3; j++) {
				pSO1[i][j] = SO1[i][j].m_pData;
			}
			for (j = 0; j < 3; j++) {
				pSO2[i][j] = SO2[i][j].m_pData;
			}
			for (j = 0; j < 6; j++) {
				pSO1F2Message[i][j] = SO1F2Message[i][j].m_pData;
			}
			for (j = 0; j < 9; j++) {
				pSO2F3Message[i][j] = SO2F3Message[i][j].m_pData;
			}
			pDualMessage[i] = DualMessage[i].m_pData;
		}

		TRACE2("cu_BP_Allocate\n");
		cu_BP_Allocate(mesh_x, mesh_y, mesh_z, mesh_ex, mesh_ey, mesh_ez, nL, K, num_d, alpha_O1, d_O1, alpha_O2, d_O2, gamma, disp_ex, disp_ey, disp_ez, in_scv_w_O1F2, in_scv_w_O2F2, in_scv_w_O2F3);
	}
#endif

	lowerBoundPrev = 0;
	for (count = 0; count < nIterations; count++) {
		energy = 0;
		lowerBound = 0;

		if (method == 0) {
#ifndef _USE_CUDA
#ifdef USE_TIME_CHECK
			DWORD t = GetTickCount();
#endif
			BP_S(count);
#ifdef USE_TIME_CHECK
			TRACE2("time = %d\r\n", GetTickCount()-t);
#endif
#else
			if (bCUDA) {
				int iter = 20;
				TRACE2("cu_BP_S\n");
				cu_BP_S(iter, pdcv, pOffset, pRangeTerm, pSO1, pSO2, pSO1F2Message, pSO2F3Message, pDualMessage, count);
				count += iter - 1;
			} else {
#ifdef USE_TIME_CHECK
				DWORD t = GetTickCount();
#endif
				BP_S(count);
#ifdef USE_TIME_CHECK
				TRACE2("time = %d\r\n", GetTickCount()-t);
#endif
			}
#endif
		} else if (method == 1) {
#ifndef _USE_CUDA
#ifdef USE_TIME_CHECK
			DWORD t = GetTickCount();
#endif
			TRW_S(count);
#ifdef USE_TIME_CHECK
			TRACE2("time = %d\r\n", GetTickCount()-t);
#endif
#else
			if (bCUDA) {
				int iter = 20;
				TRACE2("cu_TRW_S\n");
				cu_TRW_S(iter, pdcv, pOffset, pRangeTerm, pSO1, pSO2, pSO1F2Message, pSO2F3Message, pDualMessage, &lowerBound, &lowerBoundPrev, count);
				count += iter - 1;
			} else {
#ifdef USE_TIME_CHECK
				DWORD t = GetTickCount();
#endif
				TRW_S(count);
#ifdef USE_TIME_CHECK
				TRACE2("time = %d\r\n", GetTickCount()-t);
#endif
			}
#endif
		}
		
		ComputeBelief();
		FindOptimalSolution();

		energy = GetEnergy();

		TRACE2("iter %03d: lb = %f, lb_delta = %f, e = %f, gap = %f\n", count, lowerBound, lowerBound-lowerBoundPrev, energy, energy-lowerBound);

		if (pEnergy != NULL) {
			pEnergy[count] = energy;
		}
		if (pLowerBound != NULL) {
			pLowerBound[count] = lowerBound;
		}

		lowerBoundPrev = lowerBound;
	}

#ifdef _USE_CUDA
	if (bCUDA) {
		cu_BP_Free();
	}
#endif
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
#include "FFD_table.h"

void VolumeBP::ComputeVelocity(RVolume* vx, RVolume* vy, RVolume* vz, int vd_x, int vd_y, int vd_z)
{
	int i, j, k;
	RVolume tx, ty, tz;

	tx.allocate(mesh_x, mesh_y, mesh_z);
	ty.allocate(mesh_x, mesh_y, mesh_z);
	tz.allocate(mesh_x, mesh_y, mesh_z);

	for (k = 0; k < mesh_z; k++) {
		for (j = 0; j < mesh_y; j++) {
			for (i = 0; i < mesh_x; i++) {
				tx.m_pData[k][j][i][0] = disp_ex[opt_l[0].m_pData[k][j][i][0]] + Offset[0]->m_pData[k][j][i][0];
				ty.m_pData[k][j][i][0] = disp_ey[opt_l[1].m_pData[k][j][i][0]] + Offset[1]->m_pData[k][j][i][0];
				tz.m_pData[k][j][i][0] = disp_ez[opt_l[2].m_pData[k][j][i][0]] + Offset[2]->m_pData[k][j][i][0];
			}
		}
	}

	/*
	tx.imresize(*vx, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
	ty.imresize(*vy, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
	tz.imresize(*vz, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
	/*/
	{
		RVolume wx, wy, wz;

		tx.imresize(wx, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
		ty.imresize(wy, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
		tz.imresize(wz, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);

		vx->allocate(vd_x, vd_y, vd_z);
		vy->allocate(vd_x, vd_y, vd_z);
		vz->allocate(vd_x, vd_y, vd_z);

		for (k = 0; k < vd_z; k++) {
			for (j = 0; j < vd_y; j++) {
				for (i = 0; i < vd_x; i++) {
					vx->m_pData[k][j][i][0] = wx.m_pData[k][j][i][0];
					vy->m_pData[k][j][i][0] = wy.m_pData[k][j][i][0];
					vz->m_pData[k][j][i][0] = wz.m_pData[k][j][i][0];
				}
			}
		}
	}
	//*/
}

void VolumeBP::ComputeVelocityFFD(RVolume* vx, RVolume* vy, RVolume* vz, int vd_x, int vd_y, int vd_z)
{
	int i, j, k, l, m, n;
	float fx, fy, fz;
	float mx, my, mz;
	int ix, iy, iz;
	float c;
	RVolume tx, ty, tz;

	tx.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	ty.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	tz.allocate(mesh_x+3, mesh_y+3, mesh_z+3);

	for (k = 0; k < mesh_z; k++) {
		for (j = 0; j < mesh_y; j++) {
			for (i = 0; i < mesh_x; i++) {
				tx.m_pData[k+1][j+1][i+1][0] = disp_ex[opt_l[0].m_pData[k][j][i][0]] + Offset[0]->m_pData[k][j][i][0];
				ty.m_pData[k+1][j+1][i+1][0] = disp_ey[opt_l[1].m_pData[k][j][i][0]] + Offset[1]->m_pData[k][j][i][0];
				tz.m_pData[k+1][j+1][i+1][0] = disp_ez[opt_l[2].m_pData[k][j][i][0]] + Offset[2]->m_pData[k][j][i][0];
			}
		}
	}

	vx->allocate(vd_x, vd_y, vd_z);
	vy->allocate(vd_x, vd_y, vd_z);
	vz->allocate(vd_x, vd_y, vd_z);
	
	for (k = 0; k < vd_z; k++) {
		for (j = 0; j < vd_y; j++) {
			for (i = 0; i < vd_x; i++) {
				fx = (float)i / mesh_ex;
				fy = (float)j / mesh_ey;
				fz = (float)k / mesh_ez;
				ix = (int)fx;
				iy = (int)fy;
				iz = (int)fz;
				fx = fx - ix;
				fy = fy - iy;
				fz = fz - iz;

				mx = my = mz = 0.0f;
				for (n = 0; n <= 3; n++) {
					for (m = 0; m <= 3; m++) {
						for (l = 0; l <= 3; l++) {
							c = FFD_B(l, fx) * FFD_B(m, fy) * FFD_B(n, fz);
							mx += c * (float)tx.m_pData[iz+n][iy+m][ix+l][0];
							my += c * (float)ty.m_pData[iz+n][iy+m][ix+l][0];
							mz += c * (float)tz.m_pData[iz+n][iy+m][ix+l][0];
						}
					}
				}

				vx->m_pData[k][j][i][0] = mx;
				vy->m_pData[k][j][i][0] = my;
				vz->m_pData[k][j][i][0] = mz;
			}
		}
	}
}

void VolumeBP::ComputeVelocityM(RVolume* vx, RVolume* vy, RVolume* vz)
{
	int i, j, k;

	for (k = 0; k < mesh_z; k++) {
		for (j = 0; j < mesh_y; j++) {
			for (i = 0; i < mesh_x; i++) {
				vx->m_pData[k][j][i][0] = disp_ex[opt_l[0].m_pData[k][j][i][0]] + Offset[0]->m_pData[k][j][i][0];
				vy->m_pData[k][j][i][0] = disp_ey[opt_l[1].m_pData[k][j][i][0]] + Offset[1]->m_pData[k][j][i][0];
				vz->m_pData[k][j][i][0] = disp_ez[opt_l[2].m_pData[k][j][i][0]] + Offset[2]->m_pData[k][j][i][0];
			}
		}
	}
}

void VolumeBP::ComputeVelocitySyD(RVolume* vx_cs, RVolume* vy_cs, RVolume* vz_cs, RVolume* vx_ct, RVolume* vy_ct, RVolume* vz_ct, int vd_x, int vd_y, int vd_z)
{
	int i, j, k;
	RVolume tx_cs, ty_cs, tz_cs;
	RVolume tx_ct, ty_ct, tz_ct;

	tx_cs.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	ty_cs.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	tz_cs.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	tx_ct.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	ty_ct.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	tz_ct.allocate(mesh_x+3, mesh_y+3, mesh_z+3);

	for (k = 0; k < mesh_z; k++) {
		for (j = 0; j < mesh_y; j++) {
			for (i = 0; i < mesh_x; i++) {
				tx_cs.m_pData[k+1][j+1][i+1][0] = -disp_ex[opt_l[0].m_pData[k][j][i][0]] + OffsetCS[0]->m_pData[k][j][i][0];
				ty_cs.m_pData[k+1][j+1][i+1][0] = -disp_ey[opt_l[1].m_pData[k][j][i][0]] + OffsetCS[1]->m_pData[k][j][i][0];
				tz_cs.m_pData[k+1][j+1][i+1][0] = -disp_ez[opt_l[2].m_pData[k][j][i][0]] + OffsetCS[2]->m_pData[k][j][i][0];

				tx_ct.m_pData[k+1][j+1][i+1][0] = disp_ex[opt_l[0].m_pData[k][j][i][0]] + Offset[0]->m_pData[k][j][i][0];
				ty_ct.m_pData[k+1][j+1][i+1][0] = disp_ey[opt_l[1].m_pData[k][j][i][0]] + Offset[1]->m_pData[k][j][i][0];
				tz_ct.m_pData[k+1][j+1][i+1][0] = disp_ez[opt_l[2].m_pData[k][j][i][0]] + Offset[2]->m_pData[k][j][i][0];
			}
		}
	}

	{
		RVolume wx_cs, wy_cs, wz_cs;
		RVolume wx_ct, wy_ct, wz_ct;

		tx_cs.imresize(wx_cs, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
		ty_cs.imresize(wy_cs, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
		tz_cs.imresize(wz_cs, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
		tx_ct.imresize(wx_ct, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
		ty_ct.imresize(wy_ct, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);
		tz_ct.imresize(wz_ct, mesh_x * mesh_ex, mesh_y * mesh_ey, mesh_z * mesh_ez, 1);

		vx_cs->allocate(vd_x, vd_y, vd_z);
		vy_cs->allocate(vd_x, vd_y, vd_z);
		vz_cs->allocate(vd_x, vd_y, vd_z);
		vx_ct->allocate(vd_x, vd_y, vd_z);
		vy_ct->allocate(vd_x, vd_y, vd_z);
		vz_ct->allocate(vd_x, vd_y, vd_z);

		for (k = 0; k < vd_z; k++) {
			for (j = 0; j < vd_y; j++) {
				for (i = 0; i < vd_x; i++) {
					vx_cs->m_pData[k][j][i][0] = wx_cs.m_pData[k][j][i][0];
					vy_cs->m_pData[k][j][i][0] = wy_cs.m_pData[k][j][i][0];
					vz_cs->m_pData[k][j][i][0] = wz_cs.m_pData[k][j][i][0];
					vx_ct->m_pData[k][j][i][0] = wx_ct.m_pData[k][j][i][0];
					vy_ct->m_pData[k][j][i][0] = wy_ct.m_pData[k][j][i][0];
					vz_ct->m_pData[k][j][i][0] = wz_ct.m_pData[k][j][i][0];
				}
			}
		}
	}
}

void VolumeBP::ComputeVelocitySyDFFD(RVolume* vx_cs, RVolume* vy_cs, RVolume* vz_cs, RVolume* vx_ct, RVolume* vy_ct, RVolume* vz_ct, int vd_x, int vd_y, int vd_z)
{
	int i, j, k, l, m, n;
	float fx, fy, fz;
	float mx, my, mz;
	int ix, iy, iz;
	float c;
	RVolume tx_cs, ty_cs, tz_cs;
	RVolume tx_ct, ty_ct, tz_ct;

	tx_cs.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	ty_cs.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	tz_cs.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	tx_ct.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	ty_ct.allocate(mesh_x+3, mesh_y+3, mesh_z+3);
	tz_ct.allocate(mesh_x+3, mesh_y+3, mesh_z+3);

	for (k = 0; k < mesh_z; k++) {
		for (j = 0; j < mesh_y; j++) {
			for (i = 0; i < mesh_x; i++) {
				tx_cs.m_pData[k+1][j+1][i+1][0] = -disp_ex[opt_l[0].m_pData[k][j][i][0]] + OffsetCS[0]->m_pData[k][j][i][0];
				ty_cs.m_pData[k+1][j+1][i+1][0] = -disp_ey[opt_l[1].m_pData[k][j][i][0]] + OffsetCS[1]->m_pData[k][j][i][0];
				tz_cs.m_pData[k+1][j+1][i+1][0] = -disp_ez[opt_l[2].m_pData[k][j][i][0]] + OffsetCS[2]->m_pData[k][j][i][0];

				tx_ct.m_pData[k+1][j+1][i+1][0] = disp_ex[opt_l[0].m_pData[k][j][i][0]] + Offset[0]->m_pData[k][j][i][0];
				ty_ct.m_pData[k+1][j+1][i+1][0] = disp_ey[opt_l[1].m_pData[k][j][i][0]] + Offset[1]->m_pData[k][j][i][0];
				tz_ct.m_pData[k+1][j+1][i+1][0] = disp_ez[opt_l[2].m_pData[k][j][i][0]] + Offset[2]->m_pData[k][j][i][0];
			}
		}
	}

	vx_cs->allocate(vd_x, vd_y, vd_z);
	vy_cs->allocate(vd_x, vd_y, vd_z);
	vz_cs->allocate(vd_x, vd_y, vd_z);
	vx_ct->allocate(vd_x, vd_y, vd_z);
	vy_ct->allocate(vd_x, vd_y, vd_z);
	vz_ct->allocate(vd_x, vd_y, vd_z);
	
	for (k = 0; k < vd_z; k++) {
		for (j = 0; j < vd_y; j++) {
			for (i = 0; i < vd_x; i++) {
				fx = (float)i / mesh_ex;
				fy = (float)j / mesh_ey;
				fz = (float)k / mesh_ez;
				ix = (int)fx;
				iy = (int)fy;
				iz = (int)fz;
				fx = fx - ix;
				fy = fy - iy;
				fz = fz - iz;

				mx = my = mz = 0.0f;
				for (n = 0; n <= 3; n++) {
					for (m = 0; m <= 3; m++) {
						for (l = 0; l <= 3; l++) {
							c = FFD_B(l, fx) * FFD_B(m, fy) * FFD_B(n, fz);
							mx += c * (float)tx_cs.m_pData[iz+n][iy+m][ix+l][0];
							my += c * (float)ty_cs.m_pData[iz+n][iy+m][ix+l][0];
							mz += c * (float)tz_cs.m_pData[iz+n][iy+m][ix+l][0];
						}
					}
				}

				vx_cs->m_pData[k][j][i][0] = mx;
				vy_cs->m_pData[k][j][i][0] = my;
				vz_cs->m_pData[k][j][i][0] = mz;

				mx = my = mz = 0.0f;
				for (n = 0; n <= 3; n++) {
					for (m = 0; m <= 3; m++) {
						for (l = 0; l <= 3; l++) {
							c = FFD_B(l, fx) * FFD_B(m, fy) * FFD_B(n, fz);
							mx += c * (float)tx_ct.m_pData[iz+n][iy+m][ix+l][0];
							my += c * (float)ty_ct.m_pData[iz+n][iy+m][ix+l][0];
							mz += c * (float)tz_ct.m_pData[iz+n][iy+m][ix+l][0];
						}
					}
				}

				vx_ct->m_pData[k][j][i][0] = mx;
				vy_ct->m_pData[k][j][i][0] = my;
				vz_ct->m_pData[k][j][i][0] = mz;
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
void VolumeBP::Propagate(VolumeBP &vbp)
{
	int i, j, k, l, n;
	int x, y, z;
	for (k = 0; k < vbp.mesh_z; k++) {
		z = k >> 1;
		for (j = 0; j < vbp.mesh_y; j++) {
			y = j >> 1;
			for (i = 0; i < vbp.mesh_x; i++) {
				x = i >> 1;
				for (l = 0; l < 3; l++) {
					//*
					memcpy(vbp.DualMessage[l].m_pData[k][j][i], DualMessage[l].m_pData[z][y][x], K * sizeof(REALV));
					if (in_scv_w_O1F2 != -2) {
						for (n = 0; n < 6; n++) {
							memcpy(vbp.SO1F2Message[l][n].m_pData[k][j][i], SO1F2Message[l][n].m_pData[z][y][x], K * sizeof(REALV));
						}
					}
					if (in_scv_w_O2F3 != -2) {
						for (n = 0; n < 9; n++) {
							memcpy(vbp.SO2F3Message[l][n].m_pData[k][j][i], SO2F3Message[l][n].m_pData[z][y][x], K * sizeof(REALV));
						}
					}
					/*/
					for (m = 0; m < K; m++) {
						vbp.DualMessage[l].m_pData[k][j][i][m] = DualMessage[l].m_pData[z][y][x][m] / 8.0f;
					}
					if (in_scv_w_O1F2 != -2) {
						for (n = 0; n < 6; n++) {
							for (m = 0; m < K; m++) {
								vbp.SO1F2Message[l][n].m_pData[k][j][i][m] = SO1F2Message[l][n].m_pData[z][y][x][m] / 8.0f;
							}
						}
					}
					if (in_scv_w_O2F3 != -2) {
						for (n = 0; n < 9; n++) {
							for (m = 0; m < K; m++) {
								vbp.SO2F3Message[l][n].m_pData[k][j][i][m] = SO2F3Message[l][n].m_pData[z][y][x][m] / 8.0f;
							}
						}
					}
					//*/
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
