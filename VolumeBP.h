///////////////////////////////////////////////////////////////////////////////////////
// VolumeBP.h
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

#pragma once


#include "Volume.h"


#define MAX_L1 30
#define MAX_L3 27000


class VolumeBP
{
public:
	int mesh_x, mesh_y, mesh_z;
	int mesh_ex, mesh_ey, mesh_ez;
	int nL;
	REALV label_sx, label_sy, label_sz;
	int lmode;
	REALV gamma;
	REALV alpha_O1, d_O1;
	REALV alpha_O2, d_O2;
	RVolume* dcv;
	RVolume* Offset[3];
	RVolume* OffsetCS[3];
	REALV in_scv_w_O1F2, in_scv_w_O2F2, in_scv_w_O2F3;
	REALV scv_w_O1F2, O1_v_alpha, O1_v_T;
	REALV scv_w_O2F3, O2_v_alpha, O2_v_T;
	//
	int num_d;
	int K, K_2;
	REALV disp_x[MAX_L3];
	REALV disp_y[MAX_L3];
	REALV disp_z[MAX_L3];
	REALV disp_ex[MAX_L1];
	REALV disp_ey[MAX_L1];
	REALV disp_ez[MAX_L1];
	//
	int L, L2, L4, L5, L6, L7, L8, L4_1, L8_1;
	VolumeBase<REALV> SO1[3][3];
	VolumeBase<REALV> SO2[3][3];
	int O1F2k0[MAX_L3];
	int O1F2k1[MAX_L3];
	int O2F3k0[MAX_L3];
	int O2F3k1[MAX_L3];
	int O2F3k2[MAX_L3];
	//
	VolumeBase<REALV> RangeTerm[3];
	VolumeBase<REALV> SO1F2Message[3][6];
	VolumeBase<REALV> SO2F3Message[3][9];
	VolumeBase<REALV> DualMessage[3];
	VolumeBase<REALV> Belief[3];
	VolumeBase<int> opt_l[3];
	//
	double lowerBound;

public:
	VolumeBP(void);
	~VolumeBP(void);

	BOOL init(RVolume* _dcv, int _wsize, REALV _label_sx, REALV _label_sy, REALV _label_sz, int _lmode, REALV _alpha_O1, REALV _d_O1, REALV _alpha_O2, REALV _d_O2,
		RVolume* _xx, RVolume* _yy, RVolume* _zz, int _mesh_ex, int _mesh_ey, int _mesh_ez, REALV _in_scv_w_O1F2, REALV _in_scv_w_O2F2, REALV _in_scv_w_O2F3, RVolume* _sc = NULL);
	BOOL initSyD(RVolume* _dcv, int _wsize, REALV _label_sx, REALV _label_sy, REALV _label_sz, int _lmode, REALV _alpha_O1, REALV _d_O1, REALV _alpha_O2, REALV _d_O2,
		RVolume* _xx_cs, RVolume* _yy_cs, RVolume* _zz_cs, RVolume* _xx_ct, RVolume* _yy_ct, RVolume* _zz_ct, 
		int _mesh_ex, int _mesh_ey, int _mesh_ez, REALV _in_scv_w_O1F2, REALV _in_scv_w_O2F2, REALV _in_scv_w_O2F3);

	void ComputeRangeTerm(REALV _gamma);

	void AllocateMessage();
	//
	void Add2Message(REALV* message, const REALV* other, int nstates);
	void Add2Message(REALV* message, const REALV* other, int nstates, double ctrw);
	REALV FindMin(REALV* message, int nstates);
	void SubtractMin(REALV* message, int nstates, REALV& min);
	void SubtractMin(REALV* message, REALV* message_out, int nstates, REALV& min);
	void Sub2Message(REALV* message, const REALV* other, int nstates);
	void Sub2Message(REALV* message_out, REALV* message, const REALV* other, int nstates);
	REALV ComputeSpatialMessageDT(REALV* message, REALV* message_org, REALV* message_buf, int x, int y, int z, REALV d0, int plane, int nStates, int wsize);
	void Add2MessageDual(REALV* message, int x, int y, int z, int plane, int nStates);
	void Add2MessageSpatial_O1F2(REALV* message, int x, int y, int z, int plane, int direction, int nStates);
	void Add2MessageSpatial_O2F3(REALV* message, int x, int y, int z, int plane, int direction, int nStates);
	// Update Message for BP
	void UpdateSpatialMessage(int x, int y, int z, int plane, int direction);
	void UpdateSpatialMessage_O2F3(int x1, int y1, int z1, int plane, int dir1);
	void UpdateDualMessage(int x, int y, int z, int plane);
	// Update Message for TRW_S
	void UpdateMessage_TRW_S_FW_O1F2(int x, int y, int z, int plane);
	void UpdateMessage_TRW_S_FW_O2F3(int x, int y, int z, int plane);
	void UpdateMessage_TRW_S_FW_O1F2_O2F3(int x, int y, int z, int plane);
	void UpdateMessage_TRW_S_BW(int x, int y, int z, int plane);
	void UpdateSpatialMessage_TRW_S_BW_O1F2(int x, int y, int z, int plane, int direction);
	void UpdateSpatialMessage_TRW_S_BW_O2F3(int x1, int y1, int z1, int plane, int dir1);
	void UpdateDualMessage_TRW_S_BW(int x, int y, int z);
	//
	void BP_S(int count);
	void TRW_S(int count);
	//
	void ComputeBelief();
	void FindOptimalSolution();
	double GetEnergy();
	//
	void MessagePassing(int method, int nIterations, int nHierarchy, double* pEnergy = NULL, double* pLowerBound = NULL);
	//
	void ComputeVelocity(RVolume* vx, RVolume* vy, RVolume* vz, int vd_x, int vd_y, int vd_z);
	void ComputeVelocityFFD(RVolume* vx, RVolume* vy, RVolume* vz, int vd_x, int vd_y, int vd_z);
	void ComputeVelocityM(RVolume* vx, RVolume* vy, RVolume* vz);
	//
	void ComputeVelocitySyD(RVolume* vx_cs, RVolume* vy_cs, RVolume* vz_cs, RVolume* vx_ct, RVolume* vy_ct, RVolume* vz_ct, int vd_x, int vd_y, int vd_z);
	void ComputeVelocitySyDFFD(RVolume* vx_cs, RVolume* vy_cs, RVolume* vz_cs, RVolume* vx_ct, RVolume* vy_ct, RVolume* vz_ct, int vd_x, int vd_y, int vd_z);
	//
	void Propagate(VolumeBP &vbp);
};
