// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define VEC_SIZE 16

#define _CAT(a,b) a##b
#define CAT(a,b) _CAT(a,b)

inline void warpAffine(const int N, const int C,
                        const __global INPUT0_TYPE* src, const int IH, const int IW,
                        __global OUTPUT0_TYPE* dst, const int OH, const int OW,
                        const __global INPUT1_TYPE* matrix)
{
    if (IH < 2)
        return;

    int h = get_global_id(0);
    int w = get_global_id(1);

    if (w >= OW)
        return;

// This part is for warping normalized (positions in interval [-1,1]), which is same as BilinearInterpolation layer
//    INPUT1_TYPE m0 = IW/(OW-1.f)*matrix[0];
//    INPUT1_TYPE m1 = IW/(OH-1.f)*matrix[1];
//    INPUT1_TYPE m2 = 0.5f*IW*(-matrix[0]-matrix[1]+matrix[2]+1.f);
//    INPUT1_TYPE m3 = IH/(OW-1.f)*matrix[3];
//    INPUT1_TYPE m4 = IH/(OH-1.f)*matrix[4];
//    INPUT1_TYPE m5 = 0.5f*IH*(-matrix[3]-matrix[4]+matrix[5]+1.f);
//
//    INPUT0_TYPE xi = w*m0 + h*m1 + m2;
//    INPUT0_TYPE yi = w*m3 + h*m4 + m5;

// This part is for unnormalized (positions in pixel) input
    INPUT0_TYPE xi = w*matrix[0] + h*matrix[1] + matrix[2];
    INPUT0_TYPE yi = w*matrix[3] + h*matrix[4] + matrix[5];


//    INPUT0_TYPE xo_n = (2*w) / (INPUT0_TYPE)(OW-1) - (INPUT0_TYPE)(1.f);
//    INPUT0_TYPE yo_n = (2*h) / (INPUT0_TYPE)(OH-1) - (INPUT0_TYPE)(1.f);
//    INPUT0_TYPE xi_n = xo_n*(INPUT0_TYPE)(matrix[0]) + yo_n*(INPUT0_TYPE)(matrix[1]) + (INPUT0_TYPE)(matrix[2]);
//    INPUT0_TYPE yi_n = xo_n*(INPUT0_TYPE)(matrix[3]) + yo_n*(INPUT0_TYPE)(matrix[4]) + (INPUT0_TYPE)(matrix[5]);
//    INPUT0_TYPE xi = (xi_n + (INPUT0_TYPE)(1.f)) * (INPUT0_TYPE)(0.5f) * (INPUT0_TYPE)(IW);
//    INPUT0_TYPE yi = (yi_n + (INPUT0_TYPE)(1.f)) * (INPUT0_TYPE)(0.5f) * (INPUT0_TYPE)(IH);

    if (xi < (INPUT0_TYPE)(-0.5f) || yi < (INPUT0_TYPE)(-0.5f) || xi > IW-(INPUT0_TYPE)(0.5f) || yi > IH-(INPUT0_TYPE)(-0.5f)){
        __global OUTPUT0_TYPE* pdst = dst + (h)*OUTPUT0_PITCHES[2] + (w)*OUTPUT0_PITCHES[3];
#if defined(INPUT0_FORMAT_YXFB) && defined(OUTPUT0_FORMAT_YXFB)
        typedef CAT(INPUT0_TYPE, VEC_SIZE) vec16_t;
        __global vec16_t* pvdst = (__global vec16_t*)pdst;
#endif

        for (int n = 0; n < N; n++)
        {
            int c = 0;
    #if defined(INPUT0_FORMAT_YXFB) && defined(OUTPUT0_FORMAT_YXFB)
            __attribute__((opencl_unroll_hint))
            for (int vc = 0; c <= C - VEC_SIZE; c += VEC_SIZE, vc++)
            {
                int in_idx = (n*INPUT0_PITCHES[0] + vc*INPUT0_PITCHES[1]);
                int out_idx = (n*OUTPUT0_PITCHES[0] + vc*OUTPUT0_PITCHES[1]);
                pvdst[out_idx] = (vec16_t)(0.f);
            }
    #endif
            __attribute__((opencl_unroll_hint))
            for (; c < C; c++)
            {
                int in_idx = n*INPUT0_PITCHES[0] + c*INPUT0_PITCHES[1];
                int out_idx = n*OUTPUT0_PITCHES[0] + c*OUTPUT0_PITCHES[1];
                pdst[out_idx] = (OUTPUT0_TYPE)(0.f);
            }
        }
    }
    else {
        int ih0 = (int)(yi);
        int ih1 = (ih0 < IH - 1) ? ih0+1 : ih0;
        INPUT0_TYPE h_lambda0 = yi - ih0;
        INPUT0_TYPE h_lambda1 = (INPUT0_TYPE)(1.0f) - h_lambda0;

        int iw0 = (int)(xi);
        int iw1 = (iw0 < IW - 1) ? iw0 + 1 : iw0;
        INPUT0_TYPE w_lambda0 = xi - iw0;
        INPUT0_TYPE w_lambda1 = (INPUT0_TYPE)(1.0f) - w_lambda0;

        const __global INPUT0_TYPE* psrc00 = src + (ih0)*INPUT0_PITCHES[2] + (iw0)*INPUT0_PITCHES[3];
        const __global INPUT0_TYPE* psrc01 = src + (ih0)*INPUT0_PITCHES[2] + (iw1)*INPUT0_PITCHES[3];
        const __global INPUT0_TYPE* psrc10 = src + (ih1)*INPUT0_PITCHES[2] + (iw0)*INPUT0_PITCHES[3];
        const __global INPUT0_TYPE* psrc11 = src + (ih1)*INPUT0_PITCHES[2] + (iw1)*INPUT0_PITCHES[3];

        __global OUTPUT0_TYPE* pdst = dst + (h)*OUTPUT0_PITCHES[2] + (w)*OUTPUT0_PITCHES[3];

    #if defined(INPUT0_FORMAT_YXFB) && defined(OUTPUT0_FORMAT_YXFB)
        typedef CAT(INPUT0_TYPE, VEC_SIZE) vec16_t;

        const __global vec16_t* pvsrc00 = (const __global vec16_t*)psrc00;
        const __global vec16_t* pvsrc01 = (const __global vec16_t*)psrc01;
        const __global vec16_t* pvsrc10 = (const __global vec16_t*)psrc10;
        const __global vec16_t* pvsrc11 = (const __global vec16_t*)psrc11;

        __global vec16_t* pvdst = (__global vec16_t*)pdst;
    #endif

        for (int n = 0; n < N; n++)
        {
            int c = 0;
    #if defined(INPUT0_FORMAT_YXFB) && defined(OUTPUT0_FORMAT_YXFB)
            __attribute__((opencl_unroll_hint))
            for (int vc = 0; c <= C - VEC_SIZE; c += VEC_SIZE, vc++)
            {
                int in_idx = (n*INPUT0_PITCHES[0] + vc*INPUT0_PITCHES[1]);
                int out_idx = (n*OUTPUT0_PITCHES[0] + vc*OUTPUT0_PITCHES[1]);
                pvdst[out_idx] = (vec16_t)(h_lambda1 * (w_lambda1 * pvsrc00[in_idx] +
                                                        w_lambda0 * pvsrc01[in_idx]) +
                                           h_lambda0 * (w_lambda1 * pvsrc10[in_idx] +
                                                        w_lambda0 * pvsrc11[in_idx]));
            }
    #endif
            __attribute__((opencl_unroll_hint))
            for (; c < C; c++)
            {
                int in_idx = n*INPUT0_PITCHES[0] + c*INPUT0_PITCHES[1];
                int out_idx = n*OUTPUT0_PITCHES[0] + c*OUTPUT0_PITCHES[1];
                pdst[out_idx] = (OUTPUT0_TYPE)(h_lambda1 * (w_lambda1 * psrc00[in_idx] + w_lambda0 * psrc01[in_idx]) +
                                               h_lambda0 * (w_lambda1 * psrc10[in_idx] + w_lambda0 * psrc11[in_idx]));
            }
        }
    }
}

__kernel void warp_affine(const __global INPUT0_TYPE*  input,
                          const __global INPUT1_TYPE*  matrix,
                           __global OUTPUT0_TYPE* output)
{
    int IB = INPUT0_DIMS[0];
    int IF = INPUT0_DIMS[1];
    int IY = INPUT0_DIMS[2];
    int IX = INPUT0_DIMS[3];

    int OY = OUTPUT0_DIMS[2];
    int OX = OUTPUT0_DIMS[3];

    warpAffine(IB, IF, input, IY, IX, output, OY, OX, matrix);
}
