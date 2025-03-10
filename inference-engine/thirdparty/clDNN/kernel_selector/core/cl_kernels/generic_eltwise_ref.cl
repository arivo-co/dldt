/*
// Copyright (c) 2016 Intel Corporation
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
*/

#include "include/include_all.cl"

#if !ELTWISE_BROADCAST
#ifdef INPUT_STRIDED

#define GET_INDEX(prefix, num) \
    CAT(CAT(prefix, num), _OFFSET) + \
    ((d1 * CAT(CAT(prefix, num), _STRIDE_X)) % CAT(CAT(prefix, num), _SIZE_X))*CAT(CAT(prefix, num), _X_PITCH) +\
    ((d2 * CAT(CAT(prefix, num), _STRIDE_Y)) % CAT(CAT(prefix, num), _SIZE_Y))*CAT(CAT(prefix, num), _Y_PITCH) +\
    (d3 % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) + \
    (d4 % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#else

#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM

#define GET_INDEX(prefix, num)                                                          \
    CAT(CAT(prefix, num), _OFFSET) +                                                    \
    (d1 % CAT(CAT(prefix, num), _SIZE_X     ))*CAT(CAT(prefix, num), _X_PITCH) +        \
    (d2 % CAT(CAT(prefix, num), _SIZE_Y     ))*CAT(CAT(prefix, num), _Y_PITCH) +        \
    (d3 % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) +  \
    (d4 % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#elif ELTWISE_NO_PITCH_SAME_DIMS
#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) + d1

#else

#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) +                                                \
    (d1 % CAT(CAT(prefix, num), _SIZES)[0])*CAT(CAT(prefix, num), _PITCHES)[0] +    \
    (d2 % CAT(CAT(prefix, num), _SIZES)[1])*CAT(CAT(prefix, num), _PITCHES)[1] +    \
    (d3 % CAT(CAT(prefix, num), _SIZES)[2])*CAT(CAT(prefix, num), _PITCHES)[2] +    \
    (d4 % CAT(CAT(prefix, num), _SIZES)[3])*CAT(CAT(prefix, num), _PITCHES)[3]

#endif

#endif
#else
#ifdef INPUT_STRIDED

#define GET_INDEX(prefix, num) \
    CAT(CAT(prefix, num), _OFFSET) + \
    ((CAT(d1_in, num) * CAT(CAT(prefix, num), _STRIDE_X)) % CAT(CAT(prefix, num), _SIZE_X))*CAT(CAT(prefix, num), _X_PITCH) +\
    ((CAT(d2_in, num) * CAT(CAT(prefix, num), _STRIDE_Y)) % CAT(CAT(prefix, num), _SIZE_Y))*CAT(CAT(prefix, num), _Y_PITCH) +\
    (CAT(d3_in, num) % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) + \
    (CAT(d4_in, num) % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#else

#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM

#define GET_INDEX(prefix, num)                                                          \
    CAT(CAT(prefix, num), _OFFSET) +                                                    \
    (CAT(d1_in, num) % CAT(CAT(prefix, num), _SIZE_X     ))*CAT(CAT(prefix, num), _X_PITCH) +        \
    (CAT(d2_in, num) % CAT(CAT(prefix, num), _SIZE_Y     ))*CAT(CAT(prefix, num), _Y_PITCH) +        \
    (CAT(d3_in, num) % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) +  \
    (CAT(d4_in, num) % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#elif ELTWISE_NO_PITCH_SAME_DIMS
#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) + CAT(d1_in, num)

#else

#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) +                                                \
    (CAT(d1_in, num) % CAT(CAT(prefix, num), _SIZES)[0])*CAT(CAT(prefix, num), _PITCHES)[0] +    \
    (CAT(d2_in, num) % CAT(CAT(prefix, num), _SIZES)[1])*CAT(CAT(prefix, num), _PITCHES)[1] +    \
    (CAT(d3_in, num) % CAT(CAT(prefix, num), _SIZES)[2])*CAT(CAT(prefix, num), _PITCHES)[2] +    \
    (CAT(d4_in, num) % CAT(CAT(prefix, num), _SIZES)[3])*CAT(CAT(prefix, num), _PITCHES)[3]

#endif

#endif
#endif

KERNEL(eltwise)(
    INPUTS_DECLS
    __global UNIT_TYPE* output
#if CALIBRATION_TERM
    , const __global float* calibrations
#endif
    )
{
#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM
    const uint d1 = get_global_id(GWS_YX) % OUTPUT_SIZE_X;  // X
    const uint d2 = get_global_id(GWS_YX) / OUTPUT_SIZE_X;  // Y
    const uint d3 = get_global_id(GWS_FEATURE);             // Feature
    const uint d4 = get_global_id(GWS_BATCH);               // Batch

    uint output_offset = OUTPUT_OFFSET +
                         d1*OUTPUT_X_PITCH +
                         d2*OUTPUT_Y_PITCH +
                         d3*OUTPUT_FEATURE_PITCH +
                         d4*OUTPUT_BATCH_PITCH;
#elif ELTWISE_NO_PITCH_SAME_DIMS
    const uint d1 = get_global_id(0);
    uint output_offset = OUTPUT_OFFSET + d1;
#else
    const uint d1 = get_global_id(0);
    const uint d2 = get_global_id(1);
    const uint d3 = get_global_id(2) % OUTPUT_SIZES[2];
    const uint d4 = get_global_id(2) / OUTPUT_SIZES[2];
    
    uint output_offset = OUTPUT_OFFSET +
                         d1*OUTPUT_PITCHES[0] +
                         d2*OUTPUT_PITCHES[1] +
                         d3*OUTPUT_PITCHES[2] +
                         d4*OUTPUT_PITCHES[3];
#endif

#if ELTWISE_BROADCAST
    const uint d1_in0 = d1 % INPUT0_SIZE_X;
#if !ELTWISE_NO_PITCH_SAME_DIMS
    const uint d2_in0 = d2 % INPUT0_SIZE_Y;
    const uint d3_in0 = d3 % INPUT0_FEATURE_NUM;
    const uint d4_in0 = d4 % INPUT0_BATCH_NUM;
#endif
    const uint d1_in1 = d1 % INPUT1_SIZE_X;
#if !ELTWISE_NO_PITCH_SAME_DIMS
    const uint d2_in1 = d2 % INPUT1_SIZE_Y;
    const uint d3_in1 = d3 % INPUT1_FEATURE_NUM;
    const uint d4_in1 = d4 % INPUT1_BATCH_NUM;
#endif
#endif

#if QUANTIZATION_TERM
    int res;
#else
    UNIT_TYPE res;
#endif
    
    DO_ELTWISE;

#if QUANTIZATION_TERM
#if CALIBRATION_TERM
    res = (int)round(((float)res) * calibrations[d3]);
#else  // CALIBRATION_TERM
    res = (int)round(((float)res) * O_QF);
#endif // CALIBRATION_TERM
#endif // QUANTIZATION_TERM

#if QUANTIZATION_TERM
    output[output_offset] = ACTIVATION(convert_char(res), NL_M, NL_N);
#else
    output[output_offset] = ACTIVATION(res, NL_M, NL_N);
#endif
}
