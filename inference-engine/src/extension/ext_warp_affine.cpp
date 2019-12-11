// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <string>
#include <vector>
#include <limits>
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class WarpAffineImpl: public ExtLayerBase {
public:
    explicit WarpAffineImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->getTensorDesc().getDims().size() != 4)
                THROW_IE_EXCEPTION << "WarpAffine input1 supports only 4d blobs!";

            if (layer->insData[1].lock()->getTensorDesc().getDims().size() != 2)
                THROW_IE_EXCEPTION << "WarpAffine input2 supports only 2d blobs!";

            auto src_precision = layer->insData[0].lock()->getTensorDesc().getPrecision();
            if (src_precision != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input1 data tensor precision. Only U8 or FP32 are supported!";
            auto src_precision1 = layer->insData[1].lock()->getTensorDesc().getPrecision();
            if (src_precision1 != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input2 data tensor precision. Only FP32 are supported!";

            if (layer->outData[0]->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect output data tensor precision. Only FP32 is supported!";

            ConfLayout blk_layout;
#if defined(HAVE_AVX512F)
            blk_layout = ConfLayout::BLK16;
#else
            blk_layout = ConfLayout::BLK8;
#endif
            addConfig(layer, { DataConfigurator(blk_layout), DataConfigurator(ConfLayout::ANY) }, { DataConfigurator(blk_layout) });

        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        size_t IN = inputs[0]->getTensorDesc().getDims()[0];
        size_t IH = inputs[0]->getTensorDesc().getDims()[2];
        size_t IW = inputs[0]->getTensorDesc().getDims()[3];
        size_t OH = outputs[0]->getTensorDesc().getDims()[2];
        size_t OW = outputs[0]->getTensorDesc().getDims()[3];

        auto *dst_data = outputs[0]->buffer().as<float *>();

        switch (inputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32:
        {
            size_t IC = inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[1] *
                        inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[4];
            interpolate(IN, IC, inputs[0]->buffer().as<const float *>(), IH, IW, dst_data, OH, OW, inputs[1]->buffer().as<const float *>());
        }
        break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect input precision. Only FP32 is supported!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }

        return OK;
    }

private:
    void interpolate(const size_t N, const size_t C,
                     const float *src, const size_t IH, const size_t IW,
                     float *dst, const size_t OH, const size_t OW,
                     const float *matrices) {
#if defined(HAVE_AVX512F)
        const int block_size = 16;
#else
        const int block_size = 8;
#endif

        // Align channel number to block size to deal with channels padding in IE with multiple blobs
        size_t CB = (C + block_size - 1) & (-block_size);
        size_t CH = (C + block_size - 1) / block_size;

        parallel_for3d(N, CH, OH, [&](size_t n, size_t cb, size_t h) {
            const float* matrix = matrices + 6 * n;
            const float *psrc = src + n * CB * IH * IW;

            for (size_t w = 0; w < OW; ++w) {
                float xi = std::min(std::max(w*matrix[0] + h*matrix[1] + matrix[2], 0.001f), IW-1.001f);
                float yi = std::min(std::max(w*matrix[3] + h*matrix[4] + matrix[5], 0.001f), IH-1.001f);

                int ih0 = (int)(yi);
                int ih1 = ih0 + 1;
                float h_lambda0 = yi - ih0;
                float h_lambda1 = 1.0f - h_lambda0;

                int iw0 = (int)(xi);
                int iw1 = iw0 + 1;
                float w_lambda0 = xi - iw0;
                float w_lambda1 = 1.0f - w_lambda0;

                const float *psrc00 =
                        psrc + cb * block_size * IW * IH + ih0 * IW * block_size + iw0 * block_size;
                const float *psrc01 =
                        psrc + cb * block_size * IW * IH + ih0 * IW * block_size + iw1 * block_size;
                const float *psrc10 =
                        psrc + cb * block_size * IW * IH + ih1 * IW * block_size + iw0 * block_size;
                const float *psrc11 =
                        psrc + cb * block_size * IW * IH + ih1 * IW * block_size + iw1 * block_size;

                float *pdst = dst + n * CB * OH * OW + cb * block_size * OW * OH + h * OW * block_size +
                              w * block_size;

#if defined(HAVE_AVX512F)
                __m512 vwl0 = _mm512_set1_ps(w_lambda0);
                        __m512 vwl1 = _mm512_set1_ps(w_lambda1);
                        __m512 vhl0 = _mm512_set1_ps(h_lambda0);
                        __m512 vhl1 = _mm512_set1_ps(h_lambda1);
                        __m512 vsrc00 = _mm512_loadu_ps(psrc00);
                        __m512 vsrc01 = _mm512_loadu_ps(psrc01);
                        __m512 vsrc10 = _mm512_loadu_ps(psrc10);
                        __m512 vsrc11 = _mm512_loadu_ps(psrc11);

                        __m512 vdst0 = _mm512_fmadd_ps(vwl1, vsrc00, _mm512_mul_ps(vwl0, vsrc01));
                        __m512 vdst1 = _mm512_fmadd_ps(vwl1, vsrc10, _mm512_mul_ps(vwl0, vsrc11));
                        __m512 vdst  = _mm512_fmadd_ps(vhl1, vdst0, _mm512_mul_ps(vhl0, vdst1));

                        _mm512_storeu_ps(pdst, vdst);
#elif defined(HAVE_AVX2)
                __m256 vwl0 = _mm256_set1_ps(w_lambda0);
                        __m256 vwl1 = _mm256_set1_ps(w_lambda1);
                        __m256 vhl0 = _mm256_set1_ps(h_lambda0);
                        __m256 vhl1 = _mm256_set1_ps(h_lambda1);
                        __m256 vsrc00 = _mm256_loadu_ps(psrc00);
                        __m256 vsrc01 = _mm256_loadu_ps(psrc01);
                        __m256 vsrc10 = _mm256_loadu_ps(psrc10);
                        __m256 vsrc11 = _mm256_loadu_ps(psrc11);

                       __m256 vdst0 = _mm256_fmadd_ps(vwl1, vsrc00, _mm256_mul_ps(vwl0, vsrc01));
                       __m256 vdst1 = _mm256_fmadd_ps(vwl1, vsrc10, _mm256_mul_ps(vwl0, vsrc11));
                       __m256 vdst  = _mm256_fmadd_ps(vhl1, vdst0, _mm256_mul_ps(vhl0, vdst1));

                       _mm256_storeu_ps(pdst, vdst);
#elif defined(HAVE_SSE)
                __m128 vwl0 = _mm_set1_ps(w_lambda0);
                        __m128 vwl1 = _mm_set1_ps(w_lambda1);
                        __m128 vhl0 = _mm_set1_ps(h_lambda0);
                        __m128 vhl1 = _mm_set1_ps(h_lambda1);
                        for (int i = 0; i < block_size/4; i++) {
                            __m128 vsrc00 = _mm_loadu_ps(psrc00 + i*block_size/2);
                            __m128 vsrc01 = _mm_loadu_ps(psrc01 + i*block_size/2);
                            __m128 vsrc10 = _mm_loadu_ps(psrc10 + i*block_size/2);
                            __m128 vsrc11 = _mm_loadu_ps(psrc11 + i*block_size/2);

                           __m128 vdst00 = _mm_mul_ps(vwl1, vsrc00);
                           __m128 vdst01 = _mm_mul_ps(vwl0, vsrc01);
                           __m128 vdst10 = _mm_mul_ps(vwl1, vsrc10);
                           __m128 vdst11 = _mm_mul_ps(vwl0, vsrc11);

                           __m128 vdst0 = _mm_add_ps(vdst00, vdst01);
                           __m128 vdst1 = _mm_add_ps(vdst10, vdst11);

                            __m128 vdst = _mm_add_ps(_mm_mul_ps(vhl1, vdst0), _mm_mul_ps(vhl0, vdst1));

                           _mm_storeu_ps(pdst + i*block_size/2, vdst);
                        }
#else
                for (int c = 0; c < block_size; ++c) {
                    pdst[c] = h_lambda1 * (w_lambda1 * psrc00[c] + w_lambda0 * psrc01[c]) +
                              h_lambda0 * (w_lambda1 * psrc10[c] + w_lambda0 * psrc11[c]);
                }
#endif
            }
        });
    }
};

REG_FACTORY_FOR(ImplFactory<WarpAffineImpl>, WarpAffine);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
