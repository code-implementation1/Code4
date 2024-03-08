// Copyright 2023 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
//
// This file or its part has been derived from the following repository
// and modified: https://github.com/open-mmlab/mmcv/tree/v1.7.1
// ============================================================================
#ifndef ROI_ALIGN_CUDA_KERNEL_CUH
#define ROI_ALIGN_CUDA_KERNEL_CUH

#include <float.h>
#include <cuda.h>
#include <stdio.h>



#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define THREADS 1024
#define MAX_BLOCKS 65535

template <typename T>
__global__ void fill_zeros_cuda_kernel(const int nthreads, T* tensor) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        tensor[index] = 0.0;
    }
}

template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

/*** Forward ***/
template <typename T>
__global__ void roi_align_forward_cuda_kernel(
    const int nthreads, const T* input, const T* rois, T* output, T* argmax_y,
    T* argmax_x, const int* pooled_height_, const int* pooled_width_,
    const T* spatial_scale_, const int* sampling_ratio_,
    const int* pool_mode_,  // 0 - max pool, 1 - avg pool
    const int* aligned_, const int channels, const int height, const int width) {

    int pooled_height = *pooled_height_;
    int pooled_width = *pooled_width_;
    T spatial_scale = *spatial_scale_;
    int sampling_ratio = *sampling_ratio_;
    int pool_mode = *pool_mode_;
    int aligned = *aligned_;

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {  // for backward-compatibility only
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_height / pooled_height));
    int roi_bin_grid_w =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_width / pooled_width));

    if (pool_mode == 0) {
      // We do max pooling inside a bin
      T maxval = -FLT_MAX;
      T maxidx_y = -1.f, maxidx_x = -1.f;
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);
          T val =
              bilinear_interpolate(offset_input, height, width, y, x, index);
          if (val > maxval) {
            maxval = val;
            maxidx_y = y;
            maxidx_x = x;
          }
        }
      }
      output[index] = maxval;
      argmax_y[index] = maxidx_y;
      argmax_x[index] = maxidx_x;
    } else if (pool_mode == 1) {
      // We do average pooling inside a bin
      const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
      T output_val = 0.;
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);
          T val =
              bilinear_interpolate(offset_input, height, width, y, x, index);
          output_val += val;
        }
      }
      output[index] = output_val / count;
    }
  }
}

extern "C" int CustomROIAlign(
    int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra
) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  if (nparam != 11) {
    return 1;
  }
  void *input = params[0];
  void *rois = params[1];
  void *pooled_height = params[2];
  void *pooled_width = params[3];
  void *spatial_scale = params[4];
  void *sampling_ratio = params[5];
  void *pool_mode = params[6];
  void *aligned = params[7];
  void *output = params[8];
  void *argmax_y = params[9];
  void *argmax_x = params[10];

  int nthreads = 1;
  int channels = shapes[0][1];
  int height = shapes[0][2];
  int width = shapes[0][3];

  for (int i = 0; i < ndims[8]; i++) {
    nthreads *= shapes[8][i];
  }

  // int blocks = min(nthreads / THREADS + 1, MAX_BLOCKS);
  roi_align_forward_cuda_kernel<<<MAX_BLOCKS, THREADS, 0, custream>>>(
    nthreads,
    static_cast<float *>(input),
    static_cast<float *>(rois),
    static_cast<float *>(output),
    static_cast<float *>(argmax_y),
    static_cast<float *>(argmax_x),
    static_cast<int *>(pooled_height),
    static_cast<int *>(pooled_width),
    static_cast<float *>(spatial_scale),
    static_cast<int *>(sampling_ratio),
    static_cast<int *>(pool_mode),  // 0 - max pool, 1 - avg pool
    static_cast<int *>(aligned),
    channels,
    height,
    width);
  return 0;
}


template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = static_cast<int>(y);
  x_low = static_cast<int>(x);

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}


/*** Backward ***/
template <typename T>
__global__ void roi_align_backward_cuda_kernel(
    const int nthreads, const T* grad_output, const T* rois, const T* argmax_y,
    const T* argmax_x, T* grad_input, const int* pooled_height_,
    const int* pooled_width_, const T* spatial_scale_, const int* sampling_ratio_,
    const int* pool_mode_,  // 0 - max pool, 1 - avg pool
    const int* aligned_, const int channels, const int height, const int width) {

    int pooled_height = *pooled_height_;
    int pooled_width = *pooled_width_;
    T spatial_scale = *spatial_scale_;
    int sampling_ratio = *sampling_ratio_;
    int pool_mode = *pool_mode_;
    int aligned = *aligned_;

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T grad_output_this_bin = grad_output[index];
    if (abs(grad_output_this_bin) < 1e-10) {
      continue;
    }

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    T* offset_grad_input =
        grad_input + ((roi_batch_ind * channels + c) * height * width);

    if (pool_mode == 0) {
      T y = argmax_y[index], x = argmax_x[index];
      if (y != -1.f) {
        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, index);

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_grad_input + y_low * width + x_low,
                    grad_output_this_bin * w1);
          atomicAdd(offset_grad_input + y_low * width + x_high,
                    grad_output_this_bin * w2);
          atomicAdd(offset_grad_input + y_high * width + x_low,
                    grad_output_this_bin * w3);
          atomicAdd(offset_grad_input + y_high * width + x_high,
                    grad_output_this_bin * w4);
        }
      }
    } else if (pool_mode == 1) {
      // Do not using rounding; this implementation detail is critical
      T offset = aligned ? (T)0.5 : (T)0.0;
      T roi_start_w = offset_rois[1] * spatial_scale - offset;
      T roi_start_h = offset_rois[2] * spatial_scale - offset;
      T roi_end_w = offset_rois[3] * spatial_scale - offset;
      T roi_end_h = offset_rois[4] * spatial_scale - offset;

      T roi_width = roi_end_w - roi_start_w;
      T roi_height = roi_end_h - roi_start_h;
      if (!aligned) {  // for backward-compatibility only
        roi_width = max(roi_width, (T)1.);
        roi_height = max(roi_height, (T)1.);
      }

      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h =
          (sampling_ratio > 0)
              ? sampling_ratio
              : static_cast<int>(ceilf(roi_height / pooled_height));
      int roi_bin_grid_w =
          (sampling_ratio > 0)
              ? sampling_ratio
              : static_cast<int>(ceilf(roi_width / pooled_width));

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);

          T w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;
          bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                        x_low, x_high, y_low, y_high, index);

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            atomicAdd(offset_grad_input + y_low * width + x_low,
                      grad_output_this_bin * w1 / count);
            atomicAdd(offset_grad_input + y_low * width + x_high,
                      grad_output_this_bin * w2 / count);
            atomicAdd(offset_grad_input + y_high * width + x_low,
                      grad_output_this_bin * w3 / count);
            atomicAdd(offset_grad_input + y_high * width + x_high,
                      grad_output_this_bin * w4 / count);
          }
        }
      }
    }
  }
}

extern "C" int CustomROIAlignBackward(
    int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra
) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  if (nparam != 12) {
    return 1;
  }
  // void *input = params[0];
  void *rois = params[1];
  void *argmax_y = params[2];
  void *argmax_x = params[3];
  void *grad_output = params[4];
  void *pooled_height = params[5];
  void *pooled_width = params[6];
  void *spatial_scale = params[7];
  void *sampling_ratio = params[8];
  void *pool_mode = params[9];  // 0 - max pool, 1 - avg pool
  void *aligned = params[10];
  void *grad_input = params[11];

  int nthreads = 1;
  int zthreads = 1;
  int channels = shapes[0][1];
  int height = shapes[0][2];
  int width = shapes[0][3];

  for (int i = 0; i < ndims[4]; i++) {
    nthreads *= shapes[4][i];
  }
  for (int i = 0; i < ndims[11]; i++) {
    zthreads *= shapes[11][i];
  }

  // int nblocks = min(nthreads / THREADS + 1, MAX_BLOCKS);
  // int zblocks = min(zthreads / THREADS + 1, MAX_BLOCKS);
  fill_zeros_cuda_kernel<<<MAX_BLOCKS, THREADS, 0, custream>>> (
    zthreads,
    static_cast<float *>(grad_input));


  roi_align_backward_cuda_kernel<<<MAX_BLOCKS, THREADS, 0, custream>>>(
    nthreads,
    static_cast<float *>(grad_output),
    static_cast<float *>(rois),
    static_cast<float *>(argmax_y),
    static_cast<float *>(argmax_x),
    static_cast<float *>(grad_input),
    static_cast<int *>(pooled_height),
    static_cast<int *>(pooled_width),
    static_cast<float *>(spatial_scale),
    static_cast<int *>(sampling_ratio),
    static_cast<int *>(pool_mode),  // 0 - max pool, 1 - avg pool
    static_cast<int *>(aligned),
    channels,
    height,
    width);
  return 0;
}

#endif  // ROI_ALIGN_CUDA_KERNEL_CUH
