
#include <cuda_runtime.h>

extern "C" {

__device__ int clamp(int x, int low, int high)
{
  return x < low ? low : (x > high ? high : x);
}

__global__ void ResizePadNormPadvalueKernel(const unsigned char *src,
                                            int                  src_h,
                                            int                  src_w,
                                            int                  src_stride,
                                            int                  src_format_bgr,
                                            float               *dst,
                                            int                  dst_h,
                                            int                  dst_w,
                                            int                  pad_top,
                                            int                  pad_left,
                                            float                scale,
                                            float                mean0,
                                            float                mean1,
                                            float                mean2,
                                            float                val0,
                                            float                val1,
                                            float                val2,
                                            bool                 do_transpose,
                                            bool                 do_norm,
                                            int                  pad_value,
                                            float                pad_color0,
                                            float                pad_color1,
                                            float                pad_color2)
{
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst_w || dy >= dst_h)
    return;

  float out_c[3] = {0};

  int  x        = dx - pad_left;
  int  y        = dy - pad_top;
  bool in_range = (x >= 0 && x < int(src_w * scale) && y >= 0 && y < int(src_h * scale));

  if (in_range)
  {
    float src_x                = x / scale;
    float src_y                = y / scale;
    int   isrc_x               = static_cast<int>(roundf(src_x));
    int   isrc_y               = static_cast<int>(roundf(src_y));
    isrc_x                     = clamp(isrc_x, 0, src_w - 1);
    isrc_y                     = clamp(isrc_y, 0, src_h - 1);
    const unsigned char *p     = src + isrc_y * src_stride + isrc_x * 3;
    int                  r_idx = src_format_bgr ? 2 : 0;
    int                  g_idx = 1;
    int                  b_idx = src_format_bgr ? 0 : 2;
    out_c[0]                   = static_cast<float>(p[r_idx]);
    out_c[1]                   = static_cast<float>(p[g_idx]);
    out_c[2]                   = static_cast<float>(p[b_idx]);
  } else
  {
    // -------- pad区，按pad value 逻辑处理 -------------------
    if (pad_value == 0)
    { // EDGE
      int                  src_x = clamp((int)roundf((x / scale)), 0, src_w - 1);
      int                  src_y = clamp((int)roundf((y / scale)), 0, src_h - 1);
      const unsigned char *p     = src + src_y * src_stride + src_x * 3;
      int                  r_idx = src_format_bgr ? 2 : 0;
      int                  g_idx = 1;
      int                  b_idx = src_format_bgr ? 0 : 2;
      out_c[0]                   = static_cast<float>(p[r_idx]);
      out_c[1]                   = static_cast<float>(p[g_idx]);
      out_c[2]                   = static_cast<float>(p[b_idx]);
    } else
    { // CONSTANT
      out_c[0] = pad_color0;
      out_c[1] = pad_color1;
      out_c[2] = pad_color2;
    }
  }

  if (do_norm)
  {
    out_c[0] = (out_c[0] - mean0) / val0;
    out_c[1] = (out_c[1] - mean1) / val1;
    out_c[2] = (out_c[2] - mean2) / val2;
  }

  int single_channel = dst_h * dst_w;
  if (do_transpose)
  {
    dst[0 * single_channel + dy * dst_w + dx] = out_c[0];
    dst[1 * single_channel + dy * dst_w + dx] = out_c[1];
    dst[2 * single_channel + dy * dst_w + dx] = out_c[2];
  } else
  {
    int idx      = (dy * dst_w + dx) * 3;
    dst[idx + 0] = out_c[0];
    dst[idx + 1] = out_c[1];
    dst[idx + 2] = out_c[2];
  }
}

void launch_resize_pad_norm(const unsigned char *src,
                            int                  src_h,
                            int                  src_w,
                            int                  src_stride,
                            int                  src_format_bgr,
                            float               *dst,
                            int                  dst_h,
                            int                  dst_w,
                            int                  pad_top,
                            int                  pad_left,
                            float                scale,
                            float                mean0,
                            float                mean1,
                            float                mean2,
                            float                val0,
                            float                val1,
                            float                val2,
                            bool                 do_transpose,
                            bool                 do_norm,
                            int                  pad_value,
                            float                pad_color0,
                            float                pad_color1,
                            float                pad_color2,
                            cudaStream_t         stream)
{
  dim3 block(16, 16);
  dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
  ResizePadNormPadvalueKernel<<<grid, block, 0, stream>>>(
      src, src_h, src_w, src_stride, src_format_bgr, dst, dst_h, dst_w, pad_top, pad_left, scale,
      mean0, mean1, mean2, val0, val1, val2, do_transpose, do_norm, pad_value, pad_color0,
      pad_color1, pad_color2);
}
}
