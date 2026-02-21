// vec4_math.cuh
#pragma once
#include <cuda_runtime.h>

struct Mat4x4f { float4 r0, r1, r2, r3; };


__device__ __forceinline__ Mat4x4f make_mat4x4f(float4 r0, float4 r1, float4 r2, float4 r3) {
  Mat4x4f M; M.r0=r0; M.r1=r1; M.r2=r2; M.r3=r3; return M;
}

__device__ __forceinline__ float4 fma4(float4 v, float s, float4 acc) {
  acc.x = fmaf(v.x, s, acc.x);
  acc.y = fmaf(v.y, s, acc.y);
  acc.z = fmaf(v.z, s, acc.z);
  acc.w = fmaf(v.w, s, acc.w);
  return acc;
}

__device__ __forceinline__ float4 mul4(float4 v, float s) {
  return make_float4(v.x*s, v.y*s, v.z*s, v.w*s);
}

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
  return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__device__ __forceinline__ float4 vec_mul(const Mat4x4f& M, float4 v) {
  float4 out;
  out.x = M.r0.x*v.x + M.r0.y*v.y + M.r0.z*v.z + M.r0.w*v.w;
  out.y = M.r1.x*v.x + M.r1.y*v.y + M.r1.z*v.z + M.r1.w*v.w;
  out.z = M.r2.x*v.x + M.r2.y*v.y + M.r2.z*v.z + M.r2.w*v.w;
  out.w = M.r3.x*v.x + M.r3.y*v.y + M.r3.z*v.z + M.r3.w*v.w;
  return out;
}
