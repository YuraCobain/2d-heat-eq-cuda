#include <cuda_runtime.h>

#include <SDL.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include "vec4_math.cuh"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at "       \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Args {
  int nx = 512;
  int ny = 512;
  float dx = 1.0f;
  float dy = 1.0f;
  float dt = 0.05f;
  float kappa = 1.0f;
  int steps = 200000;
  int dump_every = 5;
  int fps = 60;

  int src_x = -1;
  int src_y = -1;
  int src_steps = 20;
  float src_amp = 5.0f;

  bool no_vis = false;
  std::string ppm_out;

  // Timing
  bool timing = false;          // measure kernel time each step (CUDA events)
  int timing_print_every = 200; // print running stats every N steps
  int k_ver = 0;
};

static bool parse_int(const char *s, int &out) {
  char *end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0')
    return false;
  out = (int)v;
  return true;
}
static bool parse_float(const char *s, float &out) {
  char *end = nullptr;
  float v = std::strtof(s, &end);
  if (!end || *end != '\0')
    return false;
  out = v;
  return true;
}

static Args parse_args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char *name) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(1);
      }
      return argv[++i];
    };

    if (k == "--nx")
      parse_int(need("--nx"), a.nx);
    else if (k == "--ny")
      parse_int(need("--ny"), a.ny);
    else if (k == "--dx")
      parse_float(need("--dx"), a.dx);
    else if (k == "--dy")
      parse_float(need("--dy"), a.dy);
    else if (k == "--dt")
      parse_float(need("--dt"), a.dt);
    else if (k == "--kappa")
      parse_float(need("--kappa"), a.kappa);
    else if (k == "--steps")
      parse_int(need("--steps"), a.steps);
    else if (k == "--dump-every")
      parse_int(need("--dump-every"), a.dump_every);
    else if (k == "--fps")
      parse_int(need("--fps"), a.fps);
    else if (k == "--src-x")
      parse_int(need("--src-x"), a.src_x);
    else if (k == "--src-y")
      parse_int(need("--src-y"), a.src_y);
    else if (k == "--src-steps")
      parse_int(need("--src-steps"), a.src_steps);
    else if (k == "--src-amp")
      parse_float(need("--src-amp"), a.src_amp);
    else if (k == "--no-vis")
      a.no_vis = true;
    else if (k == "--ppm-out")
      a.ppm_out = need("--ppm-out");
    else if (k == "--timing")
      a.timing = true;
    else if (k == "--timing-print-every")
      parse_int(need("--timing-print-every"), a.timing_print_every);
    else if (k == "--k_ver")
      parse_int(need("--k_ver"), a.k_ver);
    else if (k == "--help" || k == "-h") {
      std::cout << "heat2d (CUDA+SDL2)\n"
                   "--nx N --ny N --dx X --dy Y --dt T --kappa K --steps S\n"
                   "--dump-every D --fps F\n"
                   "--src-x i --src-y j --src-steps P --src-amp A\n"
                   "--no-vis --ppm-out file\n"
                   "--timing --timing-print-every N\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown arg: " << k << "\n";
      std::exit(1);
    }
  }
  if (a.src_x < 0)
    a.src_x = a.nx / 2;
  if (a.src_y < 0)
    a.src_y = a.ny / 2;
  a.dump_every = std::max(1, a.dump_every);
  a.fps = std::max(1, a.fps);
  a.timing_print_every = std::max(1, a.timing_print_every);
  return a;
}

// Online mean/stddev via Welford (sample stddev).
struct RunningStats {
  long long n = 0;
  double mean = 0.0;
  double M2 = 0.0;
  double minv = 0.0;
  double maxv = 0.0;

  void push(double x) {
    if (n == 0) {
      n = 1;
      mean = x;
      M2 = 0.0;
      minv = x;
      maxv = x;
      return;
    }
    n++;
    minv = std::min(minv, x);
    maxv = std::max(maxv, x);
    const double delta = x - mean;
    mean += delta / (double)n;
    const double delta2 = x - mean;
    M2 += delta * delta2;
  }

  double variance_sample() const {
    return (n > 1) ? (M2 / (double)(n - 1)) : 0.0;
  }
  double stddev_sample() const { return std::sqrt(variance_sample()); }
};

// 8th-order (radius-4) 1D second-derivative coefficients
// c0 u0 + c1 (u±1) + c2 (u±2) + c3 (u±3) + c4 (u±4)
__device__ __forceinline__ float d2_8th(const float u0, const float um1,
                                        const float up1, const float um2,
                                        const float up2, const float um3,
                                        const float up3, const float um4,
                                        const float up4) {
  // coefficients:
  // c0 = -205/72
  // c1 =  8/5
  // c2 = -1/5
  // c3 =  8/315
  // c4 = -1/560
  const float c0 = -205.0f / 72.0f;
  const float c1 = 8.0f / 5.0f;
  const float c2 = -1.0f / 5.0f;
  const float c3 = 8.0f / 315.0f;
  const float c4 = -1.0f / 560.0f;
  return c0 * u0 + c1 * (um1 + up1) + c2 * (um2 + up2) + c3 * (um3 + up3) +
         c4 * (um4 + up4);
}

__global__ void step_kernel(float *__restrict__ u_next,
                            const float *__restrict__ u, int nx, int ny,
                            float dt, float kappa, float inv_dx2, float inv_dy2,
                            int src_x, int src_y, float src_add, int do_src) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  // Need a 4-cell border for radius-4 stencil
  if (i < 4 || i >= nx - 4 || j < 4 || j >= ny - 4)
    return;

  const int idx = j * nx + i;

  // gather x-neighbors
  const float u0 = u[idx];
  const float um1 = u[idx - 1];
  const float up1 = u[idx + 1];
  const float um2 = u[idx - 2];
  const float up2 = u[idx + 2];
  const float um3 = u[idx - 3];
  const float up3 = u[idx + 3];
  const float um4 = u[idx - 4];
  const float up4 = u[idx + 4];

  // gather y-neighbors
  const float vm1 = u[idx - nx];
  const float vp1 = u[idx + nx];
  const float vm2 = u[idx - 2 * nx];
  const float vp2 = u[idx + 2 * nx];
  const float vm3 = u[idx - 3 * nx];
  const float vp3 = u[idx + 3 * nx];
  const float vm4 = u[idx - 4 * nx];
  const float vp4 = u[idx + 4 * nx];

  const float uxx =
      d2_8th(u0, um1, up1, um2, up2, um3, up3, um4, up4) * inv_dx2;
  const float uyy =
      d2_8th(u0, vm1, vp1, vm2, vp2, vm3, vp3, vm4, vp4) * inv_dy2;
  float un = u0 + dt * (kappa * (uxx + uyy));

  if (do_src && (i % 128 == 0) && (j % 64 == 0)) {
    un += src_add; // per-step additive injection
  }
  u_next[idx] = un;
}

__device__ __constant__ float c_a, c_b, c_c, c_d, c_e;
__device__ __constant__ Mat4x4f c_Ma, c_Mb, c_Mc;

__device__ __forceinline__ float d2_8th2(float u0, float s1, float s2, float s3, float s4)
{
  float acc = c_a * u0;
  acc = fmaf(c_b, s1, acc);
  acc = fmaf(c_c, s2, acc);
  acc = fmaf(c_d, s3, acc);
  acc = fmaf(c_e, s4, acc);
  return acc;
}

__device__ __forceinline__ float4 d8_packet_x(float4 A, float4 B, float4 C)
{
  float4 out;
  out.x = d2_8th2(B.x, A.w + B.y, A.z + B.z, A.y + B.w, A.x + C.x);
  out.y = d2_8th2(B.y, B.x + B.z, A.w + B.w, A.z + C.x, A.y + C.y);
  out.z = d2_8th2(B.z, B.y + B.w, B.x + C.x, A.w + C.y, A.z + C.z);
  out.w = d2_8th2(B.w, B.z + C.x, B.y + C.y, B.x + C.z, A.w + C.w);
  return out;
}

__device__ __forceinline__ float4 d8_packet_y(float4 u0, float4 s1, float4 s2, float4 s3, float4 s4)
{
  float4 out = mul4(u0, c_a);
  out = fma4(s1, c_b, out);
  out = fma4(s2, c_c, out);
  out = fma4(s3, c_d, out);
  out = fma4(s4, c_e, out);
  return out;
}

__global__ void step_kernel_f4(float4 *__restrict__ u_next4,
                               const float4 *__restrict__ u4,
                               int nx, int ny,
                               float dt, float kappa, float inv_dx2, float inv_dy2,
                               int src_x, int src_y, float src_add, int do_src)
{
  const int W4 = nx >> 2;
  const int x4 = (int)(blockIdx.x * blockDim.x + threadIdx.x) + 1;  // skip x4=0 halo
  const int j  = (int)(blockIdx.y * blockDim.y + threadIdx.y) + 4;  // skip top halo
  if (x4 >= W4 - 1 || j >= ny - 4) return;

  const int idx4 = j * W4 + x4;

  const float4 A = u4[idx4 - 1];
  const float4 B = u4[idx4];
  const float4 C = u4[idx4 + 1];

  // uxx = (Ma*A + Mb*B + Mc*C) * inv_dx2
  float4 uxx = add4(add4(vec_mul(c_Ma, A), vec_mul(c_Mb, B)), vec_mul(c_Mc, C));

  // uyy = (a*B + b*(±1) + c*(±2) + d*(±3) + e*(±4)) * inv_dy2
  float4 uyy = mul4(B, c_a);
  uyy = fma4(add4(u4[idx4 - 1 * W4], u4[idx4 + 1 * W4]), c_b, uyy);
  uyy = fma4(add4(u4[idx4 - 2 * W4], u4[idx4 + 2 * W4]), c_c, uyy);
  uyy = fma4(add4(u4[idx4 - 3 * W4], u4[idx4 + 3 * W4]), c_d, uyy);
  uyy = fma4(add4(u4[idx4 - 4 * W4], u4[idx4 + 4 * W4]), c_e, uyy);

  // un = B + dt*kappa*(uxx + uyy)
  float4 lap = add4(mul4(uxx, inv_dx2), mul4(uyy, inv_dy2));
  float4 un  = add4(B, mul4(lap, dt * kappa));

  // Keep injection semantics identical to your previous float4 version (lane0 only)
  if (do_src && (j % 64 == 0)) {
    int i0 = (x4 << 2);
    if ((i0 % 128) == 0) un.x += src_add;
  }

  u_next4[idx4] = un;
}

template<int SLAB_Y>
__global__ void step_kernel_f4_roll(float4 *__restrict__ u_next4,
                                    const float4 *__restrict__ u4,
                                    int nx, int ny,
                                    float dt, float kappa,
                                    float inv_dx2, float inv_dy2,
                                    int src_x, int src_y,
                                    float src_add, int do_src)
{
  (void)src_x; (void)src_y;

  const int W4 = nx >> 2;
  const int x4 = (int)(blockIdx.x * blockDim.x + threadIdx.x) + 1;
  if (x4 >= W4 - 1) return;

  const int base_y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  const int j0 = 4 + base_y * SLAB_Y;
  if (j0 >= ny - 4) return;

  int idx4 = j0 * W4 + x4;

  // Vertical spine
  float4 t4 = u4[idx4 - 4 * W4];
  float4 t3 = u4[idx4 - 3 * W4];
  float4 t2 = u4[idx4 - 2 * W4];
  float4 t1 = u4[idx4 - 1 * W4];
  float4 m = u4[idx4 + 0 * W4];
  float4 b1 = u4[idx4 + 1 * W4];
  float4 b2 = u4[idx4 + 2 * W4];
  float4 b3 = u4[idx4 + 3 * W4];
  float4 b4 = u4[idx4 + 4 * W4];

#pragma unroll
  for (int s = 0; s < SLAB_Y; ++s) {
    const int j = j0 + s;
    if (j >= ny - 4) break;

    const float4 left = u4[idx4 - 1];
    const float4 midle = m;
    const float4 right = u4[idx4 + 1];

    float4 uxx = d8_packet_x(left, midle, right);
    float4 uyy = d8_packet_y(midle, add4(t1, b1), add4(t2, b2), add4(t3, b3), add4(t4, b4));

    // un = B + dt*kappa*(uxx*inv_dx2 + uyy*inv_dy2)
    float4 lap = add4(mul4(uxx, inv_dx2), mul4(uyy, inv_dy2));
    float4 un  = add4(midle, mul4(lap, dt * kappa));

    if (do_src && ((j & 63) == 0)) {
      int i0 = (x4 << 2);
      if ((i0 & 127) == 0) un.x += src_add;
    }

    u_next4[idx4] = un;

    if (s != SLAB_Y - 1) {
      const int jn = j + 1;
      if (jn < ny - 4) {
        // roll spine
        t4 = t3; t3 = t2; t2 = t1; t1 = m; m = b1;
        b1 = b2; b2 = b3; b3 = b4; b4 = u4[idx4 + 5 * W4];
        idx4 += W4;
      }
    }
  }
}

#define BLOCK_X 32
#define BLOCK_Y 8
#define R 4
#define TILE_X (BLOCK_X + 2*R)   // 40
#define TILE_Y (BLOCK_Y + 2*R)   // 16

__global__ void step_kernel_smem(float *__restrict__ u_next,
                                 const float *__restrict__ u, int nx, int ny,
                                 float dt, float kappa, float inv_dx2,
                                 float inv_dy2, int src_x, int src_y,
                                 float src_add, int do_src)
{
  const int i = (int)(blockIdx.x * BLOCK_X + threadIdx.x);
  const int j = (int)(blockIdx.y * BLOCK_Y + threadIdx.y);

  __shared__ float s_u[TILE_Y * TILE_X];

  // top-left of tile in global coords
  int gx0 = (int)(blockIdx.x * BLOCK_X) - R;
  int gy0 = (int)(blockIdx.y * BLOCK_Y) - R;

  // cooperative load of whole tile (with clamping to avoid OOB)
  for (int ly = (int)threadIdx.y; ly < TILE_Y; ly += BLOCK_Y) {
    int gy = gy0 + ly;
    gy = (gy < 0) ? 0 : (gy >= ny ? (ny - 1) : gy);

    for (int lx = (int)threadIdx.x; lx < TILE_X; lx += BLOCK_X) {
      int gx = gx0 + lx;
      gx = (gx < 0) ? 0 : (gx >= nx ? (nx - 1) : gx);

      s_u[ly * TILE_X + lx] = u[gy * nx + gx];
    }
  }

  __syncthreads();

  // only interior points compute/store
  if (i < R || i >= nx - R || j < R || j >= ny - R) return;

  const int lx = (int)threadIdx.x + R;
  const int ly = (int)threadIdx.y + R;
  const int s_idx = ly * TILE_X + lx;

  const float u00 = s_u[s_idx];

  // x-neighbors
  const float um1 = s_u[s_idx - 1];
  const float up1 = s_u[s_idx + 1];
  const float um2 = s_u[s_idx - 2];
  const float up2 = s_u[s_idx + 2];
  const float um3 = s_u[s_idx - 3];
  const float up3 = s_u[s_idx + 3];
  const float um4 = s_u[s_idx - 4];
  const float up4 = s_u[s_idx + 4];

  // y-neighbors
  const float vm1 = s_u[s_idx - 1 * TILE_X];
  const float vp1 = s_u[s_idx + 1 * TILE_X];
  const float vm2 = s_u[s_idx - 2 * TILE_X];
  const float vp2 = s_u[s_idx + 2 * TILE_X];
  const float vm3 = s_u[s_idx - 3 * TILE_X];
  const float vp3 = s_u[s_idx + 3 * TILE_X];
  const float vm4 = s_u[s_idx - 4 * TILE_X];
  const float vp4 = s_u[s_idx + 4 * TILE_X];

  const float uxx =
      d2_8th(u00, um1, up1, um2, up2, um3, up3, um4, up4) * inv_dx2;
  const float uyy =
      d2_8th(u00, vm1, vp1, vm2, vp2, vm3, vp3, vm4, vp4) * inv_dy2;

  float un = u00 + dt * (kappa * (uxx + uyy));

  if (do_src && (i % 128 == 0) && (j % 64 == 0)) un += src_add;

  u_next[j * nx + i] = un;
}

// Simple PPM (P6) writer for final snapshot
static void write_ppm(const std::string &path, const std::vector<uint8_t> &rgb,
                      int w, int h) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::cerr << "Failed to open " << path << " for writing\n";
    return;
  }
  std::fprintf(f, "P6\n%d %d\n255\n", w, h);
  std::fwrite(rgb.data(), 1, rgb.size(), f);
  std::fclose(f);
}

int main(int argc, char **argv) {
  Args a = parse_args(argc, argv);

  const int nx = a.nx;
  const int ny = a.ny;
  const size_t n = (size_t)nx * (size_t)ny;

  // Host buffers
  std::vector<float> h_u(n, 0.0f);
  std::vector<uint8_t> h_rgba((size_t)nx * (size_t)ny * 4, 255);

  // Device buffers
  float *d_u0 = nullptr, *d_u1 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_u0, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_u1, n * sizeof(float)));


  CUDA_CHECK(cudaMemset(d_u0, 0, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_u1, 0, n * sizeof(float)));

  const float inv_dx2 = 1.0f / (a.dx * a.dx);
  const float inv_dy2 = 1.0f / (a.dy * a.dy);

  // SDL init
  SDL_Window *window = nullptr;
  SDL_Renderer *renderer = nullptr;
  SDL_Texture *texture = nullptr;

  if (!a.no_vis) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
      std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
      return 1;
    }
    window = SDL_CreateWindow("heat2d (CUDA)", SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED, nx, ny,
                              SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window) {
      std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
      return 1;
    }
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
      std::cerr << "SDL_CreateRenderer failed: " << SDL_GetError() << "\n";
      return 1;
    }
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
                                SDL_TEXTUREACCESS_STREAMING, nx, ny);
    if (!texture) {
      std::cerr << "SDL_CreateTexture failed: " << SDL_GetError() << "\n";
      return 1;
    }
  }

  dim3 block(BLOCK_X, BLOCK_Y, 1);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);
  
  const int W4 = nx / 4;
  dim3 grid4((unsigned)(((W4 - 2) + block.x - 1) / block.x),
           (unsigned)(((ny - 8) + block.y - 1) / block.y),
           1);
  
   if (a.k_ver == 2 || a.k_ver == 3)
    {
      // 8th-order Laplace coefficients
      const float a = -205.0f / 72.0f;
      const float b =   8.0f /  5.0f;
      const float c =  -1.0f /  5.0f;
      const float d =   8.0f / 315.0f;
      const float e =  -1.0f / 560.0f;

      Mat4x4f Ma{ float4{e,d,c,b},
                  float4{0,e,d,c},
                  float4{0,0,e,d},
                  float4{0,0,0,e} };
      
      Mat4x4f Mb{ float4{a,b,c,d},
                  float4{b,a,b,c},
                  float4{c,b,a,b},
                  float4{d,c,b,a} };
      
      Mat4x4f Mc{float4{e,0,0,0},
                 float4{d,e,0,0},
                 float4{c,d,e,0},
                 float4{b,c,d,e} };

      CUDA_CHECK(cudaMemcpyToSymbol(c_a, &a, sizeof(a)));
      CUDA_CHECK(cudaMemcpyToSymbol(c_b, &b, sizeof(b)));
      CUDA_CHECK(cudaMemcpyToSymbol(c_c, &c, sizeof(c)));
      CUDA_CHECK(cudaMemcpyToSymbol(c_d, &d, sizeof(d)));
      CUDA_CHECK(cudaMemcpyToSymbol(c_e, &e, sizeof(e)));

      CUDA_CHECK(cudaMemcpyToSymbol(c_Ma, &Ma, sizeof(Ma)));
      CUDA_CHECK(cudaMemcpyToSymbol(c_Mb, &Mb, sizeof(Mb)));
      CUDA_CHECK(cudaMemcpyToSymbol(c_Mc, &Mc, sizeof(Mc)));
   }

  // Optional precise per-step kernel timing (CUDA events). Note: this
  // synchronizes each step.
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_stop = nullptr;
  RunningStats kstats;
  if (a.timing) {
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  int rendered = 0;

  const double target_frame_ms = 1000.0 / (double)a.fps;
  auto last_frame = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < a.steps; ++step) {
    const int do_src = (step < a.src_steps && step < 100) ? 1 : 0;
    const float src_add = (do_src ? a.src_amp : 0.0f);

    if (a.timing)
      CUDA_CHECK(cudaEventRecord(ev_start));

    switch (a.k_ver) {
    case 0:
      step_kernel<<<grid, block>>>(
          d_u1, d_u0, nx, ny, a.dt, a.kappa, inv_dx2, inv_dy2, a.src_x, a.src_y,
          src_add, do_src);
      break;
    case 1:
      step_kernel_smem<<<grid, block>>>(
          d_u1, d_u0, nx, ny, a.dt, a.kappa, inv_dx2, inv_dy2, a.src_x, a.src_y,
          src_add, do_src);
      break;
   case 2:
      step_kernel_f4<<<grid4, block>>>((float4*)d_u1, (const float4*)d_u0,
                                   nx, ny, a.dt, a.kappa, inv_dx2, inv_dy2,
                                   a.src_x, a.src_y, src_add, do_src);
      break;
     
     case 3: {
       constexpr int SLAB_Y = 4;
       dim3 grid4_roll2(
           grid4.x,
           (unsigned)(((ny - 8) + (block.y * SLAB_Y - 1)) / (block.y * SLAB_Y)),
           1);
       step_kernel_f4_roll<SLAB_Y><<<grid4_roll2, block>>>(
                                      (float4*)d_u1, (const float4*)d_u0,
                                       nx, ny, a.dt, a.kappa, inv_dx2, inv_dy2,
                                       a.src_x, a.src_y, src_add, do_src);
     }
      break;
      
    default:
      std::abort();
    }

    CUDA_CHECK(cudaGetLastError());
    if (a.timing) {
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      kstats.push((double)ms);
      if ((step + 1) % a.timing_print_every == 0) {
        std::cout << "[timing] step=" << (step + 1)
                  << "  kernel_ms(mean)=" << kstats.mean
                  << "  kernel_ms(std)=" << kstats.stddev_sample()
                  << "  min=" << kstats.minv << "  max=" << kstats.maxv
                  << "  n=" << kstats.n << "\n";
      }
    }

    // swap buffers
    std::swap(d_u0, d_u1);

    const bool want_render = (!a.no_vis) && (step % a.dump_every == 0);
    if (want_render) {
      // simple FPS cap
      auto now = std::chrono::high_resolution_clock::now();
      double ms_since =
          std::chrono::duration<double, std::milli>(now - last_frame).count();
      if (ms_since < target_frame_ms) {
        SDL_Delay((Uint32)(target_frame_ms - ms_since));
      }
      last_frame = std::chrono::high_resolution_clock::now();

      // Copy to host
      CUDA_CHECK(cudaMemcpy(h_u.data(), d_u0, n * sizeof(float),
                            cudaMemcpyDeviceToHost));

      // Auto-contrast (robust-ish): find min/max on interior
      float mn = h_u[4 * nx + 4], mx = mn;
      for (int j = 4; j < ny - 4; ++j) {
        const int row = j * nx;
        for (int i = 4; i < nx - 4; ++i) {
          float v = h_u[row + i];
          mn = std::min(mn, v);
          mx = std::max(mx, v);
        }
      }
      // If mostly constant, avoid division by ~0
      const float eps = 1e-12f;
      const float inv = 1.0f / std::max(mx - mn, eps);

      // Map to grayscale
      for (size_t p = 0; p < n; ++p) {
        float v = (h_u[p] - mn) * inv; // 0..1
        v = std::min(1.0f, std::max(0.0f, v));
        uint8_t c = (uint8_t)(v * 255.0f);
        h_rgba[4 * p + 0] = c;
        h_rgba[4 * p + 1] = c;
        h_rgba[4 * p + 2] = c;
        h_rgba[4 * p + 3] = 255;
      }

      SDL_UpdateTexture(texture, nullptr, h_rgba.data(), nx * 4);
      SDL_RenderClear(renderer);
      SDL_RenderCopy(renderer, texture, nullptr, nullptr);
      SDL_RenderPresent(renderer);

      // handle events
      SDL_Event e;
      while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
          step = a.steps; // break outer loop
          break;
        }
        if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) {
          step = a.steps;
          break;
        }
      }

      rendered++;
      if (rendered % 60 == 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double sps = (double)(step + 1) / sec;
        std::cout << "step " << step << "/" << a.steps << "  steps/s=" << sps
                  << "\n";
      }
    }
  }

  if (a.timing) {
    std::cout << "[timing] FINAL  kernel_ms(mean)=" << kstats.mean
              << "  kernel_ms(std)=" << kstats.stddev_sample()
              << "  min=" << kstats.minv << "  max=" << kstats.maxv
              << "  n=" << kstats.n << "\n";
  }

  // Final snapshot for optional PPM
  if (!a.ppm_out.empty()) {
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u0, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float mn = h_u[4 * nx + 4], mx = mn;
    for (int j = 4; j < ny - 4; ++j) {
      const int row = j * nx;
      for (int i = 4; i < nx - 4; ++i) {
        float v = h_u[row + i];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      }
    }
    const float eps = 1e-12f;
    const float inv = 1.0f / std::max(mx - mn, eps);

    std::vector<uint8_t> rgb((size_t)nx * (size_t)ny * 3);
    for (size_t p = 0; p < n; ++p) {
      float v = (h_u[p] - mn) * inv;
      v = std::min(1.0f, std::max(0.0f, v));
      uint8_t c = (uint8_t)(v * 255.0f);
      rgb[3 * p + 0] = c;
      rgb[3 * p + 1] = c;
      rgb[3 * p + 2] = c;
    }
    write_ppm(a.ppm_out, rgb, nx, ny);
    std::cout << "Wrote " << a.ppm_out << "\n";
  }

  CUDA_CHECK(cudaFree(d_u0));
  CUDA_CHECK(cudaFree(d_u1));

  if (a.timing) {
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
  }

  if (!a.no_vis) {
    if (texture)
      SDL_DestroyTexture(texture);
    if (renderer)
      SDL_DestroyRenderer(renderer);
    if (window)
      SDL_DestroyWindow(window);
    SDL_Quit();
  }

  return 0;
}
