#include <cuda_runtime.h>

#include <SDL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

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

#define BLOCK_X 32
#define BLOCK_Y 8

#define BLOCK_HALO_X 32 + 4
#define BLOCK_HALO_Y 8 + 4
__global__ void step_kernel_smem(float *__restrict__ u_next,
                                 const float *__restrict__ u, int nx, int ny,
                                 float dt, float kappa, float inv_dx2,
                                 float inv_dy2, int src_x, int src_y,
                                 float src_add, int do_src) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  // Need a 4-cell border for radius-4 stencil
  if (i < 4 || i >= nx - 4 || j < 4 || j >= ny - 4)
    return;

  __shared__ float s_u[(BLOCK_HALO_X) * (BLOCK_HALO_Y)];

  const int idx = j * nx + i;
  const int s_idx = (threadIdx.y + 4) * (BLOCK_HALO_X) + (threadIdx.x + 4);

  s_u[s_idx] = u[idx];
  if (threadIdx.y == 0) {
    s_u[s_idx - 4] = u[idx - 4];
    s_u[s_idx - 3] = u[idx - 3];
    s_u[s_idx - 2] = u[idx - 2];
    s_u[s_idx - 1] = u[idx - 1];
  }
  if (threadIdx.x == 0) {
    s_u[s_idx - 4 * BLOCK_HALO_X] = u[idx - 4 * nx];
    s_u[s_idx - 3 * BLOCK_HALO_X] = u[idx - 3 * nx];
    s_u[s_idx - 2 * BLOCK_HALO_X] = u[idx - 2 * nx];
    s_u[s_idx - 1 * BLOCK_HALO_X] = u[idx - 1 * nx];
  }
  if (threadIdx.y == BLOCK_Y) {
    s_u[s_idx + 4] = u[idx + 4];
    s_u[s_idx + 3] = u[idx + 3];
    s_u[s_idx + 2] = u[idx + 2];
    s_u[s_idx + 1] = u[idx + 1];
  }
  if (threadIdx.x == BLOCK_X) {
    s_u[s_idx + 4 * BLOCK_HALO_X] = u[idx + 4 * nx];
    s_u[s_idx + 3 * BLOCK_HALO_X] = u[idx + 3 * nx];
    s_u[s_idx + 2 * BLOCK_HALO_X] = u[idx + 2 * nx];
    s_u[s_idx + 1 * BLOCK_HALO_X] = u[idx + 1 * nx];
  }

  __syncthreads();

  // gather x-neighbors
  const float u00 = s_u[s_idx + 0];
  const float um1 = s_u[s_idx - 1];
  const float up1 = s_u[s_idx + 1];
  const float um2 = s_u[s_idx - 2];
  const float up2 = s_u[s_idx + 2];
  const float um3 = s_u[s_idx - 3];
  const float up3 = s_u[s_idx + 3];
  const float um4 = s_u[s_idx - 4];
  const float up4 = s_u[s_idx + 4];

  // gather y-neighbors
  const float vm1 = s_u[s_idx - 1 * BLOCK_HALO_X];
  const float vp1 = s_u[s_idx + 1 * BLOCK_HALO_X];
  const float vm2 = s_u[s_idx - 2 * BLOCK_HALO_X];
  const float vp2 = s_u[s_idx + 2 * BLOCK_HALO_X];
  const float vm3 = s_u[s_idx - 3 * BLOCK_HALO_X];
  const float vp3 = s_u[s_idx + 3 * BLOCK_HALO_X];
  const float vm4 = s_u[s_idx - 4 * BLOCK_HALO_X];
  const float vp4 = s_u[s_idx + 4 * BLOCK_HALO_X];

  const float uxx =
      d2_8th(u00, um1, up1, um2, up2, um3, up3, um4, up4) * inv_dx2;
  const float uyy =
      d2_8th(u00, vm1, vp1, vm2, vp2, vm3, vp3, vm4, vp4) * inv_dy2;
  float un = u00 + dt * (kappa * (uxx + uyy));

  if (do_src && (i % 128 == 0) && (j % 64 == 0)) {
    un += src_add; // per-step additive injection
  }
  u_next[idx] = un;
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

  dim3 block(16, 16, 1);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);

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
      step_kernel<<<grid, {BLOCK_X, BLOCK_Y}>>>(
          d_u1, d_u0, nx, ny, a.dt, a.kappa, inv_dx2, inv_dy2, a.src_x, a.src_y,
          src_add, do_src);
      break;
    case 1:
      // XXX DOESNT WORK
      step_kernel_smem<<<grid, {BLOCK_X, BLOCK_Y}>>>(
          d_u1, d_u0, nx, ny, a.dt, a.kappa, inv_dx2, inv_dy2, a.src_x, a.src_y,
          src_add, do_src);
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
