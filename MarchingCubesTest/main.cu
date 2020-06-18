#include <cuda_runtime_api.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

#include <MC.h>

using namespace ComputeStuff::MC;

namespace {

  FieldFormat format = FieldFormat::UInt8;
  uint32_t nx = 128;
  uint32_t ny = 128;
  uint32_t nz = 128;

  std::vector<char> scalarField_host;

  void onGLFWError(int error, const char* what)
  {
    fprintf(stderr, "GLFW Error: %s\n", what);
  }

  const std::string simpleVS_src = R"(#version 430

in layout(location=0) vec3 inPosition;
in layout(location=1) vec3 inNormal;

out vec3 normal;

uniform layout(location=0) mat4 MV;
uniform layout(location=1) mat4 MVP;

void main() {
  normal = mat3(MV)*inNormal;
  gl_Position = MVP * vec4(inPosition, 1);
  //gl_Position = vec4(inPosition, 1);
}
)";

  const std::string simpleFS_src = R"(#version 430

in vec3 normal;

out layout(location=0) vec4 outColor;

void main() {
  outColor = vec4(abs(normal), 1);
  //outColor = vec4(1,0,0,1);
}

)";

  GLfloat vertexData[]
  {
     1.f, 0.f, 0.f,  1.f, 0.f, 0.f,
     1.f, 1.f, 0.f,  1.f, 0.f, 0.f,
     1.f, 1.f, 1.f,  1.f, 0.f, 0.f,
     1.f, 1.f, 1.f,  1.f, 0.f, 0.f,
     1.f, 0.f, 1.f,  1.f, 0.f, 0.f,
     1.f, 0.f, 0.f,  1.f, 0.f, 0.f,

     0.f, 1.f, 0.f,  0.f, 1.f, 0.f,
     0.f, 1.f, 1.f,  0.f, 1.f, 0.f,
     1.f, 1.f, 1.f,  0.f, 1.f, 0.f,
     1.f, 1.f, 1.f,  0.f, 1.f, 0.f,
     1.f, 1.f, 0.f,  0.f, 1.f, 0.f,
     0.f, 1.f, 0.f,  0.f, 1.f, 0.f,

     0.f, 0.f, 1.f,  0.f, 0.f, 1.f,
     1.f, 0.f, 1.f,  0.f, 0.f, 1.f,
     1.f, 1.f, 1.f,  0.f, 0.f, 1.f,
     1.f, 1.f, 1.f,  0.f, 0.f, 1.f,
     0.f, 1.f, 1.f,  0.f, 0.f, 1.f,
     0.f, 0.f, 1.f,  0.f, 0.f, 1.f,

     0.f, 0.f, 0.f,  -1.f, 0.f, 0.f,
     0.f, 1.f, 1.f,  -1.f, 0.f, 0.f,
     0.f, 1.f, 0.f,  -1.f, 0.f, 0.f,
     0.f, 1.f, 1.f,  -1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,  -1.f, 0.f, 0.f,
     0.f, 0.f, 1.f,  -1.f, 0.f, 0.f,

     0.f, 0.f, 0.f,  0.f, -1.f, 0.f,
     1.f, 0.f, 1.f,  0.f, -1.f, 0.f,
     0.f, 0.f, 1.f,  0.f, -1.f, 0.f,
     1.f, 0.f, 1.f,  0.f, -1.f, 0.f,
     0.f, 0.f, 0.f,  0.f, -1.f, 0.f,
     1.f, 0.f, 0.f,  0.f, -1.f, 0.f,

     0.f, 0.f, 0.f,  0.f, 0.f, -1.f,
     1.f, 1.f, 0.f,  0.f, 0.f, -1.f,
     1.f, 0.f, 0.f,  0.f, 0.f, -1.f,
     1.f, 1.f, 0.f,  0.f, 0.f, -1.f,
     0.f, 0.f, 0.f,  0.f, 0.f, -1.f,
     0.f, 1.f, 0.f,  0.f, 0.f, -1.f,
  };


  [[noreturn]]
  void handleOpenGLError(GLenum error, const std::string file, int line)
  {
    do {
      switch (error) {
      case GL_INVALID_ENUM: fprintf(stderr, "GL_INVALID_ENUM\n"); break;
      case GL_INVALID_VALUE: fprintf(stderr, "GL_INVALID_VALUE\n"); break;
      case GL_INVALID_OPERATION: fprintf(stderr, "GL_INVALID_OPERATION\n"); break;
      case GL_INVALID_FRAMEBUFFER_OPERATION: fprintf(stderr, "GL_INVALID_FRAMEBUFFER_OPERATION\n"); break;
      case GL_OUT_OF_MEMORY: fprintf(stderr, "GL_OUT_OF_MEMORY\n"); break;
      case GL_STACK_OVERFLOW: fprintf(stderr, "GL_STACK_OVERFLOW\n"); break;
      case GL_STACK_UNDERFLOW: fprintf(stderr, "GL_STACK_UNDERFLOW\n"); break;
      default: fprintf(stderr, "Unknown error"); break;
      }
      error = glGetError();
    } while (error != GL_NO_ERROR);
    exit(EXIT_FAILURE);
  }

#define CHECK_GL do { GLenum error = glGetError(); if(error != GL_NO_ERROR) handleOpenGLError(error, __FILE__, __LINE__); } while(0)

  [[noreturn]]
  void handleCudaError(cudaError_t error, const std::string file, int line)
  {
    fprintf(stderr, "%s@%d: CUDA: %s\n", file.c_str(), line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

#define CHECK_CUDA do { cudaError_t error = cudaGetLastError(); if(error != cudaSuccess) handleCudaError(error, __FILE__, __LINE__); } while(0)

#define CHECKED_CUDA(a) do { cudaError_t error = (a); if(error != cudaSuccess) handleCudaError(error, __FILE__, __LINE__); } while(0)

  GLuint createShader(const std::string& src, GLenum shader_type)
  {
    GLuint shader = glCreateShader(shader_type);

    const char* src_array[] = { src.c_str() };
    glShaderSource(shader, 1, src_array, nullptr);
    glCompileShader(shader);

    GLsizei bufSize;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &bufSize);
    if (bufSize) {
      fprintf(stderr, "Source:\n%s", src.c_str());
      std::vector<char> log(bufSize + 1);
      glGetShaderInfoLog(shader, bufSize + 1, nullptr, log.data());
      fprintf(stderr, "Compilator output:\n%s", log.data());
    }

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
      glDeleteShader(shader);
      return 0;
    }
    return shader;
  }

  GLuint createProgram(GLuint VS, GLuint FS)
  {
    GLuint program = glCreateProgram();
    glAttachShader(program, VS);
    glAttachShader(program, FS);
    glLinkProgram(program);


    GLsizei bufSize;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufSize);
    if (bufSize) {
      std::vector<char> log(bufSize + 1);
      glGetProgramInfoLog(program, bufSize + 1, nullptr, log.data());
      fprintf(stderr, "Linker output:\n%s", log.data());
    }

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
      glDeleteProgram(program);
      return 0;
    }
    return program;
  }

  GLuint createBuffer(GLenum target, GLenum usage, size_t size, const void* data)
  {
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(target, buffer);
    glBufferData(target, size, data, usage);
    CHECK_GL;
    return buffer;
  }


  void rotMatrixX(float* dst, const float angle)
  {
    const auto c = std::cos(angle);
    const auto s = std::sin(angle);
    dst[4 * 0 + 0] = 1.f; dst[4 * 0 + 1] = 0.f; dst[4 * 0 + 2] = 0.f; dst[4 * 0 + 3] = 0.f;
    dst[4 * 1 + 0] = 0.f; dst[4 * 1 + 1] = c;   dst[4 * 1 + 2] = s;   dst[4 * 1 + 3] = 0.f;
    dst[4 * 2 + 0] = 0.f; dst[4 * 2 + 1] = -s;  dst[4 * 2 + 2] = c;   dst[4 * 2 + 3] = 0.f;
    dst[4 * 3 + 0] = 0.f; dst[4 * 3 + 1] = 0.f; dst[4 * 3 + 2] = 0.f; dst[4 * 3 + 3] = 1.f;
  }

  void rotMatrixY(float* dst, const float angle)
  {
    const auto c = std::cos(angle);
    const auto s = std::sin(angle);
    dst[4 * 0 + 0] = c;   dst[4 * 0 + 1] = 0.f; dst[4 * 0 + 2] = -s;  dst[4 * 0 + 3] = 0.f;
    dst[4 * 1 + 0] = 0.f; dst[4 * 1 + 1] = 1.f; dst[4 * 1 + 2] = 0.f; dst[4 * 1 + 3] = 0.f;
    dst[4 * 2 + 0] = s;   dst[4 * 2 + 1] = 0.f; dst[4 * 2 + 2] = c;   dst[4 * 2 + 3] = 0.f;
    dst[4 * 3 + 0] = 0.f; dst[4 * 3 + 1] = 0.f; dst[4 * 3 + 2] = 0.f; dst[4 * 3 + 3] = 1.f;
  }

  void rotMatrixZ(float* dst, const float angle)
  {
    const auto c = std::cos(angle);
    const auto s = std::sin(angle);
    dst[4 * 0 + 0] = c;   dst[4 * 0 + 1] = s;   dst[4 * 0 + 2] = 0.f; dst[4 * 0 + 3] = 0.f;
    dst[4 * 1 + 0] = -s;  dst[4 * 1 + 1] = c;   dst[4 * 1 + 2] = 0.f; dst[4 * 1 + 3] = 0.f;
    dst[4 * 2 + 0] = 0.f; dst[4 * 2 + 1] = 0.f; dst[4 * 2 + 2] = 1.f; dst[4 * 2 + 3] = 0.f;
    dst[4 * 3 + 0] = 0.f; dst[4 * 3 + 1] = 0.f; dst[4 * 3 + 2] = 0.f; dst[4 * 3 + 3] = 1.f;
  }

  void translateMatrix(float* dst, const float x, const float y, const float z)
  {
    dst[4 * 0 + 0] = 1.f; dst[4 * 0 + 1] = 0.f; dst[4 * 0 + 2] = 0.f; dst[4 * 0 + 3] = 0.f;
    dst[4 * 1 + 0] = 0.f; dst[4 * 1 + 1] = 1.f; dst[4 * 1 + 2] = 0.f; dst[4 * 1 + 3] = 0.f;
    dst[4 * 2 + 0] = 0.f; dst[4 * 2 + 1] = 0.f; dst[4 * 2 + 2] = 1.f; dst[4 * 2 + 3] = 0.f;
    dst[4 * 3 + 0] = x;   dst[4 * 3 + 1] = y;   dst[4 * 3 + 2] = z;   dst[4 * 3 + 3] = 1.f;
  }


  void frustumMatrix(float* dst, const float w, const float h, const float n, const float f)
  {
    auto a = 2.f*n / w;
    auto b = 2.f*n / h;
    auto c = -(f + n) / (f - n);
    auto d = -2.f*f*n / (f - n);
    dst[4 * 0 + 0] = a;   dst[4 * 0 + 1] = 0.f; dst[4 * 0 + 2] = 0.f; dst[4 * 0 + 3] = 0.f;
    dst[4 * 1 + 0] = 0.f; dst[4 * 1 + 1] = b;   dst[4 * 1 + 2] = 0.f; dst[4 * 1 + 3] = 0.f;
    dst[4 * 2 + 0] = 0.f; dst[4 * 2 + 1] = 0.f; dst[4 * 2 + 2] = c;   dst[4 * 2 + 3] =-1.f;
    dst[4 * 3 + 0] = 0.f; dst[4 * 3 + 1] = 0;   dst[4 * 3 + 2] = d;   dst[4 * 3 + 3] = 0.f;
  }


  void matrixMul4(float* D, const float *A, const float *B)
  {
    for (unsigned i = 0; i < 4; i++) {
      for (unsigned j = 0; j < 4; j++) {

        float sum = 0.f;
        for (unsigned k = 0; k < 4; k++) {
          sum += A[4 * k + j] * B[4 * i + k];
        }
        D[4 * i + j] = sum;
      }
    }
  }

  constexpr float cayley(unsigned i, unsigned j, unsigned k, unsigned nx, unsigned ny, unsigned nz)
  {
    float x = (2.f * i) / (nx - 1.f) - 1.f;
    float y = (2.f * j) / (ny - 1.f) - 1.f;
    float z = (2.f * k) / (nz - 1.f) - 1.f;
    float v = 1.f - 16.f * x * y * z - 4.f * (x * x + y * y + z * z);
    return v;
  }

  void buildCayleyField()
  {
    const size_t N = static_cast<size_t>(nx) * ny * nz;
    switch (format) {
    case FieldFormat::UInt8: {
      scalarField_host.resize(N);
      auto* dst = reinterpret_cast<uint8_t*>(scalarField_host.data());
      for (unsigned k = 0; k < nz; k++) {
        for (unsigned j = 0; j < ny; j++) {
          for (unsigned i = 0; i < nx; i++) {
            float v = cayley(i, j, k, nx, ny, nz);
            v = 0.5f * 255.f * (v + 1.f);
            if (v < 0.f) v = 0.f;
            if (255.f < v) v = 255.f;
            *dst++ = static_cast<uint8_t>(v);
          }
        }
      }
      break;
    }
    case FieldFormat::UInt16: {
      scalarField_host.resize(sizeof(uint16_t) * N);
      auto* dst = reinterpret_cast<uint16_t*>(scalarField_host.data());
      for (unsigned k = 0; k < nz; k++) {
        for (unsigned j = 0; j < ny; j++) {
          for (unsigned i = 0; i < nx; i++) {
            float v = cayley(i, j, k, nx, ny, nz);
            v = 0.5f * 65535.f * (v + 1.f);
            if (v < 0.f) v = 0.f;
            if (65535.f < v) v = 65535.f;
            *dst++ = static_cast<uint16_t>(v);
          }
        }
      }
      break;
    }
    case FieldFormat::Float: {
      scalarField_host.resize(sizeof(float) * N);
      auto* dst = reinterpret_cast<float*>(scalarField_host.data());
      for (unsigned k = 0; k < nz; k++) {
        for (unsigned j = 0; j < ny; j++) {
          for (unsigned i = 0; i < nx; i++) {
            *dst++ = cayley(i, j, k, nx, ny, nz);
          }
        }
      }
      break;
    }
    default:
      assert(false && "Unhandled case");
      break;
    }
  }


  bool readFile(const char* path)
  {
    assert(path);
    fprintf(stderr, "Reading %s...\n", path);

    FILE* fp = fopen(path, "rb");
    if (!fp) {
      fprintf(stderr, "Error opening file \"%s\" for reading.\n", path);
      return false;
    }
    if (fseek(fp, 0L, SEEK_END) == 0) {
      uint8_t header[6];
      long size = ftell(fp);
      if (sizeof(header) <= size) {
        if (fseek(fp, 0L, SEEK_SET) == 0) {
          if (fread(header, sizeof(header), 1, fp) == 1) {
            nx = header[0] | header[1] << 8;
            ny = header[2] | header[3] << 8;
            nz = header[4] | header[5] << 8;
            size_t N = static_cast<size_t>(nx) * ny * nz;
            if ((N + 3) * 2 != size) {
              fprintf(stderr, "Unexpected file size.\n");
            }
            else {
              std::vector<uint8_t> tmp(2*N);
              if (fread(tmp.data(), 2, N, fp) == N) {
                switch (format) {
                case FieldFormat::UInt8: {
                  scalarField_host.resize(N);
                  const auto* dst = reinterpret_cast<uint8_t*>(scalarField_host.data());
                  for (size_t i = 0; i < N; i++) {
                    const uint32_t v = tmp[2 * i + 0] | tmp[2 * i + 1] << 8;
                    scalarField_host[i] = v >> 4; // 12 bits are in use.
                  }
                  break;
                }
                case FieldFormat::UInt16: {
                  scalarField_host.resize(sizeof(uint16_t) * N);
                  const auto* dst = reinterpret_cast<uint16_t*>(scalarField_host.data());
                  for (size_t i = 0; i < N; i++) {
                    const uint32_t v = tmp[2 * i + 0] | tmp[2 * i + 1] << 8;
                    scalarField_host[i] = v;
                  }
                  break;
                }
                case FieldFormat::Float: {
                  scalarField_host.resize(sizeof(float) * N);
                  const auto* dst = reinterpret_cast<float*>(scalarField_host.data());
                  for (size_t i = 0; i < N; i++) {
                    const uint32_t v = tmp[2 * i + 0] | tmp[2 * i + 1] << 8;
                    scalarField_host[i] = v;
                  }
                  break;
                }
                default:
                  assert(false && "Unhandled case");
                }
                fprintf(stderr, "Successfully loaded %s\n", path);
                fclose(fp);
                return true;
              }
            }
          }
        }
      }
    }
    fprintf(stderr, "Error loading \"%s\"", path);
    fclose(fp);
    return false;
  }


}



int main(int argc, char** argv)
{
  cudaStream_t stream;
  GLFWwindow* win;
  const char* path = nullptr;
  int deviceIndex = 0;

  for (int i = 1; i < argc; i++) {
    if (i + 1 < argc && (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0)) { deviceIndex = std::atoi(argv[i+1]); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-nx") == 0) { nx = uint32_t(std::atoi(argv[i+1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-ny") == 0) { ny = uint32_t(std::atoi(argv[i+1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-nz") == 0) { nz = uint32_t(std::atoi(argv[i+1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-n") == 0) { nx = uint32_t(std::atoi(argv[i+1])); ny = nx; nz = nx; i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-f") == 0) {
      if (strcmp(argv[i + 1], "uint8") == 0) { format = FieldFormat::UInt8; }
      else if (strcmp(argv[i + 1], "uint16") == 0) { format = FieldFormat::UInt16; }
      else if (strcmp(argv[i + 1], "float") == 0) { format = FieldFormat::Float; }
      else {
        fprintf(stderr, "Unknown format '%s'", argv[i + 1]);
        return EXIT_FAILURE;
      }
      i++;
    }
    else {
      if (path) {
        fprintf(stderr, "%s: input file already specified\n", argv[i]);
        return EXIT_FAILURE;
      }
      path = argv[i];
    }
  }


  glfwSetErrorCallback(onGLFWError);
  if (!glfwInit()) {
    fprintf(stderr, "GLFW failed to initialize.\n");
    return EXIT_FAILURE;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  win = glfwCreateWindow(1280, 720, "Marching cubes test application", nullptr, nullptr);
  glfwMakeContextCurrent(win);
  gladLoadGL(glfwGetProcAddress);

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "No CUDA-enabled devices available.");
    return EXIT_FAILURE;
  }
  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, i);
    fprintf(stderr, "%c[%i] %s cap=%d.%d\n", i==deviceIndex ? '*' : ' ', i, dev_prop.name, dev_prop.major, dev_prop.minor);
  }
  if (deviceIndex < 0 || deviceCount <= deviceIndex) {
    fprintf(stderr, "Illegal CUDA device index %d\n", deviceIndex);
    return EXIT_FAILURE;
  }
  cudaSetDevice(deviceIndex);
  CHECKED_CUDA(cudaStreamCreate(&stream));


  // Set up scalar field
  if (!path) {
    fprintf(stderr, "No input file specified.\n");
    return EXIT_FAILURE;
  }
  else if (strcmp("cayley", path) == 0) {
    buildCayleyField();
  }
  else if (!readFile(path)) {
    return EXIT_FAILURE;
  }
  fprintf(stderr, "Scalar field is [%d x %d x %d] (%d cells total)\n", nx, ny, nz, nx * ny * nz);
  void* deviceMem = nullptr;
  CHECKED_CUDA(cudaMalloc(&deviceMem, scalarField_host.size()));
  CHECKED_CUDA(cudaMemcpy(deviceMem, scalarField_host.data(), scalarField_host.size(), cudaMemcpyHostToDevice));


  auto * tables = createTables(stream);
  auto * pyramid = createHistoPyramid(stream, tables, nx, ny, nz);


  GLuint simpleVS = createShader(simpleVS_src, GL_VERTEX_SHADER);
  assert(simpleVS != 0);

  GLuint simpleFS = createShader(simpleFS_src, GL_FRAGMENT_SHADER);
  assert(simpleFS != 0);

  GLuint simplePrg = createProgram(simpleVS, simpleFS);
  assert(simplePrg != 0);

  GLuint vdatabuf = createBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(vertexData), (const void*)vertexData);
  assert(vdatabuf != 0);

  GLuint vbo = 0;
  glGenVertexArrays(1, &vbo);
  glBindVertexArray(vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vdatabuf);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, nullptr);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void*)(sizeof(float)*3));
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);


  auto start = std::chrono::system_clock::now();
  while (!glfwWindowShouldClose(win)) {
    int width, height;
    glfwGetWindowSize(win, &width, &height);

    float iso = 0.0;
    switch (format) {
    case FieldFormat::UInt8:
      iso = 0.5f*255.f*(iso + 1.f);
      break;
    case FieldFormat::UInt16:
      iso = 0.5f*65535.f*(iso + 1.f);
    default: break;
    }

    buildHistoPyramid(stream, pyramid, iso);

    glViewport(0, 0, width, height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
    auto seconds = elapsed.count();

    float center[16];
    translateMatrix(center, -0.5f, -0.5f, -0.5f);

    float rx[16];
    rotMatrixX(rx, static_cast<float>(1.1*seconds));
 
    float ry[16];
    rotMatrixY(ry, static_cast<float>(1.7*seconds));

    float rz[16];
    rotMatrixZ(rz, static_cast<float>(1.3*seconds));

    float shift[16];
    translateMatrix(shift, 0.f, 0.f, -3.0f);

    float frustum[16];
    frustumMatrix(frustum, float(width) / float(height), 1.f, 1.f, 8.f);

    float rx_center[16];
    matrixMul4(rx_center, rx, center);

    float ry_rx[16];
    matrixMul4(ry_rx, ry, rx_center);

    float rz_ry_rx[16];
    matrixMul4(rz_ry_rx, rz, ry_rx);

    float shift_rz_ry_rx[16];
    matrixMul4(shift_rz_ry_rx, shift, rz_ry_rx);

    float frustum_shift_rz_ry_rx[16];
    matrixMul4(frustum_shift_rz_ry_rx, frustum, shift_rz_ry_rx);

    glEnable(GL_CULL_FACE);
    glUseProgram(simplePrg);
    glBindVertexArray(vbo);
    glUniformMatrix4fv(0, 1, GL_FALSE, rz_ry_rx);
    glUniformMatrix4fv(1, 1, GL_FALSE, frustum_shift_rz_ry_rx);

    glDrawArrays(GL_TRIANGLES, 0, sizeof(vertexData)/(6*sizeof(float)));
    CHECK_GL;


    glfwSwapBuffers(win);
    glfwPollEvents();
  }
  glfwDestroyWindow(win);
  glfwTerminate();

  glDeleteShader(simpleVS);

  return EXIT_SUCCESS;
}
