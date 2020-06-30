// This file is part of ComputeStuff copyright (C) 2020 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#define USE_NVTOOLS_EXT

#ifdef USE_NVTOOLS_EXT
#include <nvToolsExt.h> 
#endif
#include <cuda_runtime_api.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

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

  enum struct FieldFormat : uint32_t
  {
    UInt8,
    UInt16,
    Float
  };

  FieldFormat format = FieldFormat::Float;
  uint3 field_size = make_uint3(256, 256, 256);
  bool wireframe = false;
  bool recreate_context = true;
  bool indexed = true;

  enum LogLevels {
    ALWAYS = 0,
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    DEBUG = 4,
    TRACE = 5
  };
  uint32_t loglevel = 4;

#define LOG_ALWAYS(msg, ...) do { fputs("[A] ", stderr); fprintf(stderr, msg, ##__VA_ARGS__); fputc('\n', stderr); } while (0)
#define LOG_ERROR(msg, ...) do { if(ERROR <= loglevel) {  fputs("[E] ", stderr); fprintf(stderr, msg, ##__VA_ARGS__); fputc('\n', stderr);} } while (0)
#define LOG_WARNING(msg, ...) do { if(WARNING <= loglevel) {  fputs("[W] ", stderr); fprintf(stderr, msg, ##__VA_ARGS__); fputc('\n', stderr);} } while (0)
#define LOG_INFO(msg, ...) do { if(INFO <= loglevel) {  fputs("[I] ", stderr); fprintf(stderr, msg, ##__VA_ARGS__); fputc('\n', stderr);} } while (0)
#define LOG_DEBUG(msg, ...) do { if(DEBUG <= loglevel) {  fputs("[D] ", stderr); fprintf(stderr, msg, ##__VA_ARGS__); fputc('\n', stderr);} } while (0)
#define LOG_TRACE(msg, ...) do { if(TRACE <= loglevel) {  fputs("[T] ", stderr); fprintf(stderr, msg, ##__VA_ARGS__); fputc('\n', stderr);} } while (0)

  float threshold = 0.f;

  std::vector<char> scalarField_host;

  void onGLFWError(int error, const char* what)
  {
    LOG_ERROR("GLFW Error: %s", what);
  }

  void onKey(GLFWwindow* window, int key, int scancode, int action, int mods)
  {
    bool print_threshold = false;
    if (action == GLFW_PRESS) {
      if (key == GLFW_KEY_W) {
        wireframe = !wireframe;
        LOG_INFO("Wireframe: %s", wireframe ? "on" : "off");
      }
      else if (key == GLFW_KEY_UP) {
        threshold += 10.f; print_threshold = true;
      }
      else if (key == GLFW_KEY_DOWN) {
        threshold -= 10.f; print_threshold = true;
      }
      else if (key == GLFW_KEY_RIGHT) {
        threshold += 0.01f; print_threshold = true;
      }
      else if (key == GLFW_KEY_LEFT) {
        threshold -= 0.01f; print_threshold = true;
      }
      else if (key == GLFW_KEY_BACKSPACE) {
        threshold = 0.f; print_threshold = true;
      }
      else if (key == GLFW_KEY_I) {
        indexed = !indexed;
        recreate_context = true;
        LOG_INFO("Mode is %s", indexed ? "indexed" : "non-indexed");
      }
      if (print_threshold) {
        LOG_INFO("Iso-value: %f", threshold);
      }
    }
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
}
)";

  const std::string simpleFS_src = R"(#version 430
in vec3 normal;
out layout(location=0) vec4 outColor;
uniform layout(location=2) vec4 color;
void main() {
  float d = max(0.0, dot(vec3(0,0,1), normalize(gl_FrontFacing ? -normal : normal)));
  if(gl_FrontFacing)
    outColor = d * color.rgba;
  else
    outColor = color.bgra;
}
)";

  const std::string solidVS_src = R"(#version 430
in layout(location=0) vec3 inPosition;
uniform layout(location=0) mat4 MV;
uniform layout(location=1) mat4 MVP;
void main() {
  gl_Position = MVP * vec4(inPosition, 1);
}
)";

  const std::string solidFS_src = R"(#version 430
out layout(location=0) vec4 outColor;
uniform layout(location=2) vec4 color;
void main() {
  outColor = color.rgba;
}

)";


  [[noreturn]]
  void handleOpenGLError(GLenum error, const std::string file, int line)
  {
    do {
      switch (error) {
      case GL_INVALID_ENUM: LOG_ERROR("GL_INVALID_ENUM"); break;
      case GL_INVALID_VALUE: LOG_ERROR("GL_INVALID_VALUE"); break;
      case GL_INVALID_OPERATION: LOG_ERROR("GL_INVALID_OPERATION"); break;
      case GL_INVALID_FRAMEBUFFER_OPERATION: LOG_ERROR("GL_INVALID_FRAMEBUFFER_OPERATION"); break;
      case GL_OUT_OF_MEMORY: LOG_ERROR("GL_OUT_OF_MEMORY"); break;
      case GL_STACK_OVERFLOW: LOG_ERROR("GL_STACK_OVERFLOW"); break;
      case GL_STACK_UNDERFLOW: LOG_ERROR("GL_STACK_UNDERFLOW"); break;
      default: LOG_ERROR("Unknown error"); break;
      }
      error = glGetError();
    } while (error != GL_NO_ERROR);
    exit(EXIT_FAILURE);
  }

#define CHECK_GL do { GLenum error = glGetError(); if(error != GL_NO_ERROR) handleOpenGLError(error, __FILE__, __LINE__); } while(0)

  [[noreturn]]
  void handleCudaError(cudaError_t error, const std::string file, int line)
  {
    LOG_ERROR("%s@%d: CUDA: %s", file.c_str(), line, cudaGetErrorString(error));
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
      LOG_WARNING("Source:\n%s", src.c_str());
      std::vector<char> log(bufSize + 1);
      glGetShaderInfoLog(shader, bufSize + 1, nullptr, log.data());
      LOG_WARNING("Compilator output:\n%s", log.data());
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
      LOG_WARNING("Linker output:\n%s", log.data());
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
    auto a = 2.f * n / w;
    auto b = 2.f * n / h;
    auto c = -(f + n) / (f - n);
    auto d = -2.f * f * n / (f - n);
    dst[4 * 0 + 0] = a;   dst[4 * 0 + 1] = 0.f; dst[4 * 0 + 2] = 0.f; dst[4 * 0 + 3] = 0.f;
    dst[4 * 1 + 0] = 0.f; dst[4 * 1 + 1] = b;   dst[4 * 1 + 2] = 0.f; dst[4 * 1 + 3] = 0.f;
    dst[4 * 2 + 0] = 0.f; dst[4 * 2 + 1] = 0.f; dst[4 * 2 + 2] = c;   dst[4 * 2 + 3] = -1.f;
    dst[4 * 3 + 0] = 0.f; dst[4 * 3 + 1] = 0;   dst[4 * 3 + 2] = d;   dst[4 * 3 + 3] = 0.f;
  }


  void matrixMul4(float* D, const float* A, const float* B)
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

  void buildTransforms(float* normal_matrix,
                       float* modelview_projection,
                       const int width,
                       const int height,
                       double seconds)
  {
    float center[16];
    translateMatrix(center, -0.5f, -0.5f, -0.5f);

    float rx[16];
    rotMatrixX(rx, static_cast<float>(0.3 * seconds));

    float ry[16];
    rotMatrixY(ry, static_cast<float>(0.7 * seconds));

    float rz[16];
    rotMatrixZ(rz, static_cast<float>(0.5 * seconds));

    float shift[16];
    translateMatrix(shift, 0.f, 0.f, -2.0f);

    float frustum[16];
    frustumMatrix(frustum, float(width) / float(height), 1.f, 1.f, 8.f);

    float rx_center[16];
    matrixMul4(rx_center, rx, center);

    float ry_rx[16];
    matrixMul4(ry_rx, ry, rx_center);

    matrixMul4(normal_matrix, rz, ry_rx);

    float shift_rz_ry_rx[16];
    matrixMul4(shift_rz_ry_rx, shift, normal_matrix);

    matrixMul4(modelview_projection, frustum, shift_rz_ry_rx);
  }


  constexpr float cayley(unsigned i, unsigned j, unsigned k, uint3 field_size)
  {
    float x = (2.f * i) / (field_size.x - 1.f) - 1.f;
    float y = (2.f * j) / (field_size.y - 1.f) - 1.f;
    float z = (2.f * k) / (field_size.z - 1.f) - 1.f;
    float v = 1.f - 16.f * x * y * z - 4.f * (x * x + y * y + z * z);
    return v;
  }

  GLfloat wireBoxVertexData[] =
  {
    0.f, 0.f, 0.f,  1.f, 0.f, 0.f,
    0.f, 0.f, 1.f,  1.f, 0.f, 1.f,
    0.f, 1.f, 0.f,  1.f, 1.f, 0.f,
    0.f, 1.f, 1.f,  1.f, 1.f, 1.f,

    0.f, 0.f, 0.f,  0.f, 1.f, 0.f,
    0.f, 0.f, 1.f,  0.f, 1.f, 1.f,
    1.f, 0.f, 0.f,  1.f, 1.f, 0.f,
    1.f, 0.f, 1.f,  1.f, 1.f, 1.f,

    0.f, 0.f, 0.f,  0.f, 0.f, 1.f,
    0.f, 1.f, 0.f,  0.f, 1.f, 1.f,
    1.f, 0.f, 0.f,  1.f, 0.f, 1.f,
    1.f, 1.f, 0.f,  1.f, 1.f, 1.f
  };

  void buildCayleyField()
  {
    const size_t N = static_cast<size_t>(field_size.x) * field_size.y * field_size.z;
    switch (format) {
    case FieldFormat::UInt8: {
      scalarField_host.resize(N);
      auto* dst = reinterpret_cast<uint8_t*>(scalarField_host.data());
      for (unsigned k = 0; k < field_size.z; k++) {
        for (unsigned j = 0; j < field_size.y; j++) {
          for (unsigned i = 0; i < field_size.x; i++) {
            float v = cayley(i, j, k, field_size);
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
      for (unsigned k = 0; k < field_size.z; k++) {
        for (unsigned j = 0; j < field_size.y; j++) {
          for (unsigned i = 0; i < field_size.x; i++) {
            float v = cayley(i, j, k, field_size);
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
      for (unsigned k = 0; k < field_size.z; k++) {
        for (unsigned j = 0; j < field_size.y; j++) {
          for (unsigned i = 0; i < field_size.x; i++) {
            *dst++ = cayley(i, j, k, field_size);
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
    LOG_INFO("Reading %s...", path);

    FILE* fp = fopen(path, "rb");
    if (!fp) {
      LOG_ERROR("Error opening file \"%s\" for reading.", path);
      return false;
    }
    if (fseek(fp, 0L, SEEK_END) == 0) {
      uint8_t header[6];
      long size = ftell(fp);
      if (sizeof(header) <= size) {
        if (fseek(fp, 0L, SEEK_SET) == 0) {
          if (fread(header, sizeof(header), 1, fp) == 1) {
            field_size.x = header[0] | header[1] << 8;
            field_size.y = header[2] | header[3] << 8;
            field_size.z = header[4] | header[5] << 8;
            size_t N = static_cast<size_t>(field_size.x) * field_size.y * field_size.z;
            if ((N + 3) * 2 != size) {
              LOG_ERROR("Unexpected file size.");
            }
            else {
              std::vector<uint8_t> tmp(2 * N);
              if (fread(tmp.data(), 2, N, fp) == N) {
                switch (format) {
                case FieldFormat::UInt8: {
                  scalarField_host.resize(N);
                  auto* dst = reinterpret_cast<uint8_t*>(scalarField_host.data());
                  for (size_t i = 0; i < N; i++) {
                    const uint32_t v = tmp[2 * i + 0] | tmp[2 * i + 1] << 8;
                    dst[i] = v >> 4; // 12 bits are in use.
                  }
                  break;
                }
                case FieldFormat::UInt16: {
                  scalarField_host.resize(sizeof(uint16_t) * N);
                  auto* dst = reinterpret_cast<uint16_t*>(scalarField_host.data());
                  for (size_t i = 0; i < N; i++) {
                    const uint32_t v = tmp[2 * i + 0] | tmp[2 * i + 1] << 8;
                    dst[i] = v;
                  }
                  break;
                }
                case FieldFormat::Float: {
                  scalarField_host.resize(sizeof(float) * N);
                  auto* dst = reinterpret_cast<float*>(scalarField_host.data());
                  for (size_t i = 0; i < N; i++) {
                    const uint32_t v = tmp[2 * i + 0] | tmp[2 * i + 1] << 8;
                    dst[i] = static_cast<float>(v);
                  }
                  break;
                }
                default:
                  assert(false && "Unhandled case");
                }
                LOG_INFO("Successfully loaded %s", path);
                fclose(fp);
                return true;
              }
            }
          }
        }
      }
    }
    LOG_ERROR("Error loading \"%s\"", path);
    fclose(fp);
    return false;
  }

  void setupScalarField(float*& scalar_field_d, const char* path, const uint3& field_size, cudaStream_t stream)
  {
    // Set up scalar field
    if (!path) {
      LOG_ERROR("No input file specified.");
      exit(EXIT_FAILURE);
    }
    else if (strcmp("cayley", path) == 0) {
      buildCayleyField();
    }
    else if (!readFile(path)) {
      exit(EXIT_FAILURE);
    }
    assert(static_cast<size_t>(field_size.x) * field_size.y * field_size.z * 4 == scalarField_host.size());
    LOG_INFO("Scalar field is [%d x %d x %d] (%d cells total)", field_size.x, field_size.y, field_size.z, field_size.x * field_size.y * field_size.z);
    CHECKED_CUDA(cudaMalloc(&scalar_field_d, scalarField_host.size()));
    CHECKED_CUDA(cudaMemcpyAsync(scalar_field_d, scalarField_host.data(), scalarField_host.size(), cudaMemcpyHostToDevice, stream));
  }

  void initWindowAndGL(GLFWwindow*& win, GLuint& shadedProg, GLuint& solidProg)
  {
    glfwSetErrorCallback(onGLFWError);
    if (!glfwInit()) {
      LOG_ERROR("GLFW failed to initialize.");
      exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    win = glfwCreateWindow(1280, 720, "Marching cubes test application", nullptr, nullptr);
    glfwSetKeyCallback(win, onKey);
    glfwMakeContextCurrent(win);
    gladLoadGL(glfwGetProcAddress);

    GLuint simpleVS = createShader(simpleVS_src, GL_VERTEX_SHADER);   assert(simpleVS != 0);
    GLuint simpleFS = createShader(simpleFS_src, GL_FRAGMENT_SHADER); assert(simpleFS != 0);
    shadedProg = createProgram(simpleVS, simpleFS);                   assert(shadedProg != 0);

    GLuint solidVS = createShader(solidVS_src, GL_VERTEX_SHADER);     assert(solidVS != 0);
    GLuint solidFS = createShader(solidFS_src, GL_FRAGMENT_SHADER);   assert(solidFS != 0);
    solidProg = createProgram(solidVS, solidFS);                      assert(solidProg != 0);
  }

}



int main(int argc, char** argv)
{
  cudaStream_t stream;
  const char* path = nullptr;
  int deviceIndex = 0;
  bool benchmark = false;

  for (int i = 1; i < argc; i++) {
    if (i + 1 < argc && (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0)) { deviceIndex = std::atoi(argv[i + 1]); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-nx") == 0) { field_size.x = uint32_t(std::atoi(argv[i + 1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-ny") == 0) { field_size.y = uint32_t(std::atoi(argv[i + 1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-nz") == 0) { field_size.z = uint32_t(std::atoi(argv[i + 1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-n") == 0) { field_size.x = uint32_t(std::atoi(argv[i + 1])); field_size.y = field_size.x; field_size.z = field_size.x; i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-i") == 0) { threshold = static_cast<float>(std::atof(argv[i + 1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-l") == 0) { loglevel = uint32_t(std::atoi(argv[i + 1])); i++; }
#if 0
    // Currently only float is supported
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
#endif
    else if ((strcmp(argv[i], "-b") == 0) || (strcmp(argv[i], "--benchmark") == 0)) { benchmark = true; }
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
      fprintf(stderr, "HP5 Marching Cubes test application.\n");
      fprintf(stderr, "Copyright (C) 2020 Christopher Dyken. Released under the MIT license\n\n");
      fprintf(stderr, "Usage: %s [options] [dataset]\n\n", argv[0]);
      fprintf(stderr, "Options:\n");
      fprintf(stderr, "    -d   int    Choose CUDA device.\n");
      fprintf(stderr, "    -nx  int    Set number of samples in x direction.\n");
      fprintf(stderr, "    -nx  int    Set number of samples in y direction.\n");
      fprintf(stderr, "    -nx  int    Set number of samples in z direction.\n");
      fprintf(stderr, "    -n   int    Set uniform number of samples in x,y,z directions.\n");
      fprintf(stderr, "    -i   float  Set iso-value to extract surface for.\n");
      fprintf(stderr, "    -l   int    Log-level, higher is more verbose.\n");
      fprintf(stderr, "    -b          Enable benchmark mode without OpenGL interop.\n");
      fprintf(stderr, "\nDataset:\n");
      fprintf(stderr, "    cayley    Built-in algebraic surface.\n");
      fprintf(stderr, "    file.dat  Raw binary uint16_t data with three binary uint16_t in front with x,y,z size.\n");
      fprintf(stderr, "\nKey bindings:\n");
      fprintf(stderr, "    right/left  Increase/decrease threshold by 100.\n");
      fprintf(stderr, "    up/down     Increase/decrease threshold by 0.1.\n");
      fprintf(stderr, "    w           Enable/disable wireframe.\n");
      return 0;
    }
    else {
      if (path) {
        LOG_ERROR("%s: input already specified", argv[i]);
        return EXIT_FAILURE;
      }
      path = argv[i];
    }
  }

  if (benchmark) {
    
    int deviceCount = 0;
    CHECKED_CUDA(cudaGetDeviceCount(&deviceCount));

    bool found = false;
    for (int i = 0; i < deviceCount; i++) {
      cudaDeviceProp dev_prop;
      cudaGetDeviceProperties(&dev_prop, i);
      LOG_INFO("%c[%i] %s cap=%d.%d", i == deviceIndex ? '*' : ' ', i, dev_prop.name, dev_prop.major, dev_prop.minor);
      if (i == deviceIndex) {
        found = true;
      }
    }
    if (!found) {
      LOG_ERROR("Illegal CUDA device index %d", deviceIndex);
      return EXIT_FAILURE;
    }
    cudaSetDevice(deviceIndex);
    CHECKED_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Create events for timing
    static const unsigned eventNum = 32;
    cudaEvent_t events[2 * eventNum];
    for (size_t i = 0; i < 2 * eventNum; i++) {
      CHECKED_CUDA(cudaEventCreate(&events[i]));
      CHECKED_CUDA(cudaEventRecord(events[i], stream));
    }

    size_t free, total;
    CHECKED_CUDA(cudaMemGetInfo(&free, &total));
    LOG_INFO("CUDA memory free=%zumb total=%zumb", (free + 1024 * 1024 - 1) / (1024 * 1024), (total + 1024 * 1024 - 1) / (1024 * 1024));

    float* scalar_field_d = nullptr;
    setupScalarField(scalar_field_d, path, field_size, stream);
    LOG_INFO("Built scalar field");

    CHECKED_CUDA(cudaMemGetInfo(&free, &total));
    LOG_INFO("CUDA memory free=%zumb total=%zumb", (free + 1024 * 1024 - 1) / (1024 * 1024), (total + 1024 * 1024 - 1) / (1024 * 1024));

    auto* tables = createTables(stream);

    struct {
      const char* name;
      bool indexed;
      bool sync;
    }
    benchmark_cases[] = {
      {"ix sync", true, true},
      {"noix sync", false, true},
      {"ix nosync", true, false},
      {"noix nosync", false, false}
    };

    float min_time = 0.5;
    for (auto& bc : benchmark_cases) {
#ifdef USE_NVTOOLS_EXT
      nvtxRangePush(bc.name);
#endif
      auto* ctx = createContext(tables, field_size, true, stream);
      LOG_INFO("%12s: Created context.", bc.name);

      // Run with no output buffers to get size of output.
      ComputeStuff::MC::buildPN(ctx,
                                nullptr,
                                nullptr,
                                0,
                                0,
                                field_size.x,
                                field_size.x* field_size.y,
                                make_uint3(0, 0, 0),
                                field_size,
                                scalar_field_d,
                                threshold,
                                stream,
                                true,
                                true);
      uint32_t vertex_count = 0;
      uint32_t index_count = 0;
      ComputeStuff::MC::getCounts(ctx, &vertex_count, &index_count, stream);

      float* vertex_data_d = nullptr;
      CHECKED_CUDA(cudaMalloc(&vertex_data_d, 6 * sizeof(float) * vertex_count));
      uint32_t* index_data_d = nullptr;
      CHECKED_CUDA(cudaMalloc(&index_data_d, sizeof(uint32_t)* index_count));
      LOG_INFO("%12s: Allocated output buffers.", bc.name);

      LOG_INFO("%12s: Warming up", bc.name);
      for (unsigned i = 0; i < 100; i++) {
        ComputeStuff::MC::buildPN(ctx,
                                  vertex_data_d,
                                  index_data_d,
                                  6 * sizeof(float) * vertex_count,
                                  sizeof(uint32_t) * index_count,
                                  field_size.x,
                                  field_size.x * field_size.y,
                                  make_uint3(0, 0, 0),
                                  field_size,
                                  scalar_field_d,
                                  threshold,
                                  stream,
                                  true,
                                  true);
        if (bc.sync) {
          ComputeStuff::MC::getCounts(ctx, &vertex_count, &index_count, stream);
        }
      }

      LOG_INFO("%12s: Benchmarking", bc.name);
      auto start = std::chrono::high_resolution_clock::now();
      double elapsed = 0.f;
      float cuda_ms = 0.f;
      unsigned iterations = 0;
      unsigned cuda_ms_n = 0;
#ifdef USE_NVTOOLS_EXT
      nvtxRangePush("Benchmark runs");
#endif
      while (iterations < 100 || elapsed < min_time) {
        CHECKED_CUDA(cudaEventRecord(events[2 * (iterations % eventNum) + 0], stream));
        ComputeStuff::MC::buildPN(ctx,
                                  vertex_data_d,
                                  index_data_d,
                                  6 * sizeof(float) * vertex_count,
                                  sizeof(uint32_t) * index_count,
                                  field_size.x,
                                  field_size.x * field_size.y,
                                  make_uint3(0, 0, 0),
                                  field_size,
                                  scalar_field_d,
                                  threshold,
                                  stream,
                                  true,
                                  true);
        if (bc.sync) {
          ComputeStuff::MC::getCounts(ctx, &vertex_count, &index_count, stream);
        }
        CHECKED_CUDA(cudaEventRecord(events[2 * (iterations % eventNum) + 1], stream));

        if (eventNum <= iterations) {
          float ms = 0;
          if (!bc.sync) {
            CHECKED_CUDA(cudaEventSynchronize(events[2 * ((iterations + 1) % eventNum) + 1]));
          }

          CHECKED_CUDA(cudaEventElapsedTime(&ms,
                                            events[2 * ((iterations + 1) % eventNum) + 0],
                                            events[2 * ((iterations + 1) % eventNum) + 1]));
          cuda_ms += ms;
          cuda_ms_n++;
        }

        std::chrono::duration<double> span = std::chrono::high_resolution_clock::now() - start;
        elapsed = span.count();
        iterations++;
      }
#ifdef USE_NVTOOLS_EXT
      nvtxRangePop();
#endif
      CHECKED_CUDA(cudaMemGetInfo(&free, &total));
      LOG_ALWAYS("%12s: %.2f FPS (%.0fMVPS) cuda: %.2fms (%.0f MVPS) %ux%ux%u Nv=%u Ni=%u memfree=%zumb/%zumb",
              bc.name,
              iterations / elapsed, (float(iterations) * field_size.x * field_size.y * field_size.z) / (1000000.f * elapsed),
              cuda_ms / cuda_ms_n, (float(cuda_ms_n) * field_size.x * field_size.y * field_size.z) / (1000.f * cuda_ms),
              field_size.x, field_size.y, field_size.z,
              vertex_count,
              index_count,
              (free + 1024 * 1024 - 1) / (1024 * 1024),
              (total + 1024 * 1024 - 1) / (1024 * 1024));

      freeContext(ctx, stream);
      CHECKED_CUDA(cudaStreamSynchronize(stream));
      CHECKED_CUDA(cudaFree(vertex_data_d));
      CHECKED_CUDA(cudaFree(index_data_d));

      CHECKED_CUDA(cudaMemGetInfo(&free, &total));
      LOG_INFO("%12s: Released resources free=%zumb total=%zumb", bc.name, (free + 1024 * 1024 - 1) / (1024 * 1024), (total + 1024 * 1024 - 1) / (1024 * 1024));
#ifdef USE_NVTOOLS_EXT
      nvtxRangePop();
#endif
    }

    LOG_ALWAYS("Exiting...");
    CHECKED_CUDA(cudaMemGetInfo(&free, &total));
    LOG_INFO("CUDA memory free=%zumb total=%zumb", (free + 1024 * 1024 - 1) / (1024 * 1024), (total + 1024 * 1024 - 1) / (1024 * 1024));
    return 0;
  }

  GLFWwindow* win = nullptr;
  GLuint shadedProg = 0;
  GLuint solidProg = 0;
  initWindowAndGL(win, shadedProg, solidProg);

  unsigned int deviceCount;
  CHECKED_CUDA(cudaGLGetDevices(&deviceCount, nullptr, 0, cudaGLDeviceListAll));
  if (deviceCount == 0) {
    LOG_ERROR("No CUDA-enabled devices available.");
    return EXIT_FAILURE;
  }
  std::vector<int> devices(deviceCount);
  CHECKED_CUDA(cudaGLGetDevices(&deviceCount, devices.data(), deviceCount, cudaGLDeviceListAll));

  bool found = false;
  for (unsigned k = 0; k < deviceCount; k++) {
    int i = devices[k];
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, i);
    LOG_INFO("%c[%i] %s cap=%d.%d", i == deviceIndex ? '*' : ' ', i, dev_prop.name, dev_prop.major, dev_prop.minor);
    if (i == deviceIndex) {
      found = true;
    }
  }
  if (!found) {
    LOG_ERROR("Illegal CUDA device index %d", deviceIndex);
    return EXIT_FAILURE;
  }
  cudaSetDevice(deviceIndex);
  CHECKED_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Set up scalar field
  float* scalar_field_d = nullptr;
  setupScalarField(scalar_field_d, path, field_size, stream);

  auto* tables = createTables(stream);

  GLuint wireBoxVertexBuffer = createBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(wireBoxVertexData),  wireBoxVertexData);
  uint32_t wireBoxVertexCount = sizeof(wireBoxVertexData) / (3 * sizeof(float));
  GLuint wireBoxVbo = 0;
  glGenVertexArrays(1, &wireBoxVbo);
  glBindVertexArray(wireBoxVbo);
  glBindBuffer(GL_ARRAY_BUFFER, wireBoxVertexBuffer);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr);
  glEnableVertexAttribArray(0);

  unsigned eventCounter = 0;
  cudaEvent_t events[2 * 4];
  for (size_t i = 0; i < 2 * 4; i++) {
    CHECKED_CUDA(cudaEventCreate(&events[i]));
    CHECKED_CUDA(cudaEventRecord(events[i], stream));
  }

  GLuint cudaVertexBuf = createBuffer(GL_ARRAY_BUFFER, GL_STREAM_DRAW, 3 * sizeof(float), nullptr);
  cudaGraphicsResource* vertexBufferResource = nullptr;
  CHECKED_CUDA(cudaGraphicsGLRegisterBuffer(&vertexBufferResource, cudaVertexBuf, cudaGraphicsRegisterFlagsWriteDiscard));

  GLuint cudaIndexBuf = createBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_STREAM_DRAW, 3 * sizeof(uint32_t), nullptr);
  cudaGraphicsResource* indexBufferResource = nullptr;
  CHECKED_CUDA(cudaGraphicsGLRegisterBuffer(&indexBufferResource, cudaIndexBuf, cudaGraphicsRegisterFlagsWriteDiscard));

  GLuint cudaVbo = 0;
  glGenVertexArrays(1, &cudaVbo);
  glBindVertexArray(cudaVbo);
  glBindBuffer(GL_ARRAY_BUFFER, cudaVertexBuf);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, nullptr);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void*)(sizeof(float) * 3));
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);


  auto start = std::chrono::system_clock::now();
  auto timer = std::chrono::high_resolution_clock::now();
  float cuda_ms = 0.f;
  unsigned frames = 0u;

  ComputeStuff::MC::Context* ctx = nullptr;
  while (!glfwWindowShouldClose(win)) {
    int width, height;
    glfwGetWindowSize(win, &width, &height);

    uint32_t vertex_count = 0;
    uint32_t index_count = 0;
    {
      if (ctx == nullptr || recreate_context) {
        freeContext(ctx, stream);
        ctx = createContext(tables, field_size, indexed, stream);
        recreate_context = false;
      }

      float* cudaVertexBuf_d = nullptr;
      size_t cudaVertexBuf_size = 0;

      uint32_t* cudaIndexBuf_d = nullptr;
      size_t cudaIndexBuf_size = 0;

      CHECKED_CUDA(cudaGraphicsMapResources(1, &vertexBufferResource, stream));
      CHECKED_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&cudaVertexBuf_d, &cudaVertexBuf_size, vertexBufferResource));
      if (indexed) {
        CHECKED_CUDA(cudaGraphicsMapResources(1, &indexBufferResource, stream));
        CHECKED_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&cudaIndexBuf_d, &cudaIndexBuf_size, indexBufferResource));
      }
      CHECKED_CUDA(cudaEventRecord(events[2 * eventCounter + 0], stream));
      ComputeStuff::MC::buildPN(ctx,
                                cudaVertexBuf_d,
                                cudaIndexBuf_d,
                                cudaVertexBuf_size,
                                cudaIndexBuf_size,
                                field_size.x,
                                field_size.x* field_size.y,
                                make_uint3(0, 0, 0),
                                field_size,
                                scalar_field_d,
                                threshold,
                                stream,
                                true,
                                true);
      CHECKED_CUDA(cudaEventRecord(events[2 * eventCounter + 1], stream));
      CHECKED_CUDA(cudaGraphicsUnmapResources(1, &vertexBufferResource, stream));
      if (indexed) {
        CHECKED_CUDA(cudaGraphicsUnmapResources(1, &indexBufferResource, stream));
      }

      ComputeStuff::MC::getCounts(ctx, &vertex_count, &index_count, stream);
      
      eventCounter = (eventCounter + 1) & 3;
      float ms = 0;
      CHECKED_CUDA(cudaEventElapsedTime(&ms, events[2 * eventCounter + 0], events[2 * eventCounter + 1]));
      cuda_ms += ms;

      bool vertexBufTooSmall = cudaVertexBuf_size < 6 * sizeof(float) * vertex_count;
      bool indexBufTooSmall = cudaIndexBuf_size < sizeof(uint32_t)* index_count;

      if (vertexBufTooSmall || indexBufTooSmall) {

        CHECKED_CUDA(cudaGraphicsUnregisterResource(vertexBufferResource));
        CHECKED_CUDA(cudaGraphicsUnregisterResource(indexBufferResource));

        if (vertexBufTooSmall) {
          size_t newVertexBufSize = 6 * sizeof(float) * (static_cast<size_t>(vertex_count) + vertex_count / 16);
          glBindBuffer(GL_ARRAY_BUFFER, cudaVertexBuf);
          glBufferData(GL_ARRAY_BUFFER, newVertexBufSize, nullptr, GL_STREAM_DRAW);
          glBindBuffer(GL_ARRAY_BUFFER, 0);
          LOG_INFO("Resizing: vbuf=%zub", newVertexBufSize);
        }

        if (indexBufTooSmall) {
          size_t newIndexBufSize = sizeof(uint32_t) * (index_count + index_count / 16);
          glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cudaIndexBuf);
          glBufferData(GL_ELEMENT_ARRAY_BUFFER, newIndexBufSize, nullptr, GL_STREAM_DRAW);
          glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
          LOG_INFO("Resizing: ibuf=%zub", newIndexBufSize);
        }

        CHECKED_CUDA(cudaGraphicsGLRegisterBuffer(&vertexBufferResource, cudaVertexBuf, cudaGraphicsRegisterFlagsWriteDiscard));
        CHECKED_CUDA(cudaGraphicsGLRegisterBuffer(&indexBufferResource, cudaIndexBuf, cudaGraphicsRegisterFlagsWriteDiscard));

        CHECKED_CUDA(cudaGraphicsMapResources(1, &vertexBufferResource, stream));
        CHECKED_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&cudaVertexBuf_d, &cudaVertexBuf_size, vertexBufferResource));
        if (indexed) {
          CHECKED_CUDA(cudaGraphicsMapResources(1, &indexBufferResource, stream));
          CHECKED_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&cudaIndexBuf_d, &cudaIndexBuf_size, indexBufferResource));
        }
        ComputeStuff::MC::buildPN(ctx,
                                  cudaVertexBuf_d,
                                  cudaIndexBuf_d,
                                  cudaVertexBuf_size,
                                  cudaIndexBuf_size,
                                  field_size.x,
                                  field_size.x* field_size.y,
                                  make_uint3(0, 0, 0),
                                  field_size,
                                  scalar_field_d,
                                  threshold,
                                  stream,
                                  false,
                                  indexed);
        CHECKED_CUDA(cudaGraphicsUnmapResources(1, &vertexBufferResource, stream));
        if (indexed) {
          CHECKED_CUDA(cudaGraphicsUnmapResources(1, &indexBufferResource, stream));
        }
      }
    }
    glViewport(0, 0, width, height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;

    float normal_matrix[16];
    float modelview_projection[16];
    buildTransforms(normal_matrix, modelview_projection, width, height, elapsed.count());


    glEnable(GL_DEPTH_TEST);
    glPolygonOffset(0.f, 1.f);
    if (wireframe) {
      glEnable(GL_POLYGON_OFFSET_FILL);
    }
    glBindVertexArray(cudaVbo);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glUseProgram(shadedProg);
    glUniformMatrix4fv(0, 1, GL_FALSE, normal_matrix);
    glUniformMatrix4fv(1, 1, GL_FALSE, modelview_projection);
    glUniform4f(2, 0.6f, 0.6f, 0.8f, 1.f);
    if (indexed) {
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cudaIndexBuf);
      glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, nullptr);
    }
    else {
      glDrawArrays(GL_TRIANGLES, 0, vertex_count);
    }
    glDisable(GL_POLYGON_OFFSET_FILL);


    if (wireframe) {
      glUseProgram(solidProg);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glUniformMatrix4fv(0, 1, GL_FALSE, normal_matrix);
      glUniformMatrix4fv(1, 1, GL_FALSE, modelview_projection);
      glUniform4f(2, 1.f, 1.f, 1.f, 1.f);
      if (indexed) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cudaIndexBuf);
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, nullptr);
      }
      else {
        glDrawArrays(GL_TRIANGLES, 0, vertex_count);
      }
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(wireBoxVbo);
    glUseProgram(solidProg);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glUniformMatrix4fv(0, 1, GL_FALSE, normal_matrix);
    glUniformMatrix4fv(1, 1, GL_FALSE, modelview_projection);
    glUniform4f(2, 1.f, 1.f, 1.f, 1.f);
    glDrawArrays(GL_LINES, 0, wireBoxVertexCount);

    glfwSwapBuffers(win);
    glfwPollEvents();

    {
      frames++;
      auto now = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = now - timer;
      auto s = elapsed.count();
      if (10 < frames && 3.0 < s) {
        size_t free, total;
        CHECKED_CUDA(cudaMemGetInfo(&free, &total));
        LOG_INFO("%.2f FPS (%.2f MVPS) cuda avg: %.2fms (%.2f MVPS) %ux%ux%u Nv=%u Ni=%u ix=%s memfree=%zumb/%zumb",
                 frames / s, (float(frames)* field_size.x* field_size.y* field_size.z) / (1000000.f * s),
                 cuda_ms / frames, (float(frames)* field_size.x* field_size.y* field_size.z) / (1000.f * cuda_ms),
                 field_size.x, field_size.y, field_size.z,
                 vertex_count,
                 index_count,
                 indexed ? "y" : "n",
                 (free + 1024 * 1024 - 1) / (1024 * 1024),
                 (total + 1024 * 1024 - 1) / (1024 * 1024));
        timer = now;
        frames = 0;
        cuda_ms = 0.f;
      }
    }


  }
  glfwDestroyWindow(win);
  glfwTerminate();

  return EXIT_SUCCESS;
}
