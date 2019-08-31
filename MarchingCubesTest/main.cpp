#if 1
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

namespace {

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
    0.f, 0.f, 0.f,  0.f, 0.f, 1.f,
    1.f, 0.f, 0.f,  0.f, 0.f, 1.f,
    1.f, 1.f, 0.f,  0.f, 0.f, 1.f,
  };


  bool checkGL()
  {
    GLenum error = glGetError();
    if (error == GL_NO_ERROR) return true;
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
    } while (error != GL_NO_ERROR);
    return false;
  }

#define CHECK_GL do { bool ok = checkGL(); assert(ok); } while(0)

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


}


int main(int argc, char** argv)
{
  GLFWwindow* win;

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


  while (!glfwWindowShouldClose(win)) {

    GLfloat MV[] = {
      1.f, 0.f, 0.f, 0.f,
      0.f, 1.f, 0.f, 0.f,
      0.f, 0.f, 1.f, 0.f,
      0.f, 0.f, 0.f, 1.f,
    };

    GLfloat MVP[] = {
      1.f, 0.f, 0.f, 0.f,
      0.f, 1.f, 0.f, 0.f,
      0.f, 0.f, 1.f, 0.f,
      0.f, 0.f, 0.f, 1.f,
    };

    glUseProgram(simplePrg);
    glBindVertexArray(vbo);
    glUniformMatrix4fv(0, 1, GL_FALSE, MV);
    glUniformMatrix4fv(1, 1, GL_FALSE, MVP);

    glDrawArrays(GL_TRIANGLES, 0, 3);
    CHECK_GL;


    glfwSwapBuffers(win);
    glfwPollEvents();
  }
  glfwDestroyWindow(win);
  glfwTerminate();

  glDeleteShader(simpleVS);

  return EXIT_SUCCESS;
}
#else
int main(int argc, char** argv)
{
  return 0;
}
#endif
