#include <glad/gl.h>
#include <GLFW/glfw3.h>


#include <string>
#include <iostream>
#include <fstream>

namespace {

  void onGLFWError(int error, const char* what)
  {
    fprintf(stderr, "GLFW Error: %s\n", what);
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
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  win = glfwCreateWindow(1280, 720, "Marching cubes test application", nullptr, nullptr);
  glfwMakeContextCurrent(win);
  gladLoadGL(glfwGetProcAddress);

  while (!glfwWindowShouldClose(win)) {

    glfwSwapBuffers(win);
    glfwPollEvents();
  }
  glfwDestroyWindow(win);
  glfwTerminate();

  return EXIT_SUCCESS;
}