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

  // Marching cubes tables.
  //
  // Original table from 'Marching Cubes Example Program` by Cory Bloyd,
  // released to the Public Domain.
  // http://paulbourke.net/geometry/polygonise/marchingsource.cpp
  //
  // Cell corners reorganized such that
  //
  //       6--------7
  //      /|       /|
  //     / |      / |
  //    4--------5  |
  //    |  2-----|--3     z  y
  //    | /      | /      ^ /
  //    |/       |/       |/
  //    0--------1        --->x
  //
  // Triangle indices are encoded such that bits 3, 4 and 5 tells the shift
  // along X, Y and Z respectively for the first end of the edge, and from
  // that, bits 0, 1, and 2 tells which of the X, Y and Z axes to follow, i.e.,
  // only one of the bits 0, 1, and 2 is set. Another invariant is that a
  // valid edge index is never zero.
  //
  const uint8_t index_count[256] = {
    0, 3, 3, 6, 3, 6, 6, 9, 3, 6, 6, 9, 6, 9, 9, 6,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    6, 9, 9, 6, 9, 12, 12, 9, 9, 12, 12, 9, 12, 15, 15, 6,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    6, 9, 9, 12, 9, 6, 12, 9, 9, 12, 12, 15, 12, 9, 15, 6,
    6, 9, 9, 12, 9, 12, 12, 15, 9, 12, 12, 15, 12, 15, 15, 12,
    9, 12, 12, 9, 12, 9, 15, 6, 12, 15, 15, 12, 15, 12, 6, 3,
    3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
    6, 9, 9, 12, 9, 12, 12, 15, 9, 12, 12, 15, 12, 15, 15, 12,
    6, 9, 9, 12, 9, 12, 12, 15, 9, 12, 6, 9, 12, 15, 9, 6,
    9, 12, 12, 9, 12, 15, 15, 12, 12, 15, 9, 6, 15, 6, 12, 3,
    6, 9, 9, 12, 9, 12, 12, 15, 9, 12, 12, 15, 6, 9, 9, 6,
    9, 12, 12, 15, 12, 9, 15, 12, 12, 15, 15, 6, 9, 6, 12, 3,
    9, 12, 12, 15, 12, 15, 15, 6, 12, 15, 9, 12, 9, 12, 6, 3,
    6, 9, 9, 6, 9, 6, 12, 3, 9, 12, 6, 3, 6, 3, 3, 0
  };

  const uint8_t mc_triangles[256 * 16] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    001, 004, 002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 001, 012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 004, 002, 012, 014, 002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    021, 002, 024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    021, 001, 004, 024, 021, 004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    001, 012, 014, 002, 024, 021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    021, 012, 014, 021, 014, 024, 024, 014, 004, 0, 0, 0, 0, 0, 0, 0,
    034, 012, 021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    012, 021, 034, 001, 004, 002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    034, 014, 001, 021, 034, 001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    002, 021, 034, 002, 034, 004, 004, 034, 014, 0, 0, 0, 0, 0, 0, 0,
    024, 034, 012, 002, 024, 012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    012, 001, 004, 012, 004, 034, 034, 004, 024, 0, 0, 0, 0, 0, 0, 0,
    001, 002, 024, 001, 024, 014, 014, 024, 034, 0, 0, 0, 0, 0, 0, 0,
    014, 004, 034, 034, 004, 024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    041, 042, 004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    001, 041, 042, 002, 001, 042, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 001, 012, 041, 042, 004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 041, 042, 014, 042, 012, 012, 042, 002, 0, 0, 0, 0, 0, 0, 0,
    042, 004, 041, 024, 021, 002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    042, 024, 021, 042, 021, 041, 041, 021, 001, 0, 0, 0, 0, 0, 0, 0,
    041, 042, 004, 001, 012, 014, 024, 021, 002, 0, 0, 0, 0, 0, 0, 0,
    042, 024, 021, 041, 042, 021, 041, 021, 012, 041, 012, 014, 0, 0, 0, 0,
    041, 042, 004, 012, 021, 034, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    001, 041, 042, 001, 042, 002, 021, 034, 012, 0, 0, 0, 0, 0, 0, 0,
    001, 021, 034, 001, 034, 014, 041, 042, 004, 0, 0, 0, 0, 0, 0, 0,
    041, 042, 002, 041, 002, 034, 041, 034, 014, 021, 034, 002, 0, 0, 0, 0,
    024, 034, 012, 024, 012, 002, 004, 041, 042, 0, 0, 0, 0, 0, 0, 0,
    012, 001, 041, 012, 041, 024, 012, 024, 034, 024, 041, 042, 0, 0, 0, 0,
    041, 042, 004, 001, 002, 014, 014, 002, 024, 014, 024, 034, 0, 0, 0, 0,
    042, 014, 041, 042, 024, 014, 024, 034, 014, 0, 0, 0, 0, 0, 0, 0,
    014, 052, 041, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    041, 014, 052, 004, 002, 001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    041, 001, 012, 052, 041, 012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    041, 004, 002, 041, 002, 052, 052, 002, 012, 0, 0, 0, 0, 0, 0, 0,
    014, 052, 041, 002, 024, 021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    004, 024, 021, 004, 021, 001, 014, 052, 041, 0, 0, 0, 0, 0, 0, 0,
    012, 052, 041, 012, 041, 001, 002, 024, 021, 0, 0, 0, 0, 0, 0, 0,
    041, 004, 024, 041, 024, 012, 041, 012, 052, 012, 024, 021, 0, 0, 0, 0,
    014, 052, 041, 012, 021, 034, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    052, 041, 014, 012, 021, 034, 004, 002, 001, 0, 0, 0, 0, 0, 0, 0,
    034, 052, 041, 034, 041, 021, 021, 041, 001, 0, 0, 0, 0, 0, 0, 0,
    034, 002, 021, 052, 002, 034, 052, 004, 002, 052, 041, 004, 0, 0, 0, 0,
    012, 002, 024, 012, 024, 034, 052, 041, 014, 0, 0, 0, 0, 0, 0, 0,
    041, 014, 052, 004, 034, 001, 004, 024, 034, 034, 012, 001, 0, 0, 0, 0,
    052, 024, 034, 052, 001, 024, 052, 041, 001, 002, 024, 001, 0, 0, 0, 0,
    041, 034, 052, 041, 004, 034, 004, 024, 034, 0, 0, 0, 0, 0, 0, 0,
    052, 042, 004, 014, 052, 004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    001, 014, 052, 001, 052, 002, 002, 052, 042, 0, 0, 0, 0, 0, 0, 0,
    004, 001, 012, 004, 012, 042, 042, 012, 052, 0, 0, 0, 0, 0, 0, 0,
    052, 042, 012, 042, 002, 012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    004, 014, 052, 004, 052, 042, 024, 021, 002, 0, 0, 0, 0, 0, 0, 0,
    014, 021, 001, 014, 042, 021, 014, 052, 042, 024, 021, 042, 0, 0, 0, 0,
    024, 021, 002, 004, 001, 042, 042, 001, 012, 042, 012, 052, 0, 0, 0, 0,
    021, 042, 024, 021, 012, 042, 012, 052, 042, 0, 0, 0, 0, 0, 0, 0,
    052, 042, 004, 052, 004, 014, 012, 021, 034, 0, 0, 0, 0, 0, 0, 0,
    034, 012, 021, 052, 002, 014, 052, 042, 002, 002, 001, 014, 0, 0, 0, 0,
    004, 052, 042, 004, 021, 052, 004, 001, 021, 021, 034, 052, 0, 0, 0, 0,
    034, 002, 021, 034, 052, 002, 052, 042, 002, 0, 0, 0, 0, 0, 0, 0,
    034, 012, 002, 034, 002, 024, 014, 052, 004, 004, 052, 042, 0, 0, 0, 0,
    034, 001, 024, 034, 012, 001, 024, 001, 042, 014, 052, 001, 042, 001, 052, 0,
    042, 001, 052, 042, 004, 001, 052, 001, 034, 002, 024, 001, 034, 001, 024, 0,
    042, 034, 052, 024, 034, 042, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    042, 061, 024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    004, 002, 001, 042, 061, 024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 001, 012, 061, 024, 042, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    002, 012, 014, 002, 014, 004, 042, 061, 024, 0, 0, 0, 0, 0, 0, 0,
    002, 042, 061, 021, 002, 061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    004, 042, 061, 004, 061, 001, 001, 061, 021, 0, 0, 0, 0, 0, 0, 0,
    002, 042, 061, 002, 061, 021, 012, 014, 001, 0, 0, 0, 0, 0, 0, 0,
    012, 014, 004, 012, 004, 061, 012, 061, 021, 042, 061, 004, 0, 0, 0, 0,
    061, 024, 042, 034, 012, 021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    001, 004, 002, 021, 034, 012, 042, 061, 024, 0, 0, 0, 0, 0, 0, 0,
    034, 014, 001, 034, 001, 021, 024, 042, 061, 0, 0, 0, 0, 0, 0, 0,
    061, 024, 042, 034, 004, 021, 034, 014, 004, 004, 002, 021, 0, 0, 0, 0,
    061, 034, 012, 061, 012, 042, 042, 012, 002, 0, 0, 0, 0, 0, 0, 0,
    004, 042, 061, 001, 004, 061, 001, 061, 034, 001, 034, 012, 0, 0, 0, 0,
    061, 034, 014, 061, 014, 002, 061, 002, 042, 002, 014, 001, 0, 0, 0, 0,
    061, 004, 042, 061, 034, 004, 034, 014, 004, 0, 0, 0, 0, 0, 0, 0,
    041, 061, 024, 004, 041, 024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    024, 002, 001, 024, 001, 061, 061, 001, 041, 0, 0, 0, 0, 0, 0, 0,
    041, 061, 024, 041, 024, 004, 001, 012, 014, 0, 0, 0, 0, 0, 0, 0,
    014, 002, 012, 014, 061, 002, 014, 041, 061, 061, 024, 002, 0, 0, 0, 0,
    002, 004, 041, 002, 041, 021, 021, 041, 061, 0, 0, 0, 0, 0, 0, 0,
    041, 061, 001, 001, 061, 021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 001, 012, 041, 021, 004, 041, 061, 021, 021, 002, 004, 0, 0, 0, 0,
    014, 021, 012, 014, 041, 021, 041, 061, 021, 0, 0, 0, 0, 0, 0, 0,
    024, 004, 041, 024, 041, 061, 034, 012, 021, 0, 0, 0, 0, 0, 0, 0,
    012, 021, 034, 001, 061, 002, 001, 041, 061, 061, 024, 002, 0, 0, 0, 0,
    041, 061, 004, 061, 024, 004, 001, 034, 014, 001, 021, 034, 0, 0, 0, 0,
    061, 002, 041, 061, 024, 002, 041, 002, 014, 021, 034, 002, 014, 002, 034, 0,
    004, 041, 061, 004, 061, 012, 004, 012, 002, 034, 012, 061, 0, 0, 0, 0,
    012, 061, 034, 012, 001, 061, 001, 041, 061, 0, 0, 0, 0, 0, 0, 0,
    014, 002, 034, 014, 001, 002, 034, 002, 061, 004, 041, 002, 061, 002, 041, 0,
    014, 061, 034, 041, 061, 014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    052, 041, 014, 061, 024, 042, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 052, 041, 004, 002, 001, 061, 024, 042, 0, 0, 0, 0, 0, 0, 0,
    041, 001, 012, 041, 012, 052, 061, 024, 042, 0, 0, 0, 0, 0, 0, 0,
    061, 024, 042, 041, 004, 052, 052, 004, 002, 052, 002, 012, 0, 0, 0, 0,
    061, 021, 002, 061, 002, 042, 041, 014, 052, 0, 0, 0, 0, 0, 0, 0,
    052, 041, 014, 061, 001, 042, 061, 021, 001, 001, 004, 042, 0, 0, 0, 0,
    042, 061, 021, 042, 021, 002, 052, 041, 012, 012, 041, 001, 0, 0, 0, 0,
    052, 004, 012, 052, 041, 004, 012, 004, 021, 042, 061, 004, 021, 004, 061, 0,
    041, 014, 052, 061, 024, 042, 012, 021, 034, 0, 0, 0, 0, 0, 0, 0,
    014, 052, 041, 034, 012, 021, 004, 002, 001, 061, 024, 042, 0, 0, 0, 0,
    042, 061, 024, 041, 021, 052, 041, 001, 021, 021, 034, 052, 0, 0, 0, 0,
    052, 041, 004, 052, 004, 002, 052, 002, 034, 021, 034, 002, 061, 024, 042, 0,
    014, 052, 041, 012, 042, 034, 012, 002, 042, 042, 061, 034, 0, 0, 0, 0,
    001, 004, 042, 001, 042, 061, 001, 061, 012, 034, 012, 061, 014, 052, 041, 0,
    042, 034, 002, 042, 061, 034, 002, 034, 001, 052, 041, 034, 001, 034, 041, 0,
    061, 004, 042, 061, 034, 004, 041, 004, 052, 052, 004, 034, 0, 0, 0, 0,
    052, 061, 024, 052, 024, 014, 014, 024, 004, 0, 0, 0, 0, 0, 0, 0,
    001, 024, 002, 014, 024, 001, 014, 061, 024, 014, 052, 061, 0, 0, 0, 0,
    061, 012, 052, 061, 004, 012, 061, 024, 004, 001, 012, 004, 0, 0, 0, 0,
    024, 052, 061, 024, 002, 052, 002, 012, 052, 0, 0, 0, 0, 0, 0, 0,
    052, 061, 021, 052, 021, 004, 052, 004, 014, 004, 021, 002, 0, 0, 0, 0,
    052, 001, 014, 052, 061, 001, 061, 021, 001, 0, 0, 0, 0, 0, 0, 0,
    021, 004, 061, 021, 002, 004, 061, 004, 052, 001, 012, 004, 052, 004, 012, 0,
    052, 021, 012, 061, 021, 052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    012, 021, 034, 052, 061, 014, 014, 061, 024, 014, 024, 004, 0, 0, 0, 0,
    014, 052, 061, 014, 061, 024, 014, 024, 001, 002, 001, 024, 012, 021, 034, 0,
    021, 052, 001, 021, 034, 052, 001, 052, 004, 061, 024, 052, 004, 052, 024, 0,
    024, 052, 061, 024, 002, 052, 034, 052, 021, 021, 052, 002, 0, 0, 0, 0,
    014, 061, 004, 014, 052, 061, 004, 061, 002, 034, 012, 061, 002, 061, 012, 0,
    012, 061, 034, 012, 001, 061, 052, 061, 014, 014, 061, 001, 0, 0, 0, 0,
    052, 061, 034, 004, 001, 002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    052, 061, 034, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    052, 034, 061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    052, 034, 061, 004, 002, 001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    052, 034, 061, 014, 001, 012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 004, 002, 014, 002, 012, 034, 061, 052, 0, 0, 0, 0, 0, 0, 0,
    034, 061, 052, 021, 002, 024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    021, 001, 004, 021, 004, 024, 061, 052, 034, 0, 0, 0, 0, 0, 0, 0,
    014, 001, 012, 034, 061, 052, 002, 024, 021, 0, 0, 0, 0, 0, 0, 0,
    052, 034, 061, 014, 024, 012, 014, 004, 024, 024, 021, 012, 0, 0, 0, 0,
    052, 012, 021, 061, 052, 021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    021, 061, 052, 021, 052, 012, 001, 004, 002, 0, 0, 0, 0, 0, 0, 0,
    052, 014, 001, 052, 001, 061, 061, 001, 021, 0, 0, 0, 0, 0, 0, 0,
    052, 021, 061, 052, 004, 021, 052, 014, 004, 004, 002, 021, 0, 0, 0, 0,
    024, 061, 052, 024, 052, 002, 002, 052, 012, 0, 0, 0, 0, 0, 0, 0,
    061, 052, 012, 061, 012, 004, 061, 004, 024, 001, 004, 012, 0, 0, 0, 0,
    001, 002, 024, 014, 001, 024, 014, 024, 061, 014, 061, 052, 0, 0, 0, 0,
    052, 024, 061, 052, 014, 024, 014, 004, 024, 0, 0, 0, 0, 0, 0, 0,
    041, 042, 004, 052, 034, 061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    042, 002, 001, 042, 001, 041, 052, 034, 061, 0, 0, 0, 0, 0, 0, 0,
    001, 012, 014, 041, 042, 004, 034, 061, 052, 0, 0, 0, 0, 0, 0, 0,
    034, 061, 052, 014, 041, 012, 012, 041, 042, 012, 042, 002, 0, 0, 0, 0,
    052, 034, 061, 042, 004, 041, 021, 002, 024, 0, 0, 0, 0, 0, 0, 0,
    052, 034, 061, 042, 024, 041, 041, 024, 021, 041, 021, 001, 0, 0, 0, 0,
    024, 021, 002, 041, 042, 004, 014, 001, 012, 034, 061, 052, 0, 0, 0, 0,
    041, 042, 024, 041, 024, 021, 041, 021, 014, 012, 014, 021, 052, 034, 061, 0,
    052, 012, 021, 052, 021, 061, 042, 004, 041, 0, 0, 0, 0, 0, 0, 0,
    052, 012, 061, 012, 021, 061, 042, 001, 041, 042, 002, 001, 0, 0, 0, 0,
    042, 004, 041, 052, 014, 061, 061, 014, 001, 061, 001, 021, 0, 0, 0, 0,
    061, 014, 021, 061, 052, 014, 021, 014, 002, 041, 042, 014, 002, 014, 042, 0,
    041, 042, 004, 052, 002, 061, 052, 012, 002, 002, 024, 061, 0, 0, 0, 0,
    041, 024, 001, 041, 042, 024, 001, 024, 012, 061, 052, 024, 012, 024, 052, 0,
    014, 001, 002, 014, 002, 024, 014, 024, 052, 061, 052, 024, 041, 042, 004, 0,
    042, 014, 041, 042, 024, 014, 052, 014, 061, 061, 014, 024, 0, 0, 0, 0,
    014, 034, 061, 041, 014, 061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 034, 061, 014, 061, 041, 004, 002, 001, 0, 0, 0, 0, 0, 0, 0,
    012, 034, 061, 012, 061, 001, 001, 061, 041, 0, 0, 0, 0, 0, 0, 0,
    004, 061, 041, 004, 012, 061, 004, 002, 012, 034, 061, 012, 0, 0, 0, 0,
    061, 041, 014, 061, 014, 034, 021, 002, 024, 0, 0, 0, 0, 0, 0, 0,
    001, 004, 024, 001, 024, 021, 041, 014, 061, 061, 014, 034, 0, 0, 0, 0,
    002, 024, 021, 012, 034, 001, 001, 034, 061, 001, 061, 041, 0, 0, 0, 0,
    024, 012, 004, 024, 021, 012, 004, 012, 041, 034, 061, 012, 041, 012, 061, 0,
    014, 012, 021, 014, 021, 041, 041, 021, 061, 0, 0, 0, 0, 0, 0, 0,
    004, 002, 001, 014, 012, 041, 041, 012, 021, 041, 021, 061, 0, 0, 0, 0,
    041, 001, 061, 001, 021, 061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    002, 041, 004, 002, 021, 041, 021, 061, 041, 0, 0, 0, 0, 0, 0, 0,
    014, 012, 002, 014, 002, 061, 014, 061, 041, 061, 002, 024, 0, 0, 0, 0,
    041, 012, 061, 041, 014, 012, 061, 012, 024, 001, 004, 012, 024, 012, 004, 0,
    024, 001, 002, 024, 061, 001, 061, 041, 001, 0, 0, 0, 0, 0, 0, 0,
    041, 024, 061, 004, 024, 041, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    061, 042, 004, 061, 004, 034, 034, 004, 014, 0, 0, 0, 0, 0, 0, 0,
    061, 014, 034, 061, 002, 014, 061, 042, 002, 002, 001, 014, 0, 0, 0, 0,
    004, 061, 042, 001, 061, 004, 001, 034, 061, 001, 012, 034, 0, 0, 0, 0,
    061, 012, 034, 061, 042, 012, 042, 002, 012, 0, 0, 0, 0, 0, 0, 0,
    021, 002, 024, 061, 042, 034, 034, 042, 004, 034, 004, 014, 0, 0, 0, 0,
    034, 042, 014, 034, 061, 042, 014, 042, 001, 024, 021, 042, 001, 042, 021, 0,
    001, 012, 034, 001, 034, 061, 001, 061, 004, 042, 004, 061, 002, 024, 021, 0,
    021, 042, 024, 021, 012, 042, 061, 042, 034, 034, 042, 012, 0, 0, 0, 0,
    012, 004, 014, 012, 061, 004, 012, 021, 061, 042, 004, 061, 0, 0, 0, 0,
    002, 014, 042, 002, 001, 014, 042, 014, 061, 012, 021, 014, 061, 014, 021, 0,
    004, 061, 042, 004, 001, 061, 001, 021, 061, 0, 0, 0, 0, 0, 0, 0,
    002, 061, 042, 021, 061, 002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    002, 061, 012, 002, 024, 061, 012, 061, 014, 042, 004, 061, 014, 061, 004, 0,
    014, 012, 001, 061, 042, 024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    024, 001, 002, 024, 061, 001, 004, 001, 042, 042, 001, 061, 0, 0, 0, 0,
    042, 024, 061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    042, 052, 034, 024, 042, 034, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    042, 052, 034, 042, 034, 024, 002, 001, 004, 0, 0, 0, 0, 0, 0, 0,
    034, 024, 042, 034, 042, 052, 014, 001, 012, 0, 0, 0, 0, 0, 0, 0,
    014, 004, 012, 004, 002, 012, 034, 042, 052, 034, 024, 042, 0, 0, 0, 0,
    034, 021, 002, 034, 002, 052, 052, 002, 042, 0, 0, 0, 0, 0, 0, 0,
    004, 042, 052, 004, 052, 021, 004, 021, 001, 021, 052, 034, 0, 0, 0, 0,
    014, 001, 012, 034, 021, 052, 052, 021, 002, 052, 002, 042, 0, 0, 0, 0,
    052, 021, 042, 052, 034, 021, 042, 021, 004, 012, 014, 021, 004, 021, 014, 0,
    021, 024, 042, 021, 042, 012, 012, 042, 052, 0, 0, 0, 0, 0, 0, 0,
    004, 002, 001, 042, 012, 024, 042, 052, 012, 012, 021, 024, 0, 0, 0, 0,
    014, 001, 021, 014, 021, 042, 014, 042, 052, 024, 042, 021, 0, 0, 0, 0,
    004, 021, 014, 004, 002, 021, 014, 021, 052, 024, 042, 021, 052, 021, 042, 0,
    052, 012, 042, 042, 012, 002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    004, 012, 001, 004, 042, 012, 042, 052, 012, 0, 0, 0, 0, 0, 0, 0,
    001, 052, 014, 001, 002, 052, 002, 042, 052, 0, 0, 0, 0, 0, 0, 0,
    052, 004, 042, 014, 004, 052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    041, 052, 034, 041, 034, 004, 004, 034, 024, 0, 0, 0, 0, 0, 0, 0,
    052, 034, 024, 052, 024, 001, 052, 001, 041, 002, 001, 024, 0, 0, 0, 0,
    001, 012, 014, 041, 052, 004, 004, 052, 034, 004, 034, 024, 0, 0, 0, 0,
    012, 041, 002, 012, 014, 041, 002, 041, 024, 052, 034, 041, 024, 041, 034, 0,
    034, 021, 002, 052, 034, 002, 052, 002, 004, 052, 004, 041, 0, 0, 0, 0,
    034, 041, 052, 034, 021, 041, 021, 001, 041, 0, 0, 0, 0, 0, 0, 0,
    052, 034, 021, 052, 021, 002, 052, 002, 041, 004, 041, 002, 014, 001, 012, 0,
    034, 041, 052, 034, 021, 041, 014, 041, 012, 012, 041, 021, 0, 0, 0, 0,
    041, 024, 004, 041, 012, 024, 041, 052, 012, 012, 021, 024, 0, 0, 0, 0,
    012, 024, 052, 012, 021, 024, 052, 024, 041, 002, 001, 024, 041, 024, 001, 0,
    004, 052, 024, 004, 041, 052, 024, 052, 021, 014, 001, 052, 021, 052, 001, 0,
    014, 041, 052, 002, 021, 024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    041, 002, 004, 041, 052, 002, 052, 012, 002, 0, 0, 0, 0, 0, 0, 0,
    041, 012, 001, 052, 012, 041, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    001, 052, 014, 001, 002, 052, 041, 052, 004, 004, 052, 002, 0, 0, 0, 0,
    014, 041, 052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    042, 041, 014, 042, 014, 024, 024, 014, 034, 0, 0, 0, 0, 0, 0, 0,
    001, 004, 002, 014, 024, 041, 014, 034, 024, 024, 042, 041, 0, 0, 0, 0,
    012, 041, 001, 012, 024, 041, 012, 034, 024, 024, 042, 041, 0, 0, 0, 0,
    024, 041, 034, 024, 042, 041, 034, 041, 012, 004, 002, 041, 012, 041, 002, 0,
    041, 002, 042, 041, 034, 002, 041, 014, 034, 021, 002, 034, 0, 0, 0, 0,
    001, 042, 021, 001, 004, 042, 021, 042, 034, 041, 014, 042, 034, 042, 014, 0,
    001, 034, 041, 001, 012, 034, 041, 034, 042, 021, 002, 034, 042, 034, 002, 0,
    041, 004, 042, 012, 034, 021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    042, 021, 024, 041, 021, 042, 041, 012, 021, 041, 014, 012, 0, 0, 0, 0,
    041, 014, 012, 041, 012, 021, 041, 021, 042, 024, 042, 021, 004, 002, 001, 0,
    042, 021, 024, 042, 041, 021, 041, 001, 021, 0, 0, 0, 0, 0, 0, 0,
    002, 041, 004, 002, 021, 041, 042, 041, 024, 024, 041, 021, 0, 0, 0, 0,
    014, 042, 041, 014, 012, 042, 012, 002, 042, 0, 0, 0, 0, 0, 0, 0,
    004, 012, 001, 004, 042, 012, 014, 012, 041, 041, 012, 042, 0, 0, 0, 0,
    001, 042, 041, 002, 042, 001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    041, 004, 042, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 034, 004, 034, 024, 004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    001, 024, 002, 001, 014, 024, 014, 034, 024, 0, 0, 0, 0, 0, 0, 0,
    012, 004, 001, 012, 034, 004, 034, 024, 004, 0, 0, 0, 0, 0, 0, 0,
    024, 012, 034, 002, 012, 024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    002, 034, 021, 002, 004, 034, 004, 014, 034, 0, 0, 0, 0, 0, 0, 0,
    034, 001, 014, 021, 001, 034, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    002, 034, 021, 002, 004, 034, 012, 034, 001, 001, 034, 004, 0, 0, 0, 0,
    034, 021, 012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    021, 014, 012, 021, 024, 014, 024, 004, 014, 0, 0, 0, 0, 0, 0, 0,
    021, 014, 012, 021, 024, 014, 001, 014, 002, 002, 014, 024, 0, 0, 0, 0,
    021, 004, 001, 024, 004, 021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    021, 024, 002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 002, 004, 012, 002, 014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    014, 012, 001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    001, 002, 004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  FieldFormat format = FieldFormat::Float;
  uint32_t nx = 255;
  uint32_t ny = 255;
  uint32_t nz = 255;

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
uniform layout(location=2) vec4 color;
uniform layout(location=3) uint shade;

void main() {
 
  float d = max(0.0, dot(vec3(0,0,1), gl_FrontFacing ? -normal : normal));

  if(gl_FrontFacing)
    outColor = d * color.rgba;
  else
    outColor = color.bgra;
  //outColor = vec4(abs(normal), 1);
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
                    dst[i] = v;
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
    if (i + 1 < argc && (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0)) { deviceIndex = std::atoi(argv[i + 1]); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-nx") == 0) { nx = uint32_t(std::atoi(argv[i + 1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-ny") == 0) { ny = uint32_t(std::atoi(argv[i + 1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-nz") == 0) { nz = uint32_t(std::atoi(argv[i + 1])); i++; }
    else if (i + 1 < argc && strcmp(argv[i], "-n") == 0) { nx = uint32_t(std::atoi(argv[i + 1])); ny = nx; nz = nx; i++; }
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

  unsigned int deviceCount;
  CHECKED_CUDA(cudaGLGetDevices(&deviceCount, nullptr, 0, cudaGLDeviceListAll));
  if (deviceCount == 0) {
    fprintf(stderr, "No CUDA-enabled devices available.");
    return EXIT_FAILURE;
  }
  std::vector<int> devices(deviceCount);
  CHECKED_CUDA(cudaGLGetDevices(&deviceCount, devices.data(), deviceCount, cudaGLDeviceListAll));

  bool found = false;
  for (unsigned k = 0; k < deviceCount; k++) {
    int i = devices[k];
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, i);
    fprintf(stderr, "%c[%i] %s cap=%d.%d\n", i == deviceIndex ? '*' : ' ', i, dev_prop.name, dev_prop.major, dev_prop.minor);
    if (i == deviceIndex) {
      found = true;
    }
  }
  if (!found) {
    fprintf(stderr, "Illegal CUDA device index %d\n", deviceIndex);
    return EXIT_FAILURE;
  }
  cudaSetDevice(deviceIndex);
  CHECKED_CUDA(cudaStreamCreate(&stream));

  path = "cayley";

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
  float* deviceMem = nullptr;
  CHECKED_CUDA(cudaMalloc(&deviceMem, scalarField_host.size()));
  CHECKED_CUDA(cudaMemcpyAsync(deviceMem, scalarField_host.data(), scalarField_host.size(), cudaMemcpyHostToDevice, stream));

  assert(nx * ny * nz * 4 == scalarField_host.size());
  const float threshold = 0.f;
  std::vector<float> vertices;
  unsigned populated = 0;
  const float* f = reinterpret_cast<const float*>(scalarField_host.data());
  for (unsigned k = 0; k + 1 < nz; k++) {
    for (unsigned j = 0; j + 1 < ny; j++) {
      for (unsigned i = 0; i + 1 < nx; i++) {
        unsigned mccase =
          (f[((k + 0) * ny + (j + 0)) * nx + (i + 0)] < threshold ? (1 << 0) : 0) |
          (f[((k + 0) * ny + (j + 0)) * nx + (i + 1)] < threshold ? (1 << 1) : 0) |
          (f[((k + 0) * ny + (j + 1)) * nx + (i + 0)] < threshold ? (1 << 2) : 0) |
          (f[((k + 0) * ny + (j + 1)) * nx + (i + 1)] < threshold ? (1 << 3) : 0) |
          (f[((k + 1) * ny + (j + 0)) * nx + (i + 0)] < threshold ? (1 << 4) : 0) |
          (f[((k + 1) * ny + (j + 0)) * nx + (i + 1)] < threshold ? (1 << 5) : 0) |
          (f[((k + 1) * ny + (j + 1)) * nx + (i + 0)] < threshold ? (1 << 6) : 0) |
          (f[((k + 1) * ny + (j + 1)) * nx + (i + 1)] < threshold ? (1 << 7) : 0);

        if (mccase != 0 && mccase != 255) {
          populated++;
        }

        const unsigned N = index_count[mccase];
        for (unsigned l = 0; l <N; l++) {
          const auto r = mc_triangles[16*mccase + l];

          vertices.push_back((2 * i + 2*((r >> 3) & 1) + ((r >> 0) & 1)) / (2.f * nx));
          vertices.push_back((2 * j + 2*((r >> 4) & 1) + ((r >> 1) & 1)) / (2.f * ny));
          vertices.push_back((2 * k + 2*((r >> 5) & 1) + ((r >> 2) & 1)) / (2.f * nz));
        }
      }
    }
  }
  fprintf(stderr, "indexcount %zu, populated=%u of %u\n", vertices.size() / 3, populated, (nx - 1)* (ny - 1)* (nz - 1));




  auto* tables = createTables(stream);
  auto* ctx = createContext(tables, make_uint3(nx-1, ny-1, nz-1), true, stream);

  

  GLuint simpleVS = createShader(simpleVS_src, GL_VERTEX_SHADER);
  assert(simpleVS != 0);

  GLuint simpleFS = createShader(simpleFS_src, GL_FRAGMENT_SHADER);
  assert(simpleFS != 0);

  GLuint simplePrg = createProgram(simpleVS, simpleFS);
  assert(simplePrg != 0);

  //GLuint vdatabuf = createBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(vertexData), (const void*)vertexData);
  GLuint vdatabuf = createBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(float)*vertices.size(), (const void*)vertices.data());
  assert(vdatabuf != 0);

  const uint32_t N = vertices.size() / 3;

  GLuint vbo = 0;
  glGenVertexArrays(1, &vbo);
  glBindVertexArray(vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vdatabuf);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr);
//  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void*)(sizeof(float) * 3));
  glEnableVertexAttribArray(0);
//  glEnableVertexAttribArray(1);



  unsigned eventCounter = 0;
  cudaEvent_t events[2 * 4];
  for (size_t i = 0; i < 2 * 4; i++) {
    CHECKED_CUDA(cudaEventCreate(&events[i]));
    CHECKED_CUDA(cudaEventRecord(events[i], stream));
  }

  GLuint cudaBuf = createBuffer(GL_ARRAY_BUFFER, GL_STREAM_DRAW, 3 * sizeof(float), nullptr);
  cudaGraphicsResource* bufferResource = nullptr;
  CHECKED_CUDA(cudaGraphicsGLRegisterBuffer(&bufferResource, cudaBuf, cudaGraphicsRegisterFlagsWriteDiscard));

  GLuint cudaVbo = 0;
  glGenVertexArrays(1, &cudaVbo);
  glBindVertexArray(cudaVbo);
  glBindBuffer(GL_ARRAY_BUFFER, cudaBuf);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, nullptr);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void*)(sizeof(float) * 3));
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);


  auto start = std::chrono::system_clock::now();
  auto timer = std::chrono::high_resolution_clock::now();
  float cuda_ms = 0.f;
  unsigned frames = 0;
  while (!glfwWindowShouldClose(win)) {
    int width, height;
    glfwGetWindowSize(win, &width, &height);

    float iso = 0.0;
    switch (format) {
    case FieldFormat::UInt8:
      iso = 0.5f * 255.f * (iso + 1.f);
      break;
    case FieldFormat::UInt16:
      iso = 0.5f * 65535.f * (iso + 1.f);
    default: break;
    }


    uint32_t vertices = 0;
    uint32_t indices = 0;
    {
      void* cudaBuf_d = nullptr;
      size_t cudaBuf_size = 0;

      CHECKED_CUDA(cudaGraphicsMapResources(1, &bufferResource, stream));
      CHECKED_CUDA(cudaGraphicsResourceGetMappedPointer(&cudaBuf_d, &cudaBuf_size, bufferResource));
      CHECKED_CUDA(cudaEventRecord(events[2 * eventCounter + 0], stream));
      ComputeStuff::MC::buildPN(ctx,
                                cudaBuf_d,
                                cudaBuf_size,
                                nx,
                                nx* ny,
                                make_uint3(0, 0, 0),
                                make_uint3(nx, ny, nz),
                                deviceMem,
                                threshold,
                                stream,
                                true,
                                false);
      CHECKED_CUDA(cudaEventRecord(events[2 * eventCounter + 1], stream));
      CHECKED_CUDA(cudaGraphicsUnmapResources(1, &bufferResource, stream));
      ComputeStuff::MC::getCounts(ctx, &vertices, &indices, stream);

      eventCounter = (eventCounter + 1) & 3;
      float ms = 0;
      CHECKED_CUDA(cudaEventElapsedTime(&ms, events[2 * eventCounter + 0], events[2 * eventCounter + 1]));
      cuda_ms += ms;

      if (cudaBuf_size < 6 * sizeof(float) * vertices) {

        CHECKED_CUDA(cudaGraphicsUnregisterResource(bufferResource));

        size_t newSize = 6 * sizeof(float) * (static_cast<size_t>(vertices) + vertices / 16);
        fprintf(stderr, "Resizing VBO to %zu bytes\n", newSize);
        glBindBuffer(GL_ARRAY_BUFFER, cudaBuf);
        glBufferData(GL_ARRAY_BUFFER, newSize, nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        CHECKED_CUDA(cudaGraphicsGLRegisterBuffer(&bufferResource, cudaBuf, cudaGraphicsRegisterFlagsWriteDiscard));

        CHECKED_CUDA(cudaGraphicsMapResources(1, &bufferResource, stream));
        CHECKED_CUDA(cudaGraphicsResourceGetMappedPointer(&cudaBuf_d, &cudaBuf_size, bufferResource));
        ComputeStuff::MC::buildPN(ctx,
                                  cudaBuf_d,
                                  cudaBuf_size,
                                  nx,
                                  nx*ny,
                                  make_uint3(0, 0, 0),
                                  make_uint3(nx, ny, nz),
                                  deviceMem,
                                  threshold,
                                  stream,
                                  false,
                                  true);
        CHECKED_CUDA(cudaGraphicsUnmapResources(1, &bufferResource, stream));
      }
    }
    glViewport(0, 0, width, height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
    auto seconds = elapsed.count();

    float center[16];
    translateMatrix(center, -0.5f, -0.5f, -0.5f);

    float rx[16];
    rotMatrixX(rx, static_cast<float>(1.1 * seconds));

    float ry[16];
    rotMatrixY(ry, static_cast<float>(1.7 * seconds));

    float rz[16];
    rotMatrixZ(rz, static_cast<float>(1.3 * seconds));

    float shift[16];
    translateMatrix(shift, 0.f, 0.f, -2.0f);

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

#if 0
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(simplePrg);
    glBindVertexArray(vbo);
    glUniformMatrix4fv(0, 1, GL_FALSE, rz_ry_rx);
    glUniformMatrix4fv(1, 1, GL_FALSE, frustum_shift_rz_ry_rx);



    glPolygonOffset(1.f, 1.f);
    glEnable(GL_POLYGON_OFFSET_FILL);

    glCullFace(GL_FRONT);
    glUniform4f(2, 0.6f, 0.5f, 0.5f, 1.f);
    glDrawArrays(GL_TRIANGLES, 0, N);

    glCullFace(GL_BACK);
    glUniform4f(2, 0.5f, 0.5f, 0.6f, 1.f);
    glDrawArrays(GL_TRIANGLES, 0, N);

    glDisable(GL_POLYGON_OFFSET_FILL);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glCullFace(GL_FRONT);
    glUniform4f(2, 0.2f, 0.2f, 0.2f, 1.f);
    glDrawArrays(GL_TRIANGLES, 0, N);
    glCullFace(GL_BACK);
    glUniform4f(2, 1.f, 1.f, 1.f, 1.f);
    glDrawArrays(GL_TRIANGLES, 0, N);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    CHECK_GL;

#endif
    glEnable(GL_DEPTH_TEST);
    glUseProgram(simplePrg);
    glBindVertexArray(cudaVbo);
    glUniformMatrix4fv(0, 1, GL_FALSE, rz_ry_rx);
    glUniformMatrix4fv(1, 1, GL_FALSE, frustum_shift_rz_ry_rx);


    glPolygonOffset(0.f, 1.f);
    glEnable(GL_POLYGON_OFFSET_FILL);

    glUniform4f(2, 0.6f, 0.6f, 0.8f, 1.f);
    glUniform1i(3, 1);
    glDrawArrays(GL_POINTS, 0, N);
    glDisable(GL_POLYGON_OFFSET_FILL);

#if 0
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glUniform1i(3, 0);
    glUniform4f(2, 1.f, 1.f, 1.f, 1.f);
    glDrawArrays(GL_TRIANGLES, 0, N);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif

    glfwSwapBuffers(win);
    glfwPollEvents();

    {
      frames++;
      auto now = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = now - timer;
      auto s = elapsed.count();
      if (10 < frames && 3.0 < s) {
        fprintf(stderr, "%.2f FPS (%.2f MVPS) cuda avg: %.2fms (%.2f MVPS) %ux%ux%u\n",
                frames / s, (float(frames)* nx *ny * nz) / (1000000.f * s),
                cuda_ms/frames, (float(frames)* nx* ny* nz) / (1000.f * cuda_ms),
                nx, ny, nz);
        timer = now;
        frames = 0;
        cuda_ms = 0.f;
      }
    }


  }
  glfwDestroyWindow(win);
  glfwTerminate();

  glDeleteShader(simpleVS);

  return EXIT_SUCCESS;
}
