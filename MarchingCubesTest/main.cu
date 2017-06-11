#include <string>
#include <iostream>
#include <fstream>

int main(int argc, char** argv)
{

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " filename.pvm" << std::endl;
    return EXIT_FAILURE;
  }
  std::string filename = argv[1];

  std::ifstream in(filename.c_str(), std::ifstream::in || std::ifstream::binary);
  if (!in.good()) {
    std::cerr << "Failed to read " << filename << std::endl;
    return EXIT_FAILURE;
  }

  std::string header;
  std::getline(in, header, '\n');
  if (header == "PVM") {
  }
  else if (header == "PVM2") {
  }
  else if (header == "PVM3") {
  }
  else if (header == "DDS v3d") {
  }
  else if (header == "DDS v3e") {
  }
  else {
    std::cerr << "Unknown header: " << header << std::endl;
    return EXIT_FAILURE;
  }

}