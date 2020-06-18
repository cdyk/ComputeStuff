# ComputeStuff

MIT-licensed stand-alone CUDA utility functions.

- [ComputeStuff](#computestuff)
  - [Introduction](#introduction)
    - [Building dependencies](#building-dependencies)
    - [How to use](#how-to-use)
      - [Compute capability](#compute-capability)
      - [Scratch buffer](#scratch-buffer)
      - [Write sum](#write-sum)
      - [Concurrent invocations](#concurrent-invocations)
    - [In-place operation](#in-place-operation)
  - [Prefix sum / Scan](#prefix-sum--scan)
    - [Performance](#performance)
  - [5:1 HistoPyramids](#51-histopyramids)
    - [Performance](#performance-1)
  - [Credits](#credits)
  - [License](#license)


## Introduction

The intent is to make various useful functions of mine available for the public under a liberal license --- so that anyone can freely use the code without any license issues.

The functions are designed to have a minimum of dependencies, so integration is mostly just adding the relevant files into your project.

I've started with variants of scan and the 5-to-1 HistoPyramid, and will continue with HistoPyramid variations and Marching Cubes.

### Building dependencies

The Marching Cubes test application project uses [GLFW](https://www.glfw.org/) and OpenGL to open a window that visualizes the result. If you plan to build that project, you need to  to do the following.

Clone the git repo, and update submodules
```
git submodule update --init --recursive
```
Then configure GLFW by
```
cd libs\glfw
mkdir build
cd build
cmake-gui ..
```
Press "Configure" button and select 64bit VS2017. Open libs\glfw\build\GLFW.sln and build.


### How to use

Each of the components is fully contained in its own project (currently [Scan project](Scan/Scan.vcxproj) and [HP5 project](HP5/HP5.vcxproj)). To use the code, either link against the static library produced by the project you want to use, or just add the files in the project to your project.

Typically these projects contains a header file with the public API and one or more source files with the implementation.

In addition, each component have a test project (currently [ScanTest project](ScanTest/ScanTest.vcxproj) and [HP5Test project](HP5Test/HP5Test.vcxproj)) that serves as a combined unit and performance test as well as an example of use.

#### Compute capability

Code in ComputeStuff uses the Kepler warp-shuffle instructions, and therefore the **minimum supported compute capability is 3.0**. It is straight-forward to replace the use of shuffle with use of shared memory, at the expense of some more instructions, and lowering this requirement.


#### Scratch buffer

Most function needs some scratch memory to store temporary calculations. These are uniform each component only dependent on problem size.

For example, all scan-based functions takes a scratch-buffer as an argument, where the size of the buffer is given by `ComputeStuff::Scan::scratchByteSize(N)`. It is the application developer's responsibility to create this buffer, e.g.,
````c++
uint32_t*scratch_d;
cudaMalloc(&scratch_d, ComputeStuff::Scan::scratchByteSize(N));
````
allocates an appropriate buffer.

This let you have total control of memory management, so no strange malloc's or copy's is injected behind the scenes.

This buffer can be recycled between invocations that don't overlap, that is, they don't run at the same time on different streams --- in that case you need multiple scratch buffers.

Also, the size of these buffers grows monotonically with N, so if you are going to run several different problem sizes, just use the largest problem to determine the size of a single scratch buffer size and you should be good.

#### Write sum

Some API functions can optionally write the total sum of the input to anywhere in device memory. This is useful for zero-copying the result back to the CPU, if some subseqent code (like a grid size) is dependent on this number.

An example using the scan API:
````c++
// In setup:
// Alloc host-mem that can be mapped in the GPU
uint32_t* sum_h;
cudaHostAlloc(&sum_h, sizeof(uint32_t), cudaHostAllocMapped);

// Get device pointer of sum_h mapped to GPU memory space.
uint32_t *sum_d;
cudaHostGetDevicePointer(&sum_d, sum_h, 0);

// During run:
// Now, input_d is populated with input, build offset table_
ComputeStuff::Scan::calcOffsets(output_d, sum_d, scratch_d, input_d, N, stream);
cudaEventRecord(calcOffsetsDone, stream);

// ... Optionally do something else ...

// Wait for kernel has finished running.
cudaEventSynchronize(calcOffsetsDone);

// Now, the sum is available, e.g. use it to determine a grid size.
someKernel<<<(sum_h+1023)/1024, 1024, 0, stream>>>(foo, bar, ...);
````

#### Concurrent invocations

All API functions have an optional stream argument, which, not surprisingly, will insert the GPU operations on that given stream.

Unless **explicitly stated**, the functions use no global state, and hence it should be completely safe to run several instances of the functions at the same time on different stream --- as long as they have separate scratch buffers, though.

### In-place operation

Unless **explicitly stated**, all functions supports (without any penalty) in-place operation, that is using the same buffer for input and output.

For scan, an example is:
````c++
// inout_d contains the input of N elements
ComputeStuff::Scan::exclusiveScan(inout_d, scratch_d, inout_d, N);
// inout_d contains the output of N elements
````

## Prefix sum / Scan

Implementation of

* **exclusive scan:** From [a,b,c,d] calculate [0, a, a+b, a+b+c] (a.k.a exclusive prefix-sum).
* **inclusive scan:** From [a,b,c,d] calculate [a, a+b, a+b+c, a+b+c+d] (a.k.a inclusive prefix-sum).
* **offset-table:** Calculate an offset-table (exclusive prefix-sum with total concatenated on end), and optionally write the total sum somewhere in device memory. From [a,b,c,d] calculate [0, a, a+b, a+b+c, a+b+c+d], so that element _i_ contains the offset _i_ and the difference betweem elements _i_ and _i+1_ is the corresponding value that was in the input array. 
* **compact:** Extract indices where the corresponding input has a non-zero value (a.k.a. subset, stream-compact and copy-if), and write the number of entries in the output somewhere in device memory.

Please consult [Scan/Scan.h](Scan/Scan.h) for details on the API entry points.

### Performance

The implementation has decent performance, the numbers below is a comparison with thrust's exclusive scan using CUDA 11 on an RTX 2080. See [ScanTest/main.cu](ScanTest/main.cu) for details.

| N | thrust | ComputeStuff | ratio |
|---|--------|--------------|-------|
|1|0.0447008ms|0.00257408ms|0.0575847|
|3|0.0449325ms|0.00286464ms|0.0637543|
|10|0.0448845ms|0.0026624ms|0.0593167|
|33|0.0448166ms|0.00267072ms|0.0595921|
|110|0.0458342ms|0.0027072ms|0.059065|
|366|0.0456371ms|0.00270336ms|0.059236|
|1220|0.044855ms|0.00775232ms|0.172831|
|4066|0.0570458ms|0.00770048ms|0.134988|
|13553|0.0449408ms|0.00922368ms|0.205241|
|45176|0.0450163ms|0.00804224ms|0.178652|
|150586|0.0563738ms|0.00872384ms|0.15475|
|501953|0.0529978ms|0.0173939ms|0.328201|
|1673176|0.0787661ms|0.0643507ms|0.816985|
|5577253|0.185761ms|0.203887ms|1.09757|
|18590843|0.470511ms|0.581562ms|1.23602|
|61969476|1.43524ms|1.9094ms|1.33037|
|206564920|5.09301ms|6.43358ms|1.26322|




## 5:1 HistoPyramids

The 5-to-1 HistoPyramid is a variation of the traditional 4-to-1 HistoPyramid (or just HistoPyramid) that, instead of reducing four elements into one, it reduces five elements, hence the name.

The details are explained in [GPU Accelerated Data Expansion Marching Cubes Algorithm](http://on-demand.gputechconf.com/gtc/2010/presentations/S12020-GPU-Accelerated-Data-Expansion-Marching-Cubes-Algorithm.pdf) from GTC 2010.


Implementation of

* **compact:** Extract indices where the corresponding input has a non-zero value (a.k.a. subset, stream-compact and copy-if), and write the number of entries in the output somewhere in device memory.

Please consult [HP5/HP5.h](HP5/HP5.h) for details on the API entry points.

### Performance

The implementation starts to become decent. The numbers below is a comparsion to ComputeStuff's own scan implementation (see [Scan performance](#performance)), and was run on an RTX 2080 on CUDA 11.

Ratio is the time HP5 uses compared to Scan (lower is better for HP5's case). The test runs compact on N elements where "% selected" of the elements are marked for selection. In the min-scatter scenario, all selected elements are in the start of the buffer, which is the best-case for HP cache utilization, while max-scatter evenly spaces the selected element out, which is the worst-case for HP cache utilization. Please consult [HP5Test/main.cu](HP5Test/main.cu) for further details.


| N    | % selected| selected | min-scatter scan | min-scatter hp5    |min-scatter ratio| max-scatter scan | max-scatter hp5    | max-scatter ratio|
|------|-------|-------------|-----|----------|-------|---------|----------|------|
| 1220 | 3.13% | 39 | 0.00963ms | 0.01ms | 1.04 | 0.00958ms | 0.0101ms | 1.05 |
| 4066 | 3.13% | 128 | 0.0098ms | 0.0119ms | 1.22 | 0.00975ms | 0.0121ms | 1.24 |
| 13553 | 3.13% | 424 | 0.0166ms | 0.0208ms | 1.26 | 0.00998ms | 0.0122ms | 1.22 |
| 45176 | 3.13% | 1412 | 0.00996ms | 0.015ms | 1.51 | 0.01ms | 0.0153ms | 1.53 |
| 150586 | 3.13% | 4706 | 0.0111ms | 0.0162ms | 1.46 | 0.0175ms | 0.0161ms | 0.918 |
| 501953 | 3.13% | 15687 | 0.0311ms | 0.0203ms | 0.652 | 0.0197ms | 0.0182ms | 0.922 |
| 1673176 | 3.13% | 52287 | 0.0499ms | 0.0342ms | 0.685 | 0.0506ms | 0.0331ms | 0.655 |
| 5577253 | 3.13% | 174290 | 0.126ms | 0.0872ms | 0.692 | 0.129ms | 0.0864ms | 0.672 |
| 18590843 | 3.13% | 580964 | 0.377ms | 0.254ms | 0.674 | 0.387ms | 0.259ms | 0.67 |
| 61969476 | 3.13% | 1936547 | 1.21ms | 0.809ms | 0.666 | 1.26ms | 0.828ms | 0.656 |
| 206564920 | 3.13% | 6455154 | 4.03ms | 2.71ms | 0.674 | 4.13ms | 2.81ms | 0.68 |
| 1220 | 1.56% | 20 | 0.00958ms | 0.00996ms | 1.04 | 0.0097ms | 0.0124ms | 1.28 |
| 4066 | 1.56% | 64 | 0.00974ms | 0.0173ms | 1.78 | 0.0159ms | 0.0122ms | 0.765 |
| 13553 | 1.56% | 212 | 0.01ms | 0.012ms | 1.2 | 0.0099ms | 0.0121ms | 1.22 |
| 45176 | 1.56% | 706 | 0.01ms | 0.0151ms | 1.5 | 0.0106ms | 0.018ms | 1.7 |
| 150586 | 1.56% | 2353 | 0.0109ms | 0.0156ms | 1.43 | 0.0111ms | 0.0158ms | 1.43 |
| 501953 | 1.56% | 7844 | 0.0203ms | 0.0185ms | 0.912 | 0.0197ms | 0.0189ms | 0.962 |
| 1673176 | 1.56% | 26144 | 0.0498ms | 0.0338ms | 0.679 | 0.05ms | 0.0316ms | 0.631 |
| 5577253 | 1.56% | 87145 | 0.134ms | 0.0819ms | 0.609 | 0.127ms | 0.0811ms | 0.637 |
| 18590843 | 1.56% | 290482 | 0.373ms | 0.24ms | 0.643 | 0.38ms | 0.242ms | 0.636 |
| 61969476 | 1.56% | 968274 | 1.21ms | 0.782ms | 0.644 | 1.23ms | 0.772ms | 0.628 |
| 206564920 | 1.56% | 3227577 | 3.98ms | 2.58ms | 0.648 | 4.07ms | 2.61ms | 0.642 |
| 1220 | 0.781% | 10 | 0.0129ms | 0.00999ms | 0.777 | 0.00958ms | 0.0102ms | 1.06 |
| 4066 | 0.781% | 32 | 0.00965ms | 0.012ms | 1.25 | 0.00975ms | 0.012ms | 1.23 |
| 13553 | 0.781% | 106 | 0.01ms | 0.012ms | 1.19 | 0.00995ms | 0.0121ms | 1.22 |
| 45176 | 0.781% | 353 | 0.0105ms | 0.0151ms | 1.44 | 0.0101ms | 0.0154ms | 1.53 |
| 150586 | 0.781% | 1177 | 0.011ms | 0.0155ms | 1.41 | 0.0178ms | 0.0222ms | 1.25 |
| 501953 | 0.781% | 3922 | 0.0201ms | 0.0188ms | 0.934 | 0.0198ms | 0.023ms | 1.16 |
| 1673176 | 0.781% | 13072 | 0.0495ms | 0.0327ms | 0.661 | 0.0499ms | 0.0315ms | 0.632 |
| 5577253 | 0.781% | 43573 | 0.125ms | 0.0784ms | 0.626 | 0.127ms | 0.0773ms | 0.61 |
| 18590843 | 0.781% | 145241 | 0.371ms | 0.229ms | 0.618 | 0.376ms | 0.238ms | 0.633 |
| 61969476 | 0.781% | 484137 | 1.2ms | 0.754ms | 0.63 | 1.21ms | 0.758ms | 0.625 |
| 206564920 | 0.781% | 1613789 | 4ms | 2.49ms | 0.622 | 4.03ms | 2.56ms | 0.634 |
| 1220 | 0.391% | 5 | 0.00991ms | 0.00993ms | 1 | 0.00956ms | 0.0102ms | 1.07 |
| 4066 | 0.391% | 16 | 0.00971ms | 0.012ms | 1.24 | 0.00975ms | 0.0121ms | 1.24 |
| 13553 | 0.391% | 53 | 0.0151ms | 0.018ms | 1.19 | 0.00991ms | 0.0122ms | 1.23 |
| 45176 | 0.391% | 177 | 0.00997ms | 0.015ms | 1.51 | 0.0101ms | 0.0155ms | 1.54 |
| 150586 | 0.391% | 589 | 0.011ms | 0.0154ms | 1.39 | 0.0112ms | 0.0178ms | 1.59 |
| 501953 | 0.391% | 1961 | 0.0233ms | 0.0191ms | 0.821 | 0.02ms | 0.0181ms | 0.904 |
| 1673176 | 0.391% | 6536 | 0.0492ms | 0.0325ms | 0.661 | 0.0548ms | 0.032ms | 0.583 |
| 5577253 | 0.391% | 21787 | 0.125ms | 0.0754ms | 0.603 | 0.124ms | 0.0742ms | 0.601 |
| 18590843 | 0.391% | 72621 | 0.37ms | 0.227ms | 0.613 | 0.373ms | 0.236ms | 0.633 |
| 61969476 | 0.391% | 242069 | 1.28ms | 0.785ms | 0.614 | 1.21ms | 0.763ms | 0.633 |
| 206564920 | 0.391% | 806895 | 3.96ms | 2.45ms | 0.62 | 3.98ms | 2.54ms | 0.639 |


## Credits

The ComputeStuff implementation was initially written and is maintained by Christopher Dyken, with contributions from Gernot Ziegler.

## License

ComputeStuff is licensed under the MIT license, please see [LICENSE](LICENSE) for more information.

# Old performance numbers

## CUDA 8 on GTX 980 Ti

### ScanTest

| N | thrust | ComputeStuff | ratio |
|---|--------|--------------|-------|
|1|0.0108365ms|0.00554624ms|0.511812|
|3|0.0108845ms|0.00551104ms|0.506321|
|10|0.0106534ms|0.00552704ms|0.518803|
|33|0.0105331ms|0.0055488ms|0.526796|
|110|0.0105408ms|0.00555968ms|0.527444|
|366|0.0107462ms|0.00557568ms|0.518849|
|1220|0.0109805ms|0.0159718ms|1.45457|
|4066|0.0170432ms|0.0162426ms|0.953023|
|13553|0.0322854ms|0.0167046ms|0.517405|
|45176|0.108681ms|0.0174771ms|0.160811|
|150586|0.774404ms|0.0223123ms|0.0288123|
|501953|0.143727ms|0.0459693ms|0.319838|
|1673176|0.760692ms|0.106328ms|0.139778|
|5577253|0.965567ms|0.286347ms|0.296558|
|18590843|1.98413ms|0.895598ms|0.451381|
|61969476|4.88424ms|2.91978ms|0.597797|
|206564920|14.1531ms|9.76045ms|0.689633|

### HP5Test

| N    | % selected| selected | min-scatter scan | min-scatter hp5    |min-scatter ratio| max-scatter scan | max-scatter hp5    | max-scatter ratio|
|------|-------|-------------|-----|----------|-------|---------|----------|------|
| 1220 | 3.13% | 39 | 0.0192ms | 0.0198ms | 1.03 | 0.0191ms | 0.0217ms | 1.14 |
| 4066 | 3.13% | 128 | 0.0198ms | 0.0271ms | 1.37 | 0.0199ms | 0.0271ms | 1.36 |
| 13553 | 3.13% | 424 | 0.0202ms | 0.0271ms | 1.34 | 0.0202ms | 0.0271ms | 1.34 |
| 45176 | 3.13% | 1412 | 0.0211ms | 0.0335ms | 1.59 | 0.0206ms | 0.0336ms | 1.63 |
| 150586 | 3.13% | 4706 | 0.0265ms | 0.0397ms | 1.5 | 0.0265ms | 0.0398ms | 1.5 |
| 501953 | 3.13% | 15687 | 0.0433ms | 0.0503ms | 1.16 | 0.041ms | 0.0581ms | 1.42 |
| 1673176 | 3.13% | 52287 | 0.102ms | 0.08ms | 0.786 | 0.0928ms | 0.0797ms | 0.859 |
| 5577253 | 3.13% | 174290 | 0.221ms | 0.166ms | 0.754 | 0.217ms | 0.162ms | 0.745 |
| 18590843 | 3.13% | 580964 | 0.634ms | 0.422ms | 0.666 | 0.617ms | 0.421ms | 0.683 |
| 61969476 | 3.13% | 1936547 | 2.03ms | 1.27ms | 0.625 | 1.98ms | 1.27ms | 0.642 |
| 206564920 | 3.13% | 6455154 | 6.72ms | 4.09ms | 0.608 | 6.65ms | 4.34ms | 0.653 |
| 1220 | 1.56% | 20 | 0.0192ms | 0.0199ms | 1.04 | 0.0192ms | 0.0218ms | 1.14 |
| 4066 | 1.56% | 64 | 0.026ms | 0.0313ms | 1.2 | 0.0272ms | 0.0312ms | 1.15 |
| 13553 | 1.56% | 212 | 0.0316ms | 0.0269ms | 0.853 | 0.0201ms | 0.0272ms | 1.35 |
| 45176 | 1.56% | 706 | 0.02ms | 0.0412ms | 2.06 | 0.0203ms | 0.0337ms | 1.66 |
| 150586 | 1.56% | 2353 | 0.0267ms | 0.0396ms | 1.49 | 0.0268ms | 0.0397ms | 1.48 |
| 501953 | 1.56% | 7844 | 0.0426ms | 0.0582ms | 1.37 | 0.0409ms | 0.0497ms | 1.22 |
| 1673176 | 1.56% | 26144 | 0.0925ms | 0.0744ms | 0.804 | 0.0923ms | 0.0743ms | 0.805 |
| 5577253 | 1.56% | 87145 | 0.217ms | 0.15ms | 0.693 | 0.217ms | 0.149ms | 0.685 |
| 18590843 | 1.56% | 290482 | 0.635ms | 0.405ms | 0.638 | 0.634ms | 0.398ms | 0.628 |
| 61969476 | 1.56% | 968274 | 2.02ms | 1.18ms | 0.586 | 1.98ms | 1.19ms | 0.599 |
| 206564920 | 1.56% | 3227577 | 6.61ms | 3.76ms | 0.569 | 6.55ms | 3.81ms | 0.581 |
| 1220 | 0.781% | 10 | 0.0241ms | 0.0202ms | 0.837 | 0.019ms | 0.0219ms | 1.15 |
| 4066 | 0.781% | 32 | 0.0198ms | 0.0266ms | 1.34 | 0.0246ms | 0.0272ms | 1.1 |
| 13553 | 0.781% | 106 | 0.0206ms | 0.0271ms | 1.32 | 0.02ms | 0.0336ms | 1.68 |
| 45176 | 0.781% | 353 | 0.0203ms | 0.0336ms | 1.66 | 0.0201ms | 0.0411ms | 2.04 |
| 150586 | 0.781% | 1177 | 0.0258ms | 0.0486ms | 1.88 | 0.0262ms | 0.0496ms | 1.89 |
| 501953 | 0.781% | 3922 | 0.0424ms | 0.0498ms | 1.17 | 0.0409ms | 0.0501ms | 1.23 |
| 1673176 | 0.781% | 13072 | 0.0924ms | 0.0708ms | 0.766 | 0.0919ms | 0.0704ms | 0.767 |
| 5577253 | 0.781% | 43573 | 0.216ms | 0.144ms | 0.666 | 0.209ms | 0.136ms | 0.652 |
| 18590843 | 0.781% | 145241 | 0.623ms | 0.373ms | 0.599 | 0.616ms | 0.376ms | 0.611 |
| 61969476 | 0.781% | 484137 | 2ms | 1.12ms | 0.563 | 2.03ms | 1.14ms | 0.562 |
| 206564920 | 0.781% | 1613789 | 6.59ms | 3.59ms | 0.545 | 6.54ms | 3.67ms | 0.562 |
| 1220 | 0.391% | 5 | 0.0267ms | 0.0299ms | 1.12 | 0.0284ms | 0.0296ms | 1.04 |
| 4066 | 0.391% | 16 | 0.0196ms | 0.0266ms | 1.36 | 0.0197ms | 0.0271ms | 1.38 |
| 13553 | 0.391% | 53 | 0.02ms | 0.0275ms | 1.38 | 0.0201ms | 0.0271ms | 1.35 |
| 45176 | 0.391% | 177 | 0.0204ms | 0.0337ms | 1.65 | 0.0203ms | 0.0339ms | 1.67 |
| 150586 | 0.391% | 589 | 0.0265ms | 0.0392ms | 1.48 | 0.0264ms | 0.0393ms | 1.49 |
| 501953 | 0.391% | 1961 | 0.0418ms | 0.0496ms | 1.19 | 0.0422ms | 0.057ms | 1.35 |
| 1673176 | 0.391% | 6536 | 0.0924ms | 0.0699ms | 0.756 | 0.0911ms | 0.0703ms | 0.772 |
| 5577253 | 0.391% | 21787 | 0.21ms | 0.133ms | 0.634 | 0.206ms | 0.134ms | 0.651 |
| 18590843 | 0.391% | 72621 | 0.622ms | 0.362ms | 0.582 | 0.609ms | 0.369ms | 0.606 |
| 61969476 | 0.391% | 242069 | 2ms | 1.09ms | 0.544 | 1.97ms | 1.13ms | 0.572 |
| 206564920 | 0.391% | 806895 | 6.59ms | 3.5ms | 0.531 | 6.47ms | 3.6ms | 0.557 |
