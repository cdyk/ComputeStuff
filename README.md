# ComputeStuff

MIT-licensed stand-alone CUDA utility functions.

* [Introduction](#introduction)
  * [How to use](#how-to-use)
    * [Scratch buffer](#scratch-buffer)
    * [In-place operation](#in-place-operation)
    * [Write sum](#write-sum)
    * [Concurrent invocations](#concurrent-invocations)
* [Prefix sum / Scan](#prefix-sum-Scan)
  * [Performance](#performance)
* [5:1 HistoPyramids](#histoPyramids)
  * [Performance](#performance-1)
* [Credits](#credits)
* [License](#license)


## Introduction

The intent is to make various useful functions of mine available for the public under a liberal license --- so that anyone can freely use the code without any license issues.

The functions are designed to have a minimum of dependencies, so integration is mostly just adding the relevant files into your project.

I've started with variants of scan and the 5-to-1 HistoPyramid, and will continue with HistoPyramid variations and Marching Cubes.

### How to use

Each of the components is fully contained in its own project (currently [Scan project](Scan/Scan.vcxproj) and [HP5 project](HP5/HP5.vcxproj)). To use the code, either link against the static library produced by the project you want to use, or just add the files in the project to your project.

Typically these projects contains a header file with the public API and one or more source files with the implementation.

In addition, each component have a test project (currently [ScanTest project](ScanTest/ScanTest.vcxproj) and [HP5Test project](HP5Test/HP5Test.vcxproj)) that serves as a combined unit and performance test as well as an example of use.

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

The implementation has decent performance, the numbers below is a comparison with thrust's exclusive scan on an GTX 980 Ti. See [ScanTest/main.cu](ScanTest/main.cu) for details.

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

## 5:1 HistoPyramids

The 5-to-1 HistoPyramid is a variation of the traditional 4-to-1 HistoPyramid (or just HistoPyramid) that, instead of reducing four elements into one, it reduces five elements, hence the name.

The details are explained in [GPU Accelerated Data Expansion Marching Cubes Algorithm](http://on-demand.gputechconf.com/gtc/2010/presentations/S12020-GPU-Accelerated-Data-Expansion-Marching-Cubes-Algorithm.pdf) from GTC 2010.


Implementation of

* **compact:** Extract indices where the corresponding input has a non-zero value (a.k.a. subset, stream-compact and copy-if), and write the number of entries in the output somewhere in device memory.

Please consult [HP5/HP5.h](HP5/HP5.h) for details on the API entry points.

### Performance

The implementation starts to become decent. The numbers below is a comparsion to ComputeStuff's own scan implementation, which is faster than thrust (see [Scan performance](#performance)), and was run on an GTX 980 Ti.

Ratio is the time HP5 uses compared to Scan (lower is better for HP5's case). The test runs compact on N elements where "% selected" of the elements are marked for selection. In the min-scatter scenario, all selected elements are in the start of the buffer, which is the best-case for HP cache utilization, while max-scatter evenly spaces the selected element out, which is the worst-case for HP cache utilization. Please consult [HP5Test/main.cu](HP5Test/main.cu) for further details.



| N    | % selected| min-scatter scan | min-scatter hp5    |min-scatter ratio| max-scatter scan | max-scatter hp5    | max-scatter ratio|
|------|-------|---------|----------|-------|---------|----------|------|
| 1000 | 3.13% | 0.0197ms | 0.0214ms | 1.09 | 0.0192ms | 0.0213ms | 1.11 |
| 3333 | 3.13% | 0.0195ms | 0.0199ms | 1.02 | 0.0199ms | 0.0215ms | 1.08 |
| 11110 | 3.13% | 0.02ms | 0.0391ms | 1.95 | 0.0206ms | 0.0315ms | 1.53 |
| 37033 | 3.13% | 0.0205ms | 0.0375ms | 1.83 | 0.0207ms | 0.0377ms | 1.82 |
| 123443 | 3.13% | 0.0266ms | 0.0487ms | 1.83 | 0.0266ms | 0.0442ms | 1.66 |
| 411476 | 3.13% | 0.0451ms | 0.0442ms | 0.98 | 0.0403ms | 0.0441ms | 1.09 |
| 1371586 | 3.13% | 0.0831ms | 0.0763ms | 0.918 | 0.0829ms | 0.0763ms | 0.92 |
| 4571953 | 3.13% | 0.187ms | 0.157ms | 0.841 | 0.185ms | 0.159ms | 0.858 |
| 15239843 | 3.13% | 0.535ms | 0.398ms | 0.744 | 0.527ms | 0.397ms | 0.754 |
| 50799476 | 3.13% | 1.69ms | 1.14ms | 0.675 | 1.63ms | 1.14ms | 0.703 |
| 169331586 | 3.13% | 5.48ms | 3.62ms | 0.661 | 5.39ms | 3.74ms | 0.694 |
| 1000 | 1.56% | 0.0191ms | 0.0226ms | 1.18 | 0.0208ms | 0.0231ms | 1.11 |
| 3333 | 1.56% | 0.0199ms | 0.0213ms | 1.07 | 0.0208ms | 0.022ms | 1.06 |
| 11110 | 1.56% | 0.0199ms | 0.0308ms | 1.54 | 0.0202ms | 0.0316ms | 1.57 |
| 37033 | 1.56% | 0.0204ms | 0.0422ms | 2.07 | 0.0215ms | 0.0394ms | 1.83 |
| 123443 | 1.56% | 0.0271ms | 0.0443ms | 1.63 | 0.0269ms | 0.0455ms | 1.69 |
| 411476 | 1.56% | 0.0471ms | 0.0457ms | 0.971 | 0.0396ms | 0.044ms | 1.11 |
| 1371586 | 1.56% | 0.0827ms | 0.0749ms | 0.906 | 0.0824ms | 0.0746ms | 0.906 |
| 4571953 | 1.56% | 0.186ms | 0.148ms | 0.795 | 0.187ms | 0.152ms | 0.811 |
| 15239843 | 1.56% | 0.528ms | 0.373ms | 0.708 | 0.526ms | 0.379ms | 0.722 |
| 50799476 | 1.56% | 1.67ms | 1.09ms | 0.654 | 1.67ms | 1.12ms | 0.668 |
| 169331586 | 1.56% | 5.44ms | 3.39ms | 0.622 | 5.41ms | 3.46ms | 0.64 |
| 1000 | 0.781% | 0.0635ms | 0.0694ms | 1.09 | 0.0628ms | 0.0668ms | 1.06 |
| 3333 | 0.781% | 0.0193ms | 0.0201ms | 1.04 | 0.0187ms | 0.02ms | 1.07 |
| 11110 | 0.781% | 0.0194ms | 0.0273ms | 1.41 | 0.0196ms | 0.0281ms | 1.43 |
| 37033 | 0.781% | 0.0185ms | 0.0367ms | 1.98 | 0.0185ms | 0.0333ms | 1.8 |
| 123443 | 0.781% | 0.0238ms | 0.039ms | 1.63 | 0.0271ms | 0.0392ms | 1.45 |
| 411476 | 0.781% | 0.0363ms | 0.0398ms | 1.1 | 0.0357ms | 0.0393ms | 1.1 |
| 1371586 | 0.781% | 0.0821ms | 0.232ms | 2.83 | 0.0768ms | 0.0682ms | 0.889 |
| 4571953 | 0.781% | 0.18ms | 0.135ms | 0.752 | 0.178ms | 0.139ms | 0.78 |
| 15239843 | 0.781% | 0.523ms | 0.358ms | 0.684 | 0.526ms | 0.359ms | 0.683 |
| 50799476 | 0.781% | 1.66ms | 1.04ms | 0.627 | 1.66ms | 1.05ms | 0.633 |
| 169331586 | 0.781% | 5.42ms | 3.25ms | 0.6 | 5.39ms | 3.34ms | 0.62 |
| 1000 | 0.391% | 0.0203ms | 0.0205ms | 1.01 | 0.0198ms | 0.0188ms | 0.947 |
| 3333 | 0.391% | 0.0202ms | 0.0194ms | 0.961 | 0.0209ms | 0.0204ms | 0.975 |
| 11110 | 0.391% | 0.0193ms | 0.0276ms | 1.43 | 0.0189ms | 0.0381ms | 2.02 |
| 37033 | 0.391% | 0.0183ms | 0.0339ms | 1.85 | 0.0194ms | 0.0339ms | 1.74 |
| 123443 | 0.391% | 0.0237ms | 0.0388ms | 1.64 | 0.0244ms | 0.0404ms | 1.66 |
| 411476 | 0.391% | 0.0373ms | 0.0398ms | 1.07 | 0.0379ms | 0.044ms | 1.16 |
| 1371586 | 0.391% | 0.078ms | 0.0688ms | 0.881 | 0.0776ms | 0.0681ms | 0.878 |
| 4571953 | 0.391% | 0.179ms | 0.134ms | 0.749 | 0.176ms | 0.137ms | 0.783 |
| 15239843 | 0.391% | 0.516ms | 0.335ms | 0.65 | 0.506ms | 0.347ms | 0.686 |
| 50799476 | 0.391% | 1.66ms | 1.01ms | 0.61 | 1.63ms | 1.04ms | 0.639 |
| 169331586 | 0.391% | 5.41ms | 3.19ms | 0.589 | 5.3ms | 3.3ms | 0.623 |

## Credits

The ComputeStuff implementation was initially written and is maintained by Christopher Dyken, with contributions from Gernot Ziegler.

## License

ComputeStuff is licensed under the MIT license, please see [LICENSE](LICENSE) for more information.