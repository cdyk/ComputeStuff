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

## Introduction

The intent is to make various useful functions of mine available for the public under a liberal license --- so that anyone can freely use the code without any license issues.

The functions are designed to have a minimum of dependencies, so integration is mostly just adding the relevant files into your project.

I've started with variants of scan, and the plan is to continue with HistoPyramid algorithms and Marching Cubes.

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

The current implementation should be *correct*, but *is not optimized at all* (i.e., no multiple-level reductions, no constmem/texture sampler use, no dynamic parallelism), so it is actually slower than the scan-based compact in its current form. However, this is work in progress, so performance will improve.

Implementation of

* **compact:** Extract indices where the corresponding input has a non-zero value (a.k.a. subset, stream-compact and copy-if), and write the number of entries in the output somewhere in device memory.

Please consult [HP5/HP5.h](HP5/HP5.h) for details on the API entry points.
