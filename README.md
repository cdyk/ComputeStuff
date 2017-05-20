# ComputeStuff

MIT-licensed stand-alone CUDA utility functions.

The intent is to make various useful functions of mine available for the public under a liberal license --- so that anyone can freely use the code without any license issues.

The functions are designed to have a minimum of dependencies, so integration is mostly just adding the relevant files into your project.

I've started with variants of scan, and the plan is to continue with HistoPyramid algorithms and Marching Cubes.

## Prefix sum / Scan

Implementation of

* **exclusive scan:** From [a,b,c,d] calculate [0, a, a+b, a+b+c] (a.k.a exclusive prefix-sum).
* **inclusive scan:** From [a,b,c,d] calculate [a, a+b, a+b+c, a+b+c+d] (a.k.a inclusive prefix-sum).
* **offset-table:** Calculate an offset-table (exclusive prefix-sum with total concatenated on end), and optionally write the total sum somewhere in device memory. From [a,b,c,d] calculate [0, a, a+b, a+b+c, a+b+c+d], so that element _i_ contains the offset _i_ and the difference betweem elements _i_ and _i+1_ is the corresponding value that was in the input array. 
* **compact:** Extract indices where the corresponding input has a non-zero value (a.k.a. subset, stream-compact and copy-if), and write the number of entries in the output somewhere in device memory.

Please consult [Scan/Scan.h](Scan/Scan.h) for details on the API entry points.

### How to use

Either link against the static library built by the [Scan project](Scan/Scan.vcxproj) in the solution, or just add the files [Scan/Scan.h](Scan/Scan.h) and [Scan/Scan.cu](Scan/Scan.cu) to your project.

The file [Scan/Scan.h](Scan/Scan.h) contains the public API, and [Scan/Scan.cu](Scan/Scan.cu) contains the implementation.

#### Scratch buffer

All the scan functions need a scratch buffer of device memory to store some intermediate data. The size of this buffer is given by `ComputeStuff::Scan::scratchByteSize(N)`. It is the application developer's responsibility to create this buffer, e.g.,

````c++
uint32_t*scratch_d;
cudaMalloc(&scratch_d, ComputeStuff::Scan::scratchByteSize(N));
````

This let you have total control of memory management, so no strange malloc's or copy's is injected behind the scenes.

This buffer can be recycled between invocations that don't overlap, that is, they don't run at the same time on different streams --- in that case you need multiple scratch buffers.

Also, the size of this buffer grows monotonically with N, so if you are going to run several scans for different Ns, just use the largest N to determine the scratch buffer size, and you should be good.

#### In-place operation

All scan functions supports (without any penalty) in-place operation, that is using the same buffer for input and output.
````c++
// inout_d contains the input of N elements
ComputeStuff::Scan::exclusiveScan(inout_d, scratch_d, inout_d, N);
// inout_d contains the output of N elements
````

Note that for the offsetTable-functions, the output is of size N+1, while the input is of size N, so the buffer must be large enough.


#### Write sum

The offsetTable-function can optionally write the total sum of the input to anywhere in device memory. This is useful for zero-copying the result back to the CPU, if some subseqent code (like a grid size) is dependent on this number. An example:

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

All API functions have an optional stream argument, which, not surprisingly, will insert the GPU operations on that given stream. Also, since no global state is used inside the function, it should be safe to run several scans simultaneously on different streams --- as long as they have separate scratch buffers.


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

