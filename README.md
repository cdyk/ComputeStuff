# ComputeStuff

MIT-licensed stand-alone CUDA utility functions.

The plan is to make various useful functions of mine available for the public under a liberal license. The functions are designed to have a minimum of dependencies, so integration is mostly just importing the relevant files into your project.

I've started with variants of scan, and the plan is to continue with HistoPyramid algorithms and Marching Cubes.

## Prefix sum / Scan

Implementation of
* exclusive scan,
* inclusive scan, and
* offset-table.

The offset-table  is an exclusive scan with the total sum appended to the output. In addition, the total sum can be outputed to a device pointer, which allows zero-copy of the total sum back to host memory.

The implementation has decent performance, the numbers below is a comparison with thrust's exclusive scan on an GTX 980 Ti. See code for details on the benchmark.

| N       | thrust | ComputeStuff | ratio|
|---------|--------|--------------|------|
|1        |0.022ms |0.006ms       |0.288 |
|3        |0.012ms |0.006ms       |0.518 |
|10       |0.012ms |0.011ms       |0.912 |
|33       |0.012ms |0.006ms       |0.516 |
|110      |0.012ms |0.006ms       |0.512 |
|366      |0.022ms |0.008ms       |0.361 |
|1220     |0.013ms |0.026ms       |2.101 |
|4066     |0.020ms |0.015ms       |0.768 |
|13553    |0.038ms |0.031ms       |0.827 |
|45176    |0.112ms |0.017ms       |0.150 |
|150586   |0.573ms |0.019ms       |0.033 |
|501953   |0.151ms |0.043ms       |0.286 |
|1673176  |0.648ms |0.102ms       |0.157 |
|5577253  |0.935ms |0.282ms       |0.302 |
|18590843 |1.986ms |0.890ms       |0.448 |
|61969476 |4.752ms |2.926ms       |0.616 |
|206564920|14.186ms|9.789ms       |0.690 |

