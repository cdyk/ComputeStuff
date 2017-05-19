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

