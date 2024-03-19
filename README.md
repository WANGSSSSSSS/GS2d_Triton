# GS2d_Triton

Gaussian Splating 2d implemented in triton

Cause I am not very familiar with **GPU Arch**, so the inference performance is suboptimal.

but the memory reduction is substansial! so i am very happy to share this repo to public.

### Introduction

Currently, when rendering **100 256x256 images with 100 x 2000 GS**, my impl only consumes 1.2 GB on 2080ti, and after some helps from  my friend, the slow speed has been solved, my impl achieves **0.02s/iter**, (forward, backward, optimize step) . Because i tagged `num_renders` with `tl.constexpr` , but `num_renders` changes along with training, `triton.jit` will compile every related functions when function calls happened. 

So feel free to use it! And leaving a star helps my job seeking(maybe) 

Feel free to leave some suggestion to further improve my codebase....

### Some Rendered Images:

with rotation matrix:

![](assets/rot.png)

without GS prob:

![](./assets/no_gs.jpg)
