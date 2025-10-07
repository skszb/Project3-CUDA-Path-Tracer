CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

- Zhanbo Lin
    - [LinkedIn](https://www.linkedin.com/in/zhanbo-lin)
- Tested on: Windows 10, i5-10400F @ 2.90GHz 48GB, RTX-3080 10GB (Personal) 
- GPU Compute Capability: 8.6

## Renders ##
![](./img/results/ShowCase1.png)

## Project Description ##



## Features 

<!--
List of core features completed https://github.com/CIS5650-Fall-2025/Project3-CUDA-Path-Tracer/blob/main/INSTRUCTION.md#part-1---core-features
-->
### Shading kernel with BSDF evaluation (diffuse, perfect specular surfaces)
![](./img/results/diffuse_specular.png)


### Visual Improvements
**Stochastic Sampled Antialiasing + Stratified Sampling**

![](./img/results/jitter/jitter-aa.png)
<sub>Jitter enabled</sub> 

![](./img/results/jitter/no-jitter.png)

<sub>Jitter disabled</sub> 



**Arbitrary mesh import using gltf**



### Performance Improvements

**BVH Acceleration**
![](./img/results/bvh/bvh.png)
<sub> bvh enabled</sub> 

![](./img/results/bvh/no_bvh.png)
<sub> naive mesh iterative triangle intersection </sub> 

Direct Mesh Render
With BVH


**Material Sorting**

It actually slows the render when enabled, might work better if we have more materials.


**Early-Out for Missed or Light Intersections**

It also actually slows the render when enabled, might work better in an opened scene.

![](./img/results/early_out/early_out.png)

![](./img/results/early_out/no_early_out.png)



### Bloopers
