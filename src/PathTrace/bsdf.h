#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <thrust/random.h>
struct Material;


/* -------- bsdf --------- */
__device__ glm::vec3 sample_f(glm::vec3& debug, glm::vec3& wi_world, float& pdf,
                              glm::vec3 wo_world, const Material& material, glm::vec3 normal_world, bool outside, thrust::default_random_engine& rng);