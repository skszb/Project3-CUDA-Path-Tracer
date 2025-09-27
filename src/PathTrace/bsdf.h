#pragma once

#include <cuda_runtime.h>
#include "sceneStructs.h"
#include <thrust/random.h>

#include "utilities.h"

enum class BxDFFlags
{
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    Diffuse = 1 << 2,
    Glossy = 1 << 3,
    Specular = 1 << 4,
};

/* -------- bsdf --------- */
__host__ __device__ glm::vec3 f_diffuse(glm::vec3 albedo);


__host__ __device__
glm::vec3 sample_diffuse(glm::vec3& wi_world, float& pdf, 
    const Material& material, const glm::vec3& nW, thrust::default_random_engine& rng);

__host__ __device__
glm::vec3 sample_specular(glm::vec3& wi_world, float& pdf, 
    const glm::vec3& wo_world, const Material& material, const glm::vec3& nW, thrust::default_random_engine& rng);
