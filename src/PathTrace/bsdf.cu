#include "bsdf.h"

#include "intersections.h"
#include "sampleWarping.h"
#include "pathTraceUtils.h"

using glm::vec3;
using glm::vec2;

__host__ __device__ vec3 f_diffuse(vec3 albedo)
{
    return albedo * INV_PI;
}

__host__ __device__
vec3 sample_diffuse(vec3& wi_world, float& pdf, 
    const Material& material, const vec3& nW, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    vec2 xi{ u01(rng), u01(rng) };

    // Generate a random direction in the hemisphere using cosine-weighted sampling
    vec3 dirN = squareToHemisphereCosine(xi);
    pdf = max(EPSILON, squareToHemisphereCosinePDF(dirN)); 
    wi_world = localToWorld(nW) * dirN;

    return f_diffuse(material.color);
}

__host__ __device__
vec3 sample_specular(vec3& wi_world, float& pdf, const vec3& wo_world,
    const Material& material, const vec3& nW, thrust::default_random_engine& rng)
{
    wi_world = glm::reflect(-wo_world, nW);
    pdf = 1.0f;

    return material.color * glm::dot(nW, wi_world);
}

