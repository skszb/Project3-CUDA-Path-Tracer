#include "bsdf.h"


#include "sampleWarping.h"
#include "pathTraceUtils.h"
#include "sceneStructs.h"

using glm::vec3;
using glm::vec2;

__device__ vec3 f_diffuse(vec3 albedo)
{
    return albedo * INV_PI;
}

__device__
vec3 sample_diffuse(vec3& wi_world, float& pdf, 
    const Material& material, vec3 normal_world, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    vec2 xi{ u01(rng), u01(rng) };

    // Generate a random direction in the hemisphere using cosine-weighted sampling
    vec3 dirN = squareToHemisphereCosine(xi);
    pdf = max(EPSILON, squareToHemisphereCosinePDF(dirN)); 
    wi_world = localToWorld(normal_world) * dirN;

    return f_diffuse(material.color);
}

// both wi_world and wo_world towards away from the surface
__device__
vec3 sample_specular_reflect(vec3& wi_world, float& pdf, vec3 wo_world,
    const Material& material, vec3 normal_world)
{
    wi_world = -wo_world + 2.0f * glm::dot(wo_world, normal_world) * normal_world;
    pdf = 1.0f;

    return material.color / glm::dot(wi_world, normal_world);
}

__device__
vec3 sample_specular_refract(vec3&debug, vec3& wi_world, float& pdf,  vec3 wo_world,
    const Material& material, vec3 normal_world, float etaI, float etaT)
{

    pdf = 1.0f;
    vec3 wo = glm::normalize(worldToLocal(normal_world) * wo_world);
    float cosThetaT = wo.z;

    float etaRatio = etaT / etaI;

    float sin2ThetaT = max(0.f, 1.f - cosThetaT * cosThetaT);
    float sin2ThetaI = etaRatio * etaRatio * sin2ThetaT;

    // total internal reflection
    if (sin2ThetaI > 1.f) 
    {
        return vec3(0);
    }

    wi_world = glm::refract(-wo_world, normal_world, etaRatio);


    return material.color / glm::dot(wi_world, normal_world);
}

__device__ inline float schlickFresnel(float cosI, float etaI, float etaT)
{
    float R0 = (etaI - etaT) / (etaI + etaT);
    R0 *= R0;
    return R0 + (1.f - R0) * powf(1.f - cosI, 5.f);
}

__device__
glm::vec3 sample_f(glm::vec3& debug, glm::vec3& wi_world, float& pdf,
                              glm::vec3 wo_world, const Material& material, glm::vec3 normal_world, bool outside, thrust::default_random_engine& rng)
{
    if (material.hasReflective)
    {
        return sample_specular_reflect(wi_world, pdf, wo_world, material, normal_world);
    }
    else if (material.hasRefractive)
    {
        return material.color;
    }
    else
    {
        return sample_diffuse(wi_world, pdf, material, normal_world, rng);
    }

    // return glm::vec3(1);
}