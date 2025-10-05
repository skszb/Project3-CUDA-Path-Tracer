#include "bsdf.h"


#include "sampleWarping.h"
#include "pathTraceUtils.h"
#include "sceneStructs.h"
#include "cuda_runtime_api.h"

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
float fresnelDielectricEval(float cosThetaI, float etaI, float etaT)
{
    // Compute cosThetaT using Snell's law>
    float sinThetaI = sqrt(max(0.f,1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // total internal reflection
    if (sinThetaT > 1.f) 
    { 
        return 1.f;
    }

    float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (Rparl * Rparl + Rperp * Rperp) * 0.5f;     
}

__device__
vec3 sample_specular_refract(vec3&debug, vec3& wi_world, float& pdf,  vec3 wo_world,
    const Material& material, vec3 normal_i, float etaI, float etaT)
{
    pdf = 1.0f;
    vec3 wo = glm::normalize(worldToLocal(normal_i) * wo_world);
    float cosThetaT = -wo.z;

    float etaRatio = etaT / etaI;

    float sin2ThetaT = max(0.f, 1.f - cosThetaT * cosThetaT);
    float sin2ThetaI = etaRatio * etaRatio * sin2ThetaT;

    // total internal reflection
    if (sin2ThetaI > 1.f) 
    {
        return vec3(0);
    }

    wi_world = glm::refract(-wo_world, normal_i, etaRatio);


    return material.color / glm::dot(wi_world, normal_i);
}

__device__ bool isNan(glm::vec3 x)
{
    return x.x!=x.x|| x.y != x.y|| x.z != x.z;
}

__device__ void btdfSample(glm::vec3& debug, bool outside, glm::vec3& outBSDF, float& outPDF, Ray& out_wi, glm::vec3 ViewDir, glm::vec3 p,
    glm::vec3 surfaceNormal, glm::vec2 uv, const Material& material,
    thrust::default_random_engine& rng)
{
    outBSDF = glm::vec3(1.0f);
    outPDF = 1.f;
    float ETA = 1.f / 1.5f;
    glm::vec3 wo_Tangent, tanX_World, tanZ_World, N  = surfaceNormal;
    tanZ_World = glm::normalize(glm::cross(ViewDir, N));
    tanX_World = glm::normalize(glm::cross(N, tanZ_World));
    wo_Tangent.x = glm::dot(ViewDir, tanX_World);
    wo_Tangent.z = glm::dot(ViewDir, tanZ_World);
    wo_Tangent.y = glm::dot(ViewDir, N);
    wo_Tangent = glm::normalize(wo_Tangent);
    float SinTheta1 = wo_Tangent.x;
    glm::vec3 wi_Tangent;
    float eta = outside ? ETA : 1.f / ETA;
    wi_Tangent.x = -eta * SinTheta1;
    if (abs(wi_Tangent.x)>=1.0f)
    {
        // retract a little bit so no self intersect
        out_wi.origin = p + 0.0001f * N;
        out_wi.direction = - glm::reflect(ViewDir, N);
        return;
    }
    wi_Tangent.y = -sqrt(1.f - wi_Tangent.x * wi_Tangent.x);
    wi_Tangent.z = 0.f;
    glm::vec3 wi_World = wi_Tangent.x * tanX_World + wi_Tangent.y * N + wi_Tangent.z * tanZ_World;
    out_wi.direction = wi_World;
    if (outside)
    {
        out_wi.origin = p - 0.001f * surfaceNormal; // minus at least this value
    }
    else
    {
        out_wi.origin = p + 0.001f * surfaceNormal;
    }
    //ViewDir = glm::normalize(ViewDir);
    //
    //glm::vec3 N = surfaceNormal;
    //N = glm::normalize(N);
    //// march a little bit when passing through surface
    //float ETA = 1.f / 1.5f;
    //if (outside)
    //{
    //    out_wi.direction = glm::refract(-ViewDir, N, ETA);
    //}
    //else
    //{
    //    out_wi.direction = glm::refract(-ViewDir, N, 1.f / ETA);
    //}
    //debug = ViewDir + 0.5f;
    //if (isNan(out_wi.direction) || glm::length(out_wi.direction) == 0.f)
    //{
    //    // retract a little bit so no self intersect
    //    out_wi.direction = -glm::reflect(ViewDir, N);
    //}

}

__device__
glm::vec3 sample_f(glm::vec3& debug, glm::vec3& wi_world, float& pdf,
                              glm::vec3 wo_world, const Material& material, glm::vec3 normal_world, bool outside, 
    thrust::default_random_engine& rng)
{
    if (material.hasReflective)
    {
        return sample_specular_reflect(wi_world, pdf, wo_world, material, normal_world);
    }
    else if (material.hasRefractive)
    {
#if 0
        pdf = 1;
        // acquire the direction of the ray
        const glm::vec3 direction{ -glm::normalize(wo_world) };

        // generate a random decimal
        thrust::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        float random_decimal{ distribution(rng) };

        float etaI = outside ? material.indexOfRefraction : 1.0f;
        float etaT = outside ? 1.0f : material.indexOfRefraction;

        vec3 refr = glm::refract(-wo_world, normal_world, etaT / etaI);
        float fresnel;
        if (glm::length(refr) < 0.1f)
        {
            refr = glm::reflect(-wo_world, normal_world);
            fresnel = 1;
        }
        else
        {
            /*if (!outside)
            {
                etaI = material.indexOfRefraction;
                etaT = 1.0f;
            }*/
            float cos = glm::abs(glm::dot(refr, normal_world));
            float R0 = (etaI - etaT) / (etaI + etaT);
            R0 = R0 * R0;
            fresnel = R0 + (1 - R0) * pow(1 - cos, 5);
        }

        vec3 spectrum;
        // perform refraction when the Fresnel factor is small
        if (random_decimal > 0)
        {
            wi_world = refr;
            spectrum = 2.0f * material.color * (1- fresnel) / glm::abs(glm::dot(wi_world, normal_world));
        }
        /*else
        {
            wi_world = glm::reflect(direction, normal_world);
            spectrum = 2.0f *  material.color * fresnel / glm::abs(glm::dot(wi_world, normal_world));
        }*/
        return spectrum;
#else
        vec3 f;
        Ray r;
        vec3 itsc;
        btdfSample(debug, outside, f, pdf, r, wo_world, itsc, normal_world, vec2(0), material, rng);

        pdf = 1;
        wi_world = r.direction;
        return f / glm::abs(glm::dot(normal_world, r.direction));
#endif

    }
    else
    {
        return sample_diffuse(wi_world, pdf, material, normal_world, rng);
    }

    return glm::vec3(1);
}