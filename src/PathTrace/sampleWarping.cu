#pragma once

#include "sampleWarping.h"
#include "sceneStructs.h"
#include "pathTraceUtils.h"


using glm::vec3;
using glm::vec2;


/* -------- math --------- */
__host__ __device__
vec3 squareToDiskConcentric(vec2 xi)
{
    vec2 remapXi = xi * 2.0f - vec2(1.0);

    float r;
    float theta;
    if (abs(remapXi.x) > abs(remapXi.y))
    {
        if (remapXi.x == 0.0)
        {
            return vec3(0);
        }
        r = remapXi.x;
        theta = remapXi.y / remapXi.x * PI / 4.0;
    }
    else
    {
        if (remapXi.y == 0.0)
        {
            return vec3(0);
        }
        r = remapXi.y;
        theta = (PI / 2) - (remapXi.x / remapXi.y * PI / 4.0);
    }
        
    return vec3(cos(theta) * r, sin(theta) * r, 0);
}

__host__ __device__
vec3 squareToHemisphereCosine(vec2 xi)
{
    vec3 d = squareToDiskConcentric(xi);
    float z = sqrt(max(0.0, 1.0 - d.x * d.x - d.y * d.y));
    return vec3(d.x, d.y, z);
}

__host__ __device__
float squareToHemisphereCosinePDF(vec3 sample)
{
    // sample.z is cosine theta (in object space)
    return sample.z * INV_PI;
}