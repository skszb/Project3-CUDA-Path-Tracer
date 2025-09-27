#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "sceneStructs.h"

// Numeric constants
#define PI           3.14159265358979323846f
#define INV_PI       0.31830988618379067154f
#define INV_2PI      0.15915494309189533577f
#define INV_4PI      0.07957747154594766788f
#define PI_OVER_2    1.57079632679489661923f
#define PI_OVER_4    0.78539816339744830961f
#define SQRT_2       1.41421356237309504880f
#define TWO_PI       6.28318530717958647692f

#define SQRT_OF_ONE_THIRD   0.5773502691896257645091487805019574556476f

#define EPSILON      0.00001f



// Functions
__host__ __device__
void coordinateSystem(glm::vec3 v1, glm::vec3& v2, glm::vec3& v3);

__host__ __device__
glm::mat3 localToWorld(glm::vec3 nor);

__host__ __device__
glm::mat3 worldToLocal(glm::vec3 nor);
 


template <int V>
struct bounce_less_than
{
    __host__ __device__
    auto operator()(const PathSegment& seg) -> bool
    {
        return seg.remainingBounces < V;
    }
};

template <int V>
struct bounce_more_than
{
    __host__ __device__
        auto operator()(const PathSegment& seg) -> bool
    {
        return seg.remainingBounces > V;
    }
};

struct should_terminate_thread
{
    __host__ __device__
        auto operator()(const PathSegment& seg) -> bool
    {
        return seg.remainingBounces < 1;
    }
};