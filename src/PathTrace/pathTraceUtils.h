#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

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

#define COLOR_DEBUG (glm::vec3(1.0f, 0.0f, 1.0f)) 


__device__
inline bool epsilonCheck(float a, float b)
{
    return fabs(a - b) < EPSILON;
}


// Functions (tangent space, (tangent, bitangent, normal))
__device__
void coordinateSystem(glm::vec3 v1, glm::vec3& v2, glm::vec3& v3);

__device__
glm::mat3 localToWorld(glm::vec3 nor);

__device__
glm::mat3 worldToLocal(glm::vec3 nor);

__device__
bool sameHemisphere(glm::vec3 w, glm::vec3 wp);

