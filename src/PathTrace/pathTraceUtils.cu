#include "pathTraceUtils.h"

__device__
void coordinateSystem(glm::vec3 v1, glm::vec3& v2, glm::vec3& v3)
{
    if (abs(v1.x) > abs(v1.y))
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = glm::cross(v1, v2);
}

__device__
glm::mat3 localToWorld(glm::vec3 nor)
{
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3{ tan, bit, nor };
}

__device__
glm::mat3 worldToLocal(glm::vec3 nor)
{
    return glm::transpose(localToWorld(nor));
}

__device__ float absCosTheta(glm::vec3 wi)
{
    return glm::abs(wi.z);
}
