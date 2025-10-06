#pragma once

#include "sceneStructs.h"
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "bvh.h"


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + t * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}


// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);


// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);


/* ------------------------------ Mesh ------------------------------ */
// Test ray triangle
__device__ float triangleIntersectionTest(
    const Geom& geom, Ray r,
    glm::vec3 va, glm::vec3 vb, glm::vec3 vc,
    glm::vec3& intersectionPoint, glm::vec3& barycentric);

__device__ float meshIntersectionTest(
    const Geom& geom,
    const glm::vec3* positions, int triangleCount,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

__device__ bool aabbIntersectionTest(
    const Geom& mesh,
    AABB aabb,
    Ray r);

__device__ float bvhIntersectionTest(
    const Geom& geom,
    // bvh
    const NodeProxy* nodes,
    const glm::vec3* positions,
    const int* triangleIndices,

    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

__device__ float leafNodeIntersectionTest(
    const Geom& geom,
    const glm::vec3* positions,
    const int* triangleIndices,
    int triangleCount,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);


// objspace
__device__ float triangleIntersectionTest_ObjectSpace(
    const Geom& geom, Ray rt,
    glm::vec3 va, glm::vec3 vb, glm::vec3 vc, glm::vec3& barycentric);


__device__ float meshIntersectionTest_ObjectSpace(
    const Geom& geom,
    const glm::vec3* positions, int triangleCount,
    Ray r,
    glm::vec3& normal,
    bool& outside);