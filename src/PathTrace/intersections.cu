#include "intersections.h"
#include "pathTraceUtils.h"

using glm::vec3;

__device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }
    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__device__ bool triangleIntersectionTest(glm::vec3& intersectionPoint, glm::vec3& normal, float& t, vec3& barycentric,
    const Geom& triangle,  Ray r,
    glm::vec3 va, glm::vec3 vb, glm::vec3 vc, const float t_min = EPSILON, const float t_max = 10000)
{
    vec3 o = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    vec3 D = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));
    vec3 E1 = vb - va;
    vec3 E2 = vc - va;
    vec3 D_cross_E2 = glm::cross(D, E2);

    float det = glm::dot(E1, D_cross_E2);
    if (epsilonCheck(det, 0.0f))
        return false;

    float inv_det = 1.0f / det; // __fdividef(1, det);

    // u
    vec3 T = o - va;
    float u = glm::dot(T, D_cross_E2) * inv_det;
    if (u < 0.0f || u > 1.0f)
        return false;

    // v
    vec3 Q = glm::cross(T, E1);
    float v = glm::dot(D, Q) * inv_det;
    if (v < 0.0f || (u+v) > 1.0f)
        return false;

    // t
    t = glm::dot(E2, Q) * inv_det;
    if(t < t_min || t > t_max)
        return false;

    barycentric = vec3(1 - u - v, u, v);
    intersectionPoint = o + t * D;
    normal = glm::cross(E1, E2);
    // TODO: normal mapping?

    // TODO: backface test
    return true;
}
    

    