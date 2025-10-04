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
    q.origin    =                multiplyMV(box.invTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.invTransform, glm::vec4(r.direction, 0.0f)));

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

    glm::vec3 ro = multiplyMV(sphere.invTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.invTransform, glm::vec4(r.direction, 0.0f)));

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
    normal = glm::normalize(multiplyMV(sphere.invTransform, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__device__ float triangleIntersectionTest(
    const Geom& geom, Ray r,
    glm::vec3 va, glm::vec3 vb, glm::vec3 vc, 
    glm::vec3& intersectionPoint, glm::vec3& barycentric)
{
    vec3 o = multiplyMV(geom.invTransform, glm::vec4(r.origin, 1.0f));
    vec3 D = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(r.direction, 0.0f)));
    vec3 E1 = vb - va;
    vec3 E2 = vc - va;
    vec3 D_cross_E2 = glm::cross(D, E2);

    float det = glm::dot(E1, D_cross_E2);
    if (epsilonCheck(det, 0.0f))
        return -1;

    float inv_det = 1.0f / det; // __fdividef(1, det);

    // u
    vec3 T = o - va;
    float u = glm::dot(T, D_cross_E2) * inv_det;
    if (u < 0.0f || u > 1.0f)
        return -1;

    // v
    vec3 Q = glm::cross(T, E1);
    float v = glm::dot(D, Q) * inv_det;
    if (v < 0.0f || (u+v) > 1.0f)
        return -1;

    // t
    float t = glm::dot(E2, Q) * inv_det;
    if(t < 0.0f)
        return -1;

    barycentric = vec3(1 - u - v, u, v);
    glm::vec3 objspaceIntersection = o + D * t;

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(objspaceIntersection, 1.f));

    return glm::length(r.origin - intersectionPoint);
}


__device__ float meshIntersectionTest(
    const Geom& geom,
    const glm::vec3* positions, int triangleCount,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float t_min = FLT_MAX;
    glm::vec3 barycentric;
    glm::vec3 vb_va;
    glm::vec3 vc_va;

    for (int tri = 0; tri < triangleCount; ++tri)
    {
        const glm::vec3 va = positions[3 * tri + 0];
        const glm::vec3 vb = positions[3 * tri + 1];
        const glm::vec3 vc = positions[3 * tri + 2];

        float tmp_t;
        glm::vec3 tmp_barycentric;
        glm::vec3 tmp_itsc;


        tmp_t = triangleIntersectionTest(geom, r, va, vb, vc, tmp_itsc, tmp_barycentric);
        if (tmp_t > 0 && tmp_t < t_min)
        {
            t_min = tmp_t;
            barycentric = tmp_barycentric;
            intersectionPoint = tmp_itsc;
            vb_va = vb - va;
            vc_va = vc - va;
        }
    }

    if (!(t_min < FLT_MAX))
        return -1;

    // flip normal to face the incident
    normal = glm::normalize(glm::cross(vb_va, vc_va));
    outside = glm::dot(normal, r.direction) < 0;

    // TODO: normal mapping

    if (!outside)
        normal = -normal;

    return glm::length(intersectionPoint - r.origin);
}


__device__ bool aabbIntersectionTest(const Geom& mesh, AABB aabb, Ray r)
{
    Ray rt {
        multiplyMV(mesh.invTransform, glm::vec4(r.origin, 1.0f)),
        glm::normalize(multiplyMV(mesh.invTransform, glm::vec4(r.direction, 0.0f)))
    };

    float tmin = -FLT_MAX;
    float tmax = FLT_MAX;

    for (int i = 0; i < 3; ++i)
    {
        float invD = 1.0f / rt.direction[i];
        float t0 = (aabb.min[i] - rt.origin[i]) * invD;
        float t1 = (aabb.max[i] - rt.origin[i]) * invD;
        if (invD < 0.0f)
        {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax < tmin)
            return false;
    }
    return true;
}