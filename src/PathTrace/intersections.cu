#include "intersections.h"
#include "pathTraceUtils.h"

using glm::vec3;
using glm::vec4;

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


/* ------------------------------------------------------------ Mesh ------------------------------------------------------------ */
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

    if (t_min == FLT_MAX)
        return -1;

    // flip normal to face the incident
    normal = glm::normalize(multiplyMV(geom.invTranspose, vec4(glm::cross(vb_va, vc_va), 0.0f)));
    outside = glm::dot(normal, r.direction) < 0;

    // TODO: normal mapping

    if (!outside)
        normal = -normal;

    return glm::length(intersectionPoint - r.origin);
}


__device__ bool aabbIntersectionTest(const Geom& mesh, AABB aabb, Ray r)
{
    vec3 oT = multiplyMV(mesh.invTransform, glm::vec4(r.origin, 1.0f));
    vec3 dirT = glm::normalize(multiplyMV(mesh.invTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -FLT_MAX;
    float tmax = FLT_MAX;

    for (int i = 0; i < 3; ++i)
    {
        float invD = 1.0f / dirT[i];
        float t0 = (aabb.min[i] - oT[i]) * invD;
        float t1 = (aabb.max[i] - oT[i]) * invD;
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


__device__ float leafNodeIntersectionTest(
    const Geom& geom,
    const glm::vec3* positions, 
    const int* triangleIndices,
    int triangleCount,
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
        int triangleBufferIdx = triangleIndices[tri];
        const glm::vec3 va = positions[triangleBufferIdx + 0];
        const glm::vec3 vb = positions[triangleBufferIdx + 1];
        const glm::vec3 vc = positions[triangleBufferIdx + 2];

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

    if (t_min == FLT_MAX)
        return -1;

    // flip normal to face the incident
    normal = glm::normalize(glm::cross(vb_va, vc_va));
    outside = glm::dot(normal, r.direction) < 0;

    // TODO: normal mapping

    if (!outside)
        normal = -normal;

    return glm::length(intersectionPoint - r.origin);
}


__device__ float bvhIntersectionTest(
    const Geom& geom,
    // bvh
    const NodeProxy* nodes,
    const glm::vec3* positions,
    const int* triangleIndices,

    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    const int STACK_LIMIT  = 32;
    int stack[STACK_LIMIT ];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    float tMin = FLT_MAX;
    glm::vec3 bestP, bestN;
    bool bestOutside = false;

    while (stackPtr > 0)
    {
        int nodeIdx = stack[--stackPtr];
        const NodeProxy& nd = nodes[nodeIdx];

        bool hitAABB = aabbIntersectionTest(geom, nd.bound, r);
        if (!hitAABB)
            continue;

        if (nd.triangleCount > 0)
        {
            // Leaf node
            const int* trisBegin = triangleIndices + nd.triangleBufferOffset;
            glm::vec3 tmpP, tmpN;
            bool tmpOutside;
            float t = leafNodeIntersectionTest(
                geom, positions, trisBegin, nd.triangleCount, r,
                tmpP, tmpN, tmpOutside);

            if (t > 0.f && t < tMin)
            {
                tMin = t;
                bestP = tmpP;
                bestN = tmpN;
                bestOutside = tmpOutside;
            }
        }
        else
        {
            // Internal node - push children
            const int left = nd.leftChildIdx;
            const int right = nd.rightChildIdx;

            if (left != -1 && stackPtr < STACK_LIMIT - 1)
                stack[stackPtr++] = left;
            if (right != -1 && stackPtr < STACK_LIMIT -1)
                stack[stackPtr++] = right;
        }
    }

    if (tMin != FLT_MAX)
    {
        intersectionPoint = bestP;
        normal = bestN;
        outside = bestOutside;
        return tMin;
    }
    return -1.f;
}


// Object space
__device__ float triangleIntersectionTest_ObjectSpace(
    const Geom& geom, Ray rt,
    glm::vec3 va, glm::vec3 vb, glm::vec3 vc, glm::vec3& barycentric)
{
    vec3 E1 = vb - va;
    vec3 E2 = vc - va;
    vec3 D_cross_E2 = glm::cross(rt.direction, E2);

    float det = glm::dot(E1, D_cross_E2);
    if (epsilonCheck(det, 0.0f))
        return -1;

    float inv_det = 1.0f / det; // __fdividef(1, det);

    // u
    vec3 T = rt.origin - va;
    float u = glm::dot(T, D_cross_E2) * inv_det;
    if (u < 0.0f || u > 1.0f)
        return -1;

    // v
    vec3 Q = glm::cross(T, E1);
    float v = glm::dot(rt.direction, Q) * inv_det;
    if (v < 0.0f || (u + v) > 1.0f)
        return -1;

    // t
    float t = glm::dot(E2, Q) * inv_det;
    if (t < 0.0f)
        return -1;

    barycentric = vec3(1 - u - v, u, v);

    return t;
}

__device__ float meshIntersectionTest_ObjectSpace(
    const Geom& geom,
    const glm::vec3* positions, int triangleCount,
    Ray r,
    glm::vec3& normal,
    bool& outside)
{
    float t_min = FLT_MAX;
    vec3 barycentric;
    vec3 vb_va;
    vec3 vc_va;

    for (int tri = 0; tri < triangleCount; ++tri)
    {
        const glm::vec3 va = positions[3 * tri + 0];
        const glm::vec3 vb = positions[3 * tri + 1];
        const glm::vec3 vc = positions[3 * tri + 2];

        float tmp_t;
        vec3 tmp_barycentric;
        vec3 tmp_itsc;

        tmp_t = triangleIntersectionTest_ObjectSpace(geom, r, va, vb, vc, tmp_barycentric);
        if (tmp_t > 0 && tmp_t < t_min)
        {
            t_min = tmp_t;
            barycentric = tmp_barycentric;
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

    return t_min;
}