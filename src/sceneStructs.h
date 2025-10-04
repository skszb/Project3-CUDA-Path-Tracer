#pragma once
#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH,
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

// TODO: Unused yet, for non-mesh primitives like spheres and cubes 
struct NativeGeom
{
    enum GeomType type;
    int materialId;

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Geom
{
    enum GeomType type; // TODO: separate from NativeGeom
    int materialId;
    int meshId; // only for mesh type

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 invTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    glm::vec3 color;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 forward;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    glm::vec3 throughput;   // the accumulated BSDF attenuation
    glm::vec3 color_debug;
    int pixelIndex;
    int remainingBounces;
    int prevMaterialID;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    int materialId;
    bool outside; 
};


/* -------------------------- BVH -------------------------- */
struct AABB
{
    glm::vec3 min;
    glm::vec3 max;
};

/* -------------------------- Mesh -------------------------- */
// For scene storage
struct Mesh
{
    Mesh() : id(-1)
    {
        bound.min = glm::vec3(FLT_MAX);
        bound.max = glm::vec3(FLT_MIN);
    }
    int id;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;

    std::vector<int> indices;

    AABB bound;
};




/* -------------------------- Light -------------------------- */
// temporary solution TODO zb: use own data instead
struct AreaLight
{
    Geom geom;
};

struct PointLight
{
    glm::vec3 translation;
};

class SceneLights
{
    std::vector<AreaLight> areaLights;
    std::vector<PointLight> pointLight;

};


