#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
public:
    Scene() = default;
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    std::vector<AABB> meshBounds;

    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfName, bool fullReload);

    RenderState state;
};
