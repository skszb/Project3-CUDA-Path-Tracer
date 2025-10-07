#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "tiny_gltf.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <glm/gtc/matrix_transform.hpp>

using std::string;
using std::cout;
using std::endl;
using json = nlohmann::json;
using glm::vec3;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = true;
        }
        else if (p["TYPE"] == "Transparent")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = true;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialId = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.invTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (pi / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / pi;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.forward, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.forward = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}


/* ------------------------------------------------ GLTF ------------------------------------------------ */
// Helpers for converting gltf to scene data
struct BufferOffsets
{
    int material;
    int texture;
    int mesh;
    int instance;
};

glm::mat4 GetNodeTransform(const tinygltf::Node& node);

void ProcessNode(const tinygltf::Model& model, int nodeIdx, const glm::mat4& parentTransform,
    const BufferOffsets& bufferOffsets, const std::map<int, std::map<int, int> > gltfMeshPrimitiveIdLut,
    std::vector<Geom>& geomInstances);


// Load scene from a glTF file
void Scene::loadFromGLTF(const std::string& gltfName, bool fullReload)
{
    using tinygltf::Model;
    using tinygltf::TinyGLTF;

    Model model;
    TinyGLTF loader;
    std::string err;
    std::string warn;

    // Parse gltf scene
    bool parseComplete = false;
    string ext = utilityCore::GetFilePathExtension(gltfName);
    if (ext == "gltf")
        parseComplete = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);
    else if (ext == "glb")
        parseComplete = loader.LoadBinaryFromFile(&model, &err, &warn, gltfName);
    else
    {
        std::cerr << "Invalid file extension, need \"gltf\" or \"glb\".\n";
        return;
    }

    if (!warn.empty()) std::cout << "Warn: " << warn << std::endl;
    if (!err.empty()) std::cerr << "Err: " << err << std::endl;
    if (!parseComplete) throw std::runtime_error("Failed to parse glTF");

    // Clear existing scene data if doing a full reload, otherwise record the current sizes to offset new data ids
    BufferOffsets bufferBeginOffsets;
    if (fullReload)
    {
        meshes.clear();
        geoms.clear();
        materials.clear();
    }

    bufferBeginOffsets.mesh = meshes.size();
    bufferBeginOffsets.instance = geoms.size();
    bufferBeginOffsets.material = materials.size();
    //bufferOffsets.texture = textures.size();

    // we store each gltf primitive as a separate mesh in Scene, so we need to convert gltf mesh/primitive ids to scene mesh ids
    std::map<int, std::map<int, int>> gltfMeshPrimitiveIdLut;  

    // Convert to scene data
    // Materials
    if (model.materials.size() > 0)
    {
        for (const auto& mat : model.materials)
        {
            Material newMaterial{ glm::vec3(0), 0, 0, 1, 0, };
            //newMaterial.emittance = mat.emissiveFactor;
            newMaterial.color = glm::vec3(
                mat.pbrMetallicRoughness.baseColorFactor[0],
                mat.pbrMetallicRoughness.baseColorFactor[1],
                mat.pbrMetallicRoughness.baseColorFactor[2]);

            materials.push_back(newMaterial);
        }
    }
    else
    {
        // Add a default material if none exist
        if (bufferBeginOffsets.material == 0)
        {
            Material newMaterial;
            newMaterial.color = glm::vec3(0.5f, 0.5f, 0.5f);
            newMaterial.emittance = false;
            materials.push_back(newMaterial);
        }
    }
    

    // TODO: Textures

    // Meshes
    for (int mid = 0; mid < model.meshes.size(); mid++)
    {
        gltfMeshPrimitiveIdLut[mid] = std::map<int, int>();

        const auto& mesh = model.meshes[mid];
        for (int pid = 0; pid < mesh.primitives.size(); pid++)
        {
            Mesh newMesh;
            newMesh.id = meshes.size();
            gltfMeshPrimitiveIdLut[mid][pid] = newMesh.id;

            const auto& primitive = mesh.primitives[pid];

            // Positions and AABB
            if (primitive.attributes.count("POSITION"))
            {
                const auto& accessor = model.accessors[primitive.attributes.at("POSITION")];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                const float* positions = reinterpret_cast<const float*>(
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; ++i)
                {
                    vec3 vertexPos {
                        positions[i * 3 + 0],
                        positions[i * 3 + 1],
                        positions[i * 3 + 2] };

                    newMesh.positions.emplace_back(vertexPos);

                    newMesh.bound.max = glm::max(newMesh.bound.max, vertexPos);
                    newMesh.bound.min = glm::min(newMesh.bound.min, vertexPos);
                }
            }
            // Normals
            if (primitive.attributes.count("NORMAL"))
            {
                const auto& accessor = model.accessors[primitive.attributes.at("NORMAL")];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                const float* normals = reinterpret_cast<const float*>(
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; ++i)
                {
                    newMesh.normals.emplace_back(
                        normals[i * 3 + 0],
                        normals[i * 3 + 1],
                        normals[i * 3 + 2]
                    );
                }
            }
            // UVs
            if (primitive.attributes.count("TEXCOORD_0"))
            {
                const auto& accessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                const float* uvs = reinterpret_cast<const float*>(
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; ++i)
                {
                    newMesh.uvs.emplace_back(
                        uvs[i * 2 + 0],
                        uvs[i * 2 + 1]
                    );
                }
            }
            // Indices
            if (primitive.indices >= 0)
            {
                const auto& accessor = model.accessors[primitive.indices];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                {
                    const uint16_t* indices = reinterpret_cast<const uint16_t*>(
                        &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                    for (size_t i = 0; i < accessor.count; ++i)
                        newMesh.indices.push_back(indices[i]);
                }
                else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                {
                    const uint32_t* indices = reinterpret_cast<const uint32_t*>(
                        &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                    for (size_t i = 0; i < accessor.count; ++i)
                        newMesh.indices.push_back(indices[i]);
                }
            }

            meshes.push_back(newMesh);
        }
    }

    // Geom instances
    if (!model.scenes.empty())
    {
        const auto& scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
        for (int nodeIdx : scene.nodes)
        {
            ProcessNode(model, nodeIdx, glm::mat4(1.0f), bufferBeginOffsets, gltfMeshPrimitiveIdLut, geoms); 
        }
    }


}

glm::mat4 GetNodeTransform(const tinygltf::Node& node)
{
    glm::mat4 mat(1.0f);
    if (!node.matrix.empty())
    {
        // GLTF matrix is column-major, glm uses column-major
        for (int i = 0; i < 16; ++i)
            mat[i / 4][i % 4] = static_cast<float>(node.matrix[i]);
    }
    else
    {
        glm::vec3 translation(0.0f);
        glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale(1.0f);

        if (!node.translation.empty())
            translation = glm::vec3(
                static_cast<float>(node.translation[0]),
                static_cast<float>(node.translation[1]),
                static_cast<float>(node.translation[2]));
        if (!node.rotation.empty())
            rotation = glm::quat(
                static_cast<float>(node.rotation[3]), // w
                static_cast<float>(node.rotation[0]), // x
                static_cast<float>(node.rotation[1]), // y
                static_cast<float>(node.rotation[2])); // z
        if (!node.scale.empty())
            scale = glm::vec3(
                static_cast<float>(node.scale[0]),
                static_cast<float>(node.scale[1]),
                static_cast<float>(node.scale[2]));

        mat = glm::translate(glm::mat4(1.0f), translation)
            * glm::mat4_cast(rotation)
            * glm::scale(glm::mat4(1.0f), scale);
    }
    return mat;
}

void ProcessNode(const tinygltf::Model& model, int nodeIdx, const glm::mat4& parentTransform, 
    const BufferOffsets& bufferOffsets, const std::map<int, std::map<int, int> > gltfMeshPrimitiveIdLut,
    std::vector<Geom>& geomInstances)
{
    const auto& node = model.nodes[nodeIdx];
    glm::mat4 nodeTransform = parentTransform * GetNodeTransform(node);

    if (node.mesh >= 0)
    {
        // For each primitive in the mesh, create an instance
        const auto& mesh = model.meshes[node.mesh];
        for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx)
        {
            Geom geom;
            geom.type = MESH;
            int sceneMeshID = gltfMeshPrimitiveIdLut.at(node.mesh).at(primIdx);
            geom.meshId = sceneMeshID;
            geom.transform = nodeTransform;
            geom.invTransform = glm::inverse(nodeTransform);
            geom.invTranspose = glm::inverseTranspose(nodeTransform);
            // Assign material if available
            int matID = mesh.primitives[primIdx].material;
            if (matID < 0)
            {
                geom.materialId = 0; // default material
            }
            else
            {
                geom.materialId = bufferOffsets.material + matID;
            }
            geomInstances.push_back(geom);
        }
    }
    // Recursively process children
    for (int childIdx : node.children)
    {
        ProcessNode(model, childIdx, nodeTransform, bufferOffsets, gltfMeshPrimitiveIdLut, geomInstances);
    }
}
