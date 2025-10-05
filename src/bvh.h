#pragma once

#include "sceneStructs.h"
#include <memory>


struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};

struct Node
{
    AABB bound;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    bool isLeaf;

    std::vector<int> triangleIndices;

    Node() : left(nullptr),right(nullptr), isLeaf(false) {}

    ~Node() = default;
};


struct NodeProxy
{
    AABB bound;

    int leftChildIdx;
    int rightChildIdx;

    int triangleCount = 0;
    int triangleBufferOffset = -1;
};


std::unique_ptr<Node> buildBvhFromPositionBuffer(const std::vector<glm::vec3>& positions);

void buildNodeProxyBuffers(const std::unique_ptr<Node>& in_node, std::vector<NodeProxy>& out_nodeProxies, std::vector<int>& out_triangles);
