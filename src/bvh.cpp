#include "bvh.h"

#include <stdexcept>
#include <algorithm>
#include <deque>
#include <unordered_map>

template<typename T>
using uPtr = std::unique_ptr<T>;

using std::make_unique;



uPtr<Node>& buildBvhFromRoot(uPtr<Node>& node, const std::vector<glm::vec3>& positions)
{
    // If leaf node (1 triangle), stop recursion
    if (node->triangleIndices.size() <= 4)
    {
        node->isLeaf = true;
        // Compute bounding box for the triangle
        node->bound = AABB();
        for (int idx : node->triangleIndices)
        {
            node->bound.include(positions[idx + 0]);
            node->bound.include(positions[idx + 1]);
            node->bound.include(positions[idx + 2]);
        }
        return node;
    }

    node->isLeaf = false;

    // Compute bounding box and centroids
    node->bound = AABB();
    std::vector<glm::vec3> centroids;
    for (int idx : node->triangleIndices)
    {
        glm::vec3 centroid = (positions[idx + 0] + positions[idx + 1] + positions[idx + 2]) / 3.0f;
        centroids.push_back(centroid);
        node->bound.include(positions[idx + 0]);
        node->bound.include(positions[idx + 1]);
        node->bound.include(positions[idx + 2]);
    }

    // Find axis with largest extent
    glm::vec3 extent = node->bound.max - node->bound.min;
    int axis = 0;
    if (extent.y > extent.x && extent.y > extent.z) axis = 1;
    else if (extent.z > extent.x && extent.z > extent.y) axis = 2;

    // Sort triangle indices by centroid along chosen axis
    std::vector<int> sortedIdx = node->triangleIndices;
    std::sort(sortedIdx.begin(), sortedIdx.end(), [&](int a, int b) {
        glm::vec3 ca = (positions[a + 0] + positions[a + 1] + positions[a + 2]) / 3.0f;
        glm::vec3 cb = (positions[b + 0] + positions[b + 1] + positions[b + 2]) / 3.0f;
        return ca[axis] < cb[axis];
        });

    // Split indices into two halves
    size_t mid = sortedIdx.size() / 2;
    std::vector<int> leftIdx(sortedIdx.begin(), sortedIdx.begin() + mid);
    std::vector<int> rightIdx(sortedIdx.begin() + mid, sortedIdx.end());

    // Create child nodes
    node->left = std::make_unique<Node>();
    node->left->triangleIndices = leftIdx;

    node->right = std::make_unique<Node>();
    node->right->triangleIndices = rightIdx;

    // Recursively build children
    buildBvhFromRoot(node->left, positions);
    buildBvhFromRoot(node->right, positions);

    return node;
}


int getNodeCount(const uPtr<Node>& node)
{
    if (node->isLeaf)
    {
        return 1;
    }
    return 1 + getNodeCount(node->left) + getNodeCount(node->right);
}

uPtr<Node> buildBvhFromPositionBuffer(const std::vector<glm::vec3>& positions)
{
    if (positions.size() % 3 != 0)
    {
        throw std::runtime_error("Position buffer size is not a multiple of 3");
    }

    uPtr<Node> root = make_unique<Node>();

    for (int i = 0; i < positions.size(); i += 3)
    {
        root->triangleIndices.push_back(i);

        root->bound.include(positions[i]);
        root->bound.include(positions[i+1]);
        root->bound.include(positions[i+2]);
    }

    return std::move(buildBvhFromRoot(root, positions));
}

// Convert the bvh tree to GPU friendly NodeProxy data
void buildNodeProxyBuffers(const uPtr<Node>& in_node, std::vector<NodeProxy>& out_nodeProxies, std::vector<int>& out_triangles)
{
    out_nodeProxies.clear();
    out_triangles.clear();

    // initialize
    out_nodeProxies.reserve(getNodeCount(in_node));
    out_nodeProxies.push_back({});
    std::deque<Node*> workQueue{ in_node.get() };

    std::unordered_map<Node*, int> nodeToProxyLut {{in_node.get(), 0} }; 

    while (workQueue.size() > 0)
    {
        Node* curNode = workQueue.front();
        workQueue.pop_front();

        NodeProxy& curNodeProxy = out_nodeProxies[nodeToProxyLut.at(curNode)];

        curNodeProxy.bound = curNode->bound;

        if (curNode->isLeaf)
        {
            curNodeProxy.triangleCount = curNode->triangleIndices.size();
            curNodeProxy.triangleBufferOffset = out_triangles.size();
            out_triangles.insert(out_triangles.end(), curNode->triangleIndices.begin(), curNode->triangleIndices.end());
            continue;
        }

        // Push empty child Nodeproxy to result and set linkage
        if (curNode->left)
        {
            Node* leftChildNode = curNode->left.get();

            int childIdx = out_nodeProxies.size();
            out_nodeProxies.push_back(NodeProxy{});
            curNodeProxy.leftChildIdx = childIdx;
            nodeToProxyLut[leftChildNode] = childIdx;

            // push child node to queue
            workQueue.push_back(leftChildNode);
        }
        if (curNode->right)
        {
            Node* rightChildNode = curNode->right.get();

            int childIdx = out_nodeProxies.size();
            out_nodeProxies.push_back(NodeProxy{});
            curNodeProxy.rightChildIdx = childIdx;
            nodeToProxyLut[rightChildNode] = childIdx;

            workQueue.push_back(rightChildNode);

        }
    }
}

