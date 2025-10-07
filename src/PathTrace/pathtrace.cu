#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <__msvc_filebuf.hpp>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include "utilities.h"

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "pathTraceUtils.h"
#include "intersections.h"
#include "ImGui/imgui.h"
#include "PathTrace/bsdf.h"
#include "bvh.h"


#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

/* ------------------ Mesh ------------------ */
// TODO:  multi mesh support
static Mesh* hst_singleMesh = NULL;
static int hst_singleMeshTriangleCount;
static std::vector<glm::vec3> hst_positions;

static AABB* dev_singleMeshAABB = NULL;
static glm::vec3* dev_singleMeshPosition = NULL;
// static glm::vec3 dev_mesh_normals;
// static glm::vec2 dev_mesh_uvs;

static NodeProxy* dev_nodes = NULL;
static int* dev_nodeTriangles = NULL;

/* ------------------ Light ------------------ */
static AreaLight* dev_areaLights;


// Debug buffers
static PathSegment* host_paths = nullptr;


void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // Mesh data
    // TODO:  multi mesh support
    hst_positions.clear();
    if (scene->meshes.size() > 0)
    {
        hst_singleMesh = &scene->meshes[0];

        // Position
        for (int index : hst_singleMesh->indices)
        {
            hst_positions.push_back(hst_singleMesh->positions.at(index));
        }
        hst_singleMeshTriangleCount = hst_positions.size() / 3;
        cudaMalloc(&dev_singleMeshPosition, hst_positions.size() * sizeof(glm::vec3));   // TODO:  multi mesh support
        cudaMemcpy(dev_singleMeshPosition, hst_positions.data(), hst_singleMesh->indices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

        // AABB
        cudaMalloc(&dev_singleMeshAABB, sizeof(AABB));
        cudaMemcpy(dev_singleMeshAABB, &hst_singleMesh->bound, sizeof(AABB), cudaMemcpyHostToDevice);
    }

    // BVH
    auto root = buildBvhFromPositionBuffer(hst_positions);
    std::vector<NodeProxy> nodePorxies;
    std::vector<int> nodeTriangles;
    buildNodeProxyBuffers(root, nodePorxies, nodeTriangles);

    cudaMalloc(&dev_nodes, nodePorxies.size() * sizeof(NodeProxy));
    cudaMemcpy(dev_nodes, nodePorxies.data(), nodePorxies.size() * sizeof(NodeProxy), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_nodeTriangles, nodeTriangles.size() * sizeof(int));
    cudaMemcpy(dev_nodeTriangles, nodeTriangles.data(), nodeTriangles.size() * sizeof(int), cudaMemcpyHostToDevice);


    // Debug buffers
    host_paths = new PathSegment[pixelcount];

    cudaDeviceSynchronize();
    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaDeviceSynchronize();

    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    // Mesh data
    cudaFree(dev_singleMeshPosition);
    cudaFree(dev_singleMeshAABB);
    cudaFree(dev_nodes);
    cudaFree(dev_nodeTriangles);

    // debug 
    delete[] host_paths;

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool jitter)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;


    int index = x + (y * cam.resolution.x);
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 1); // TODO: depth = 0 creates artifacts of black seams on the sphere, need to figure out later
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (x < cam.resolution.x && y < cam.resolution.y)
    {
        PathSegment& segment = pathSegments[index];

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.color = glm::vec3(0);
        segment.throughput = glm::vec3(1);

        segment.ray.origin = cam.position;

        float pixel_x_local;
        float pixel_y_local;
         
        if (jitter)
        {
            pixel_x_local = u01(rng);
            pixel_y_local = u01(rng);
        }
        else
        {
            pixel_x_local = 0.5f;
            pixel_y_local = 0.5f;
        }
        float pixel_x_world = (float)x + pixel_x_local - (float)cam.resolution.x * 0.5f;
        float pixel_y_world = (float)y + pixel_y_local - (float)cam.resolution.y * 0.5f;


        segment.ray.direction = glm::normalize(cam.forward
            - cam.right * cam.pixelLength.x * pixel_x_world     // TODO: it's subtraction here, but it works, will come back later
            - cam.up * cam.pixelLength.y * pixel_y_world);

    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    const glm::vec3* meshPositions,
    const NodeProxy* nodes,
    const int* triangleIndices,
    int meshTriangleCount,
    const AABB* meshBound,
    ShadeableIntersection* intersections, 
    bool useBVH)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t = -1;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;

        bool outside = true;
        bool tmp_outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // Intersection test: naive parse through global geoms
        // TODO: use BVH instead
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            Ray rt{
                multiplyMV(geom.invTransform, glm::vec4(pathSegment.ray.origin, 1.0f)),
                glm::normalize(multiplyMV(geom.invTransform, glm::vec4(pathSegment.ray.direction, 0.0f))),
            };

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == MESH) 
            {
                AABB aabb = meshBound[0];
                if (aabbIntersectionTest(geom, aabb, pathSegment.ray))
                {
                    if (useBVH)
                    {
                        t = bvhIntersectionTest(geom, nodes, meshPositions, triangleIndices, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
                    }
                    else
                    {
                        t = meshIntersectionTest(geom, meshPositions, meshTriangleCount, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
                    }
                }

            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialId;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = outside;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(int iter, int num_paths, ShadeableIntersection* shadeableIntersections, 
    PathSegment* pathSegments, const Material* materials, const int depth)
{
    using glm::vec3;
    using glm::dot;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) { return; }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment segment = pathSegments[idx];
    if (segment.remainingBounces < 0) {return;} // do one more calculation when bounce reaches zero
    if (intersection.t > 0.0f) // if the intersection exists...
    {
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material material = materials[intersection.materialId];
        vec3 materialColor = material.color;
        vec3 normal = intersection.surfaceNormal;
        // If the material indicates that the object was a light, "light" the ray
        if (material.emittance > 0.0f) {
            segment.color = materialColor * material.emittance * segment.throughput;
            segment.remainingBounces = 0;
        } 
        else 
        {
            vec3 intersectionPos = getPointOnRay(segment.ray, intersection.t);

            vec3 f; vec3 wiW; float pdf;
            vec3 debug;
            f = sample_f(debug, wiW, pdf, -segment.ray.direction, material, normal, intersection.outside, rng);
            wiW = glm::normalize(wiW);
            float absCosThetaI = abs(dot(wiW, normal));
            segment.throughput *= f * absCosThetaI / pdf; // __fdividef(absCosTheta, pdf)
            
            segment.color_debug = debug;

            // update next 
            segment.ray.origin = intersectionPos + wiW * 0.01f; // fix self-intersection
            segment.ray.direction = wiW;

            segment.remainingBounces -= 1;
        }
    }
    else {
        segment.color = vec3(0.0f);
        segment.remainingBounces = 0;
    }
    pathSegments[idx] = segment;

}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths, const bool showDebugColor)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (!showDebugColor)
        {
            image[iterationPath.pixelIndex] += iterationPath.color;
        }
        else
        {
            image[iterationPath.pixelIndex] += iterationPath.color_debug;
        }
    }
}

struct thread_is_active
{
    __device__
        auto operator()(const PathSegment& seg) -> bool
    {
        return seg.remainingBounces > 0 && glm::length(seg.throughput) > EPSILON;
    }
};

struct sort_by_material
{
    __device__
    auto operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
    {
        return a.materialId < b.materialId;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, guiData->jitter);

    // cudaDeviceSynchronize();
    // checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks


    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_singleMeshPosition,
            dev_nodes,
            dev_nodeTriangles,
            hst_singleMeshTriangleCount,
            dev_singleMeshAABB,
            dev_intersections,
            guiData->useBVH
        );
        // cudaDeviceSynchronize();
        // checkCUDAError("trace one bounce");

        if (guiData->sortMaterial)
        {
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sort_by_material());
        }


        // shading
        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth
        );
        // cudaDeviceSynchronize();
        // checkCUDAError("shade material");

        // compact
        if (guiData->earlyOut)
        {
            PathSegment* a = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, thread_is_active());
            num_paths = a - dev_paths;
        }
        fprintf(stdout, "depth: %i, num_paths: %i\n", depth, num_paths);

        depth++;
        if (num_paths < 1) iterationComplete = true;
        if (depth >= 8) iterationComplete = true;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths, guiData->ShowDebugColor);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);


    checkCUDAError("pathtrace");
}

void gatherImageFromDevice()
{
    int pixelcount = hst_scene->state.camera.resolution.x * hst_scene->state.camera.resolution.y;
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}
