#pragma once

#include "glm/glm.hpp"
#include <cuda_runtime.h>

/* -------- math --------- */
__device__
glm::vec3 squareToDiskConcentric(glm::vec2 xi);

__device__
glm::vec3 squareToHemisphereCosine(glm::vec2 xi);

__device__
float squareToHemisphereCosinePDF(glm::vec3 sample);