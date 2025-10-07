#pragma once

#include "glm/glm.hpp"

#include <sstream>
#include <string>
#include <vector>

// Numeric constants
constexpr float pi = 3.1415926535897932384626422832795028841971f;
constexpr float two_pi = 2 * pi;
constexpr float pi_over_2 = pi * 0.5;
constexpr float epsilon = 0.00001f;

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0), ShowDebugColor(false){}
    int TracedDepth;
    bool ShowDebugColor;


    bool useBVH = true;
    bool useAABB = true;

    bool jitter = true;

    bool earlyOut = false;
    bool sortMaterial = false;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); // Thanks to http://stackoverflow.com/a/6089413
    extern std::string GetFilePathExtension(const std::string& fileName); // From tiny_gltf.h

    template <class T>
    T divUp(T size, T div)
    {
        return (size + div - 1) / div;
    }

    

    template <class T>
    bool epsilonCheck(T a, T b)
    {
        return fabs(a - b) < epsilon;
    }

    template<>
    inline bool epsilonCheck<float>(float a, float b)
    {
        return fabs(a - b) < epsilon;
    }
}

