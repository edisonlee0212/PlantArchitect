#pragma once

#include <plant_architect_export.h>
#include "Application.hpp"
#include "glm/gtc/noise.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API FBM {
    public:
        [[nodiscard]] glm::vec3 GetT(const glm::vec3& q, float t, float frequency, float density, int level);
        glm::mat3 m_m = glm::mat3(0.5f);
        float m_t = 0.0f;

        [[nodiscard]] float Get(const glm::vec3 &in, unsigned octaves);

        [[nodiscard]] glm::vec3 Get3(const glm::vec3 &in, unsigned level);

        void OnInspect();

        void OnCreate();
    };
}