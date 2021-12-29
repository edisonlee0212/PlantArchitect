#pragma once

#include <plant_architect_export.h>
#include "Application.hpp"
#include "glm/gtc/noise.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API FBMField : public IAsset {
    public:
        [[nodiscard]] glm::vec3 GetT(const glm::vec3& q, float t, float frequency, float density, int level);
        glm::mat3 m_m = glm::mat3(0.5f);
        std::function<float(const glm::vec3 &in)> m_noise = [](const glm::vec3 &i) {
            return glm::simplex(i);
        };
        float m_t = 0.0f;

        [[nodiscard]] float Get(const glm::vec3 &in, unsigned level);

        [[nodiscard]] glm::vec3 Get3(const glm::vec3 &in, unsigned level);

        void OnInspect() override;

        void OnCreate() override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;
    };
}