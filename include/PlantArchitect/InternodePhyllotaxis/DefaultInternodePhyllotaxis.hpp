#pragma once
#include <plant_architect_export.h>
#include "IInternodePhyllotaxis.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API DefaultInternodePhyllotaxis : public IInternodePhyllotaxis {
    public:
        float m_positionVariance = 0.5f;
        float m_randomRotation = 10.0f;
        glm::vec2 m_leafSize = glm::vec2(0.1f, 0.1f);
        int m_leafCount = 40;
        void OnInspect() override;
        void GenerateFoliage(const std::shared_ptr<Internode> &internode, const InternodeInfo &internodeInfo,
                             const GlobalTransform &relativeGlobalTransform, const GlobalTransform &relativeParentGlobalTransform) override;

        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
    };
}