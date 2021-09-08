#pragma once
#include <plant_architect_export.h>
#include "IInternodePhyllotaxis.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API DefaultInternodePhyllotaxis : public IInternodePhyllotaxis {
    public:
        float m_randomRadius = 0.2f;
        float m_randomRotation = 10.0f;
        glm::vec2 m_leafSize = glm::vec2(0.1f, 0.3f);
        int m_leafCount = 10;
        void OnInspect() override;
        void GenerateFoliage(const std::shared_ptr<Internode> &internode, const InternodeInfo &internodeInfo,
                             const GlobalTransform &relativeGlobalTransform) override;
    };
}