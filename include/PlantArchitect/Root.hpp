#pragma once

#include "InternodeRingSegment.hpp"
#include <plant_architect_export.h>
#include <IInternodeResource.hpp>
#include <PlantDataComponents.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API Root : public IPrivateComponent {
    public:
        glm::vec3 m_center;

        void OnInspect() override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;
    };
}