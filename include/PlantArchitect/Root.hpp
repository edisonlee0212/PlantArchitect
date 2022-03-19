#pragma once

#include "InternodeRingSegment.hpp"
#include <plant_architect_export.h>
#include <IInternodeResource.hpp>
#include <PlantDataComponents.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API Root : public IPrivateComponent {
    public:
        glm::vec3 m_center = glm::vec3(0.0f);;
        void OnInspect() override;
        void OnCreate() override;
        void Serialize(YAML::Emitter &out) override;
        void OnDestroy() override;
        void Deserialize(const YAML::Node &in) override;
    };
}