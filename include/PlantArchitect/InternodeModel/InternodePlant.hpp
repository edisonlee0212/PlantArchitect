#pragma once

#include "RingSegment.hpp"
#include "plant_architect_export.h"
#include "IInternodeResource.hpp"
#include "DataComponents.hpp"
#include "IPlantDescriptor.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API InternodePlant : public IPrivateComponent {
    public:
        AssetRef m_plantDescriptor;
        glm::vec3 m_center = glm::vec3(0.0f);
        void OnInspect() override;
        void OnCreate() override;
        void Serialize(YAML::Emitter &out) override;
        void OnDestroy() override;
        void Deserialize(const YAML::Node &in) override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;
    };
}