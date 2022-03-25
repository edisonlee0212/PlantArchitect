#pragma once

#include "InternodeRingSegment.hpp"
#include <plant_architect_export.h>
#include <IInternodeResource.hpp>
#include <PlantDataComponents.hpp>
#include "IPlantDescriptor.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API Root : public IPrivateComponent {
    public:
        /**
         * The foliage module.
         */
        AssetRef m_foliagePhyllotaxis;
        AssetRef m_foliageTexture;
        AssetRef m_branchTexture;
        AssetRef m_plantDescriptor;
        glm::vec3 m_foliageColor = glm::vec3(0, 1, 0);
        glm::vec3 m_branchColor = glm::vec3(40.0f / 255, 15.0f / 255, 0.0f);
        glm::vec3 m_center = glm::vec3(0.0f);
        void OnInspect() override;
        void OnCreate() override;
        void Serialize(YAML::Emitter &out) override;
        void OnDestroy() override;
        void Deserialize(const YAML::Node &in) override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;
    };
}