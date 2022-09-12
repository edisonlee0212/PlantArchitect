#pragma once

#include <plant_architect_export.h>
#include "Application.hpp"
#include "InternodeModel/Internode.hpp"
#include "InternodeLayer.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API IInternodeFoliage : public IAsset {
    public:
        AssetRef m_foliageTexture;
        virtual void GenerateFoliage(const std::shared_ptr<Internode> &internode, const InternodeInfo &internodeInfo,
                                     const GlobalTransform &relativeGlobalTransform, const GlobalTransform &relativeParentGlobalTransform) = 0;

        void OnInspect() override;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
        void CollectAssetRef(std::vector<AssetRef> &list) override;
    };
}