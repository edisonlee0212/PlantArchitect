#pragma once
#include <UniEngine-pch.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>

#include <Entity.hpp>
#include <Mesh.hpp>
#include <Texture2D.hpp>

using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API MLVQRenderer : public IPrivateComponent {
    public:
        AssetRef m_mesh;
        int m_materialIndex = 0;
        void OnGui() override;
        void Sync();

        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;
        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
    };
} // namespace RayTracerFacility
