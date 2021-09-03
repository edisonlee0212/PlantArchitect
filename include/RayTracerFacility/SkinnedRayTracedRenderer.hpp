#pragma once
#include <UniEngine-pch.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>

#include <Entity.hpp>
#include <Mesh.hpp>
#include <Texture2D.hpp>
#include "SkinnedMeshRenderer.hpp"

using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API SkinnedRayTracedRenderer : public IPrivateComponent {
    public:
        float m_diffuseIntensity = 0;
        float m_transparency = 1.0f;
        float m_metallic = 0.3f;
        float m_roughness = 0.3f;
        glm::vec3 m_surfaceColor = glm::vec3(1.0f);
        PrivateComponentRef m_skinnedMeshRenderer;
        AssetRef m_albedoTexture;
        AssetRef m_normalTexture;
        void OnGui() override;
        void Sync();

        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
        void Relink(const std::unordered_map<Handle, Handle> &map);
        void CollectAssetRef(std::vector<AssetRef> &list) override;
        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
    };
} // namespace RayTracerFacility
