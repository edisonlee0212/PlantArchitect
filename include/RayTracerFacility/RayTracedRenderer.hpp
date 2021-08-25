#pragma once
#include <UniEngine-pch.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>

#include <Entity.hpp>
#include <Mesh.hpp>
#include <Texture2D.hpp>

using namespace UniEngine;
namespace RayTracerFacility {
class RAY_TRACER_FACILITY_API RayTracedRenderer : public IPrivateComponent {
public:
  float m_diffuseIntensity = 0;
  float m_transparency = 1.0f;
  float m_metallic = 0.3f;
  float m_roughness = 0.3f;
  glm::vec3 m_surfaceColor = glm::vec3(1.0f);
  AssetRef m_mesh;
  AssetRef m_albedoTexture;
  AssetRef m_normalTexture;
  void OnGui() override;
  void SyncWithMeshRenderer();

  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;

  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
};
} // namespace RayTracerFacility
