#pragma once
#include <IVolume.hpp>
#include <plant_architect_export.h>
using namespace UniEngine;
namespace PlantArchitect {
class PLANT_ARCHITECT_API CubeVolume : public IVolume {
public:
  void ApplyMeshRendererBounds();
  void OnCreate() override;
  bool m_displayBounds = false;
  Bound m_minMaxBound;
  void OnInspect() override;
  bool InVolume(const GlobalTransform& globalTransform, const glm::vec3 &position) override;
  bool InVolume(const glm::vec3 &position) override;
  glm::vec3 GetRandomPoint() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void OnDestroy() override;
};
} // namespace PlantFactory