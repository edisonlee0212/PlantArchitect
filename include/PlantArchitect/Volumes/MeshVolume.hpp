#pragma once
#include <IVolume.hpp>
#include <plant_architect_export.h>
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API MeshVolume : public IVolume {
    public:
        void OnCreate() override;
        PrivateComponentRef m_meshRendererRef;
        void OnInspect() override;
        bool InVolume(const GlobalTransform& globalTransform, const glm::vec3 &position) override;
        bool InVolume(const glm::vec3 &position) override;
        void InVolume(const GlobalTransform &globalTransform, const std::vector<glm::vec3> &positions, std::vector<bool>& results) override;
        void InVolume(const std::vector<glm::vec3> &positions, std::vector<bool>& results) override;
        glm::vec3 GetRandomPoint() override;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
        void OnDestroy() override;

        void Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) override;
    };
} // namespace PlantFactory