#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API IVolume : public IPrivateComponent {
    public:
        bool m_asObstacle = true;
        bool m_displayBounds = true;

        virtual glm::vec3 GetRandomPoint() { return glm::vec3(0.0f); }

        virtual bool InVolume(const GlobalTransform &globalTransform, const glm::vec3 &position);

        virtual bool InVolume(const glm::vec3 &position);
        virtual void InVolume(const GlobalTransform &globalTransform, const std::vector<glm::vec3> &positions, std::vector<bool>& results);

        virtual void InVolume(const std::vector<glm::vec3> &positions, std::vector<bool>& results);
        void Serialize(YAML::Emitter &out) override;

        void OnInspect() override;

        void Deserialize(const YAML::Node &in) override;;

        void OnDestroy() override;
    };
} // namespace PlantFactory
