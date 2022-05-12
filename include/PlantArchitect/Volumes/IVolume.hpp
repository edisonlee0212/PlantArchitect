#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API IVolume : public IPrivateComponent {
    public:
        bool m_asObstacle = false;

        virtual glm::vec3 GetRandomPoint() { return glm::vec3(0.0f); }

        virtual bool InVolume(const GlobalTransform &globalTransform, const glm::vec3 &position) { return false; }

        virtual bool InVolume(const glm::vec3 &position) { return false; }

        void Serialize(YAML::Emitter &out) override {
            out << YAML::Key << "m_asObstacle" << YAML::Value << m_asObstacle;
        }

        void Deserialize(
                const YAML::Node &in) override { if (in["m_asObstacle"]) m_asObstacle = in["m_asObstacle"].as<bool>(); };

        void OnDestroy() override { m_asObstacle = false; }
    };
} // namespace PlantFactory
