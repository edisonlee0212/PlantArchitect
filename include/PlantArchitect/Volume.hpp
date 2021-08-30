#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API Volume : public IPrivateComponent {
    public:
        bool m_asObstacle = false;

        virtual glm::vec3 GetRandomPoint() { return glm::vec3(0.0f); }

        virtual bool InVolume(const GlobalTransform &globalTransform, const glm::vec3 &position) { return false; }

        virtual bool InVolume(const glm::vec3 &position) { return false; }

        void Clone(const std::shared_ptr<UniEngine::IPrivateComponent> & target) override { }
    };
} // namespace PlantFactory
