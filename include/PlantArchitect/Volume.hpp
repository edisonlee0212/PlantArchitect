#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API Volume : public IPrivateComponent {
    public:
        bool m_asObstacle = false;

        virtual glm::vec3 GetRandomPoint() = 0;

        virtual bool InVolume(const GlobalTransform &globalTransform, const glm::vec3 &position) = 0;

        virtual bool InVolume(const glm::vec3 &position) = 0;
    };
} // namespace PlantFactory
