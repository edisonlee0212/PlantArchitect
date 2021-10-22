#pragma once

#include <plant_architect_export.h>
#include <Application.hpp>
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API Camera2DVectorField : public IPrivateComponent{

    public:
        glm::ivec2 m_resolution = {64, 64};

        std::vector<std::vector<glm::vec2>> m_vectorField;
        void Construct();
    };
}