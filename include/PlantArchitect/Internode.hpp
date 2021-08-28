#pragma once
#include <plant_architect_export.h>
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API Internode : public IPrivateComponent {
    public:
        AssetRef m_internodeBehaviour;
        void Clone(const std::shared_ptr<IPrivateComponent> &target);


    };
}