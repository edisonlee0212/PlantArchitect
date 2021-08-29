#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API DefaultInternodeTag : public IDataComponent {

    };

    class PLANT_ARCHITECT_API DefaultInternodeBehaviour : public IInternodeBehaviour {
    public:
        void OnInspect() override;
        void OnCreate() override;
        void PreProcess() override;
        void Grow() override;
        void PostProcess() override;
    };
}