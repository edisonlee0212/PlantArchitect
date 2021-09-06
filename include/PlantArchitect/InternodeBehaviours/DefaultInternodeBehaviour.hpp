#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API DefaultInternodeTag : public IDataComponent {

    };

    class PLANT_ARCHITECT_API DefaultInternodeBehaviour : public IInternodeBehaviour {
    protected:
        bool InternalInternodeCheck(const Entity &target) override;
    public:
        void OnInspect() override;
        void OnCreate() override;
        void PreProcess(float deltaTime) override;
        void Grow(float deltaTime) override;
        void PostProcess(float deltaTime) override;
        Entity Retrieve() override;
        Entity Retrieve(const Entity &parent) override;
    };
}