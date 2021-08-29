#pragma once
#include <plant_architect_export.h>
#include <IInternodeResource.hpp>
using namespace UniEngine;
namespace PlantArchitect {
    class DefaultInternodeResource : public IInternodeResource{
    public:
        void Collect(float deltaTime, const Entity& self) override;
        void DownStream(float deltaTime, const Entity& self, const Entity& target) override;
        void UpStream(float deltaTime, const Entity& self, const Entity& target) override;
    };
}