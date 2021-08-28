#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API InternodeBehaviour : public IAsset {
    public:
        virtual void CollectResources(const Entity &internode) = 0;

        virtual void DownStreamResources(const Entity &internode) = 0;

        virtual void UpStreamResources(const Entity &internode) = 0;
    };

    class PLANT_ARCHITECT_API DefaultInternodeBehaviour : public InternodeBehaviour {
        void CollectResources(const Entity &internode) override {};

        void DownStreamResources(const Entity &internode) override {};

        void UpStreamResources(const Entity &internode) override {};
    };
}