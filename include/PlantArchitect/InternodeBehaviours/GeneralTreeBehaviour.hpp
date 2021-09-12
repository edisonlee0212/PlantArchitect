#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>
#include <GeneralTreeParameters.hpp>
using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API GeneralTreeTag : public IDataComponent {

    };



    class PLANT_ARCHITECT_API InternodeWaterFeeder : public IPrivateComponent{
    public:
        float m_waterPerIteration = 1.0f;
        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
        void OnInspect() override;
    };

    class PLANT_ARCHITECT_API GeneralTreeBehaviour : public IInternodeBehaviour {
        std::vector<Entity> m_currentPlants;

    protected:
        bool InternalInternodeCheck(const Entity &target) override;

    public:
        void OnInspect() override;

        void OnCreate() override;

        void Grow(int iterations) override;

        Entity Retrieve() override;

        Entity Retrieve(const Entity &parent) override;

        Entity NewPlant(const GeneralTreeParameters &params, const Transform &transform);
    };
}