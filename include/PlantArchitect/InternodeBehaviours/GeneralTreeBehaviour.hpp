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
        float m_lastRequest = 0;
        float m_waterDividends = 1.0f;
        void OnInspect() override;
    };

    class PLANT_ARCHITECT_API GeneralTreeBehaviour : public IInternodeBehaviour {
        std::vector<Entity> m_currentPlants;
        Entity ImportGANTree(const std::filesystem::path& path, const GeneralTreeParameters& parameters);

    protected:
        bool InternalInternodeCheck(const Entity &target) override;

    public:
        void OnInspect() override;

        void OnCreate() override;

        void Grow(int iteration) override;

        Entity Retrieve() override;

        Entity Retrieve(const Entity &parent) override;

        Entity NewPlant(const GeneralTreeParameters &params, const Transform &transform);
    };
}