#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>
#include <GeneralTreeParameters.hpp>
using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API GeneralTreeTag : public IDataComponent {

    };

    struct GanNode{
        Entity m_internode;
        glm::vec3 m_start;
        float m_length;
        glm::quat m_globalRotation;
    };

    class PLANT_ARCHITECT_API InternodeWaterFeeder : public IPrivateComponent{
    public:
        float m_lastRequest = 0;
        float m_waterDividends = 1.0f;
        void OnInspect() override;
    };

    class PLANT_ARCHITECT_API GeneralTreeBehaviour : public IInternodeBehaviour {

        Entity ImportGraphTree(const std::filesystem::path& path, const GeneralTreeParameters& parameters);
    protected:
        bool InternalInternodeCheck(const Entity &target) override;

    public:
        void Preprocess();

        void OnInspect() override;

        void OnCreate() override;

        void Grow(int iteration) override;

        Entity CreateInternode() override;

        Entity CreateInternode(const Entity &parent) override;

        Entity NewPlant(const GeneralTreeParameters &params, const Transform &transform);
    };
}