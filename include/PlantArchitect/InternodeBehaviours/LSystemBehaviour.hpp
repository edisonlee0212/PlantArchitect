#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API LSystemTag : public IDataComponent {
    };

    struct PLANT_ARCHITECT_API LSystemParameters : public IDataComponent {
        float m_internodeLength = 1.0f;
        float m_thicknessFactor = 0.5f;
        float m_endNodeThickness = 0.01f;
        void OnInspect();
    };

    class PLANT_ARCHITECT_API LSystemBehaviour : public IInternodeBehaviour {
    public:
        void OnInspect() override;
        void OnCreate() override;
        void PreProcess() override;
        void Grow() override;
        void PostProcess() override;
    };
}