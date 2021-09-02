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

    enum class PLANT_ARCHITECT_API LSystemCommandType{
        /**
         * Command F
         */
        Forward,
        /**
         * Command +
         */
        TurnLeft,
        /**
         * Command -
         */
        TurnRight,
        /**
         * Command ^
         */
        PitchUp,
        /**
         * Command &
         */
        PitchDown,
        /**
         * Command \
         */
        RollLeft,
        /**
         * Command /
         */
        RollRight,
        /**
         * Command [
         */
        Push,
        /**
         * Command ]
         */
        Pop
    };

    struct PLANT_ARCHITECT_API LSystemCommand{
        LSystemCommandType m_type;
        float m_value;
    };

    class PLANT_ARCHITECT_API LSystemBehaviour : public IInternodeBehaviour {
    protected:
        bool InternalInternodeCheck(const Entity &target) override;
    public:
        Entity FormPlant(std::vector<LSystemCommand>& commands, const LSystemParameters& parameters);
        void ParseLString(const std::string& string, std::vector<LSystemCommand>& commands);
        void OnInspect() override;
        void OnCreate() override;
        Entity Retrieve() override;
        Entity Retrieve(const Entity &parent) override;
    };
}