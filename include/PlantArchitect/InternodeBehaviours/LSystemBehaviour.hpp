#pragma once

#include <plant_architect_export.h>
#include <IPlantBehaviour.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API LSystemTag : public IDataComponent {
    };

    struct PLANT_ARCHITECT_API LSystemParameters : public IDataComponent {
        float m_internodeLength = 1.0f;
        float m_thicknessFactor = 0.5f;
        float m_endNodeThickness = 0.02f;

        void OnInspect();
    };

    enum class PLANT_ARCHITECT_API LSystemCommandType {
        Unknown,
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

    struct PLANT_ARCHITECT_API LSystemCommand {
        LSystemCommandType m_type = LSystemCommandType::Unknown;
        float m_value = 0.0f;
    };

    class PLANT_ARCHITECT_API LString : public IAsset {
    protected:
        bool SaveInternal(const std::filesystem::path &path) override;

        bool LoadInternal(const std::filesystem::path &path) override;

    public:
        void ParseLString(const std::string &string);

        void OnInspect() override;

        std::vector<LSystemCommand> commands;

    };

    struct PLANT_ARCHITECT_API LSystemState {
        glm::vec3 m_eulerRotation = glm::vec3(0.0f);
        int m_index = 0;
    };

    class PLANT_ARCHITECT_API LSystemBehaviour : public IPlantBehaviour {
    protected:
        bool InternalInternodeCheck(const Entity &target) override;
        bool InternalRootCheck(const Entity &target) override;
        bool InternalBranchCheck(const Entity &target) override;

    public:
        Entity FormPlant(const std::shared_ptr<LString> &lString, const LSystemParameters &parameters);

        void OnInspect() override;

        LSystemBehaviour();

        Entity CreateRoot(Entity& rootInternode, Entity& rootBranch) override;
        Entity CreateBranch(const Entity &parent, const Entity &internode) override;
        Entity CreateInternode(const Entity &parent) override;
    };
}