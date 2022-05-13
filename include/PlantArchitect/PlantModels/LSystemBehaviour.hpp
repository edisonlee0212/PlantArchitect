#pragma once

#include <plant_architect_export.h>
#include "IPlantBehaviour.hpp"
#include "IPlantDescriptor.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API LSystemTag : public IDataComponent {
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

    class PLANT_ARCHITECT_API LSystemString : public IPlantDescriptor {
    protected:
        bool SaveInternal(const std::filesystem::path &path) override;

        bool LoadInternal(const std::filesystem::path &path) override;

    public:
        float m_internodeLength = 1.0f;
        float m_thicknessFactor = 0.5f;
        float m_endNodeThickness = 0.02f;

        void ParseLString(const std::string &string);

        void OnInspect() override;

        Entity InstantiateTree() override;

        std::vector<LSystemCommand> m_commands;

    };

    struct PLANT_ARCHITECT_API LSystemState {
        glm::vec3 m_eulerRotation = glm::vec3(0.0f);
        int m_index = 0;
    };

    class PLANT_ARCHITECT_API LSystemBehaviour : public IPlantBehaviour {
    protected:
        bool InternalInternodeCheck(const std::shared_ptr<Scene> &scene, const Entity &target) override;

        bool InternalRootCheck(const std::shared_ptr<Scene> &scene, const Entity &target) override;

        bool InternalBranchCheck(const std::shared_ptr<Scene> &scene, const Entity &target) override;

    public:
        Entity NewPlant(const std::shared_ptr<Scene> &scene, const std::shared_ptr<LSystemString> &descriptor, const Transform &transform);

        void OnMenu() override;

        LSystemBehaviour();

        Entity CreateRoot(const std::shared_ptr<Scene> &scene, AssetRef descriptor, Entity &rootInternode, Entity &rootBranch) override;

        Entity CreateBranch(const std::shared_ptr<Scene> &scene, const Entity &parent, const Entity &internode) override;

        Entity CreateInternode(const std::shared_ptr<Scene> &scene, const Entity &parent) override;
    };
}