#pragma once

#include <plant_architect_export.h>
#include <Internode.hpp>
#include <IInternodeBehaviour.hpp>
#include <InternodeDataComponents.hpp>
#include <VoxelSpace.hpp>
using namespace UniEngine;

namespace PlantArchitect {
    enum class BranchColorMode{
        None,
        Order,
        Level,
        Water,
        ApicalControl,
        WaterPressure,
        Proximity,
        Inhibitor,
        IndexDivider,
        IndexRange
    };
    class IInternodeBehaviour;
    class PLANT_ARCHITECT_API InternodeSystem : public ISystem {
    public:
        /**
         * The EntityQuery for filtering all internodes.
         */
        EntityQuery m_internodesQuery;

        void LateUpdate() override;

        void Simulate(int iterations);

        void OnCreate() override;

        void OnInspect() override;

        std::shared_ptr<Camera> m_internodeDebuggingCamera;

        template<typename T>
        void PushInternodeBehaviour(const std::shared_ptr<T>& behaviour);
        template<typename T>
        std::shared_ptr<T> GetInternodeBehaviour();
        /**
         * Check if the entity is valid internode.
         * @param target Target for check.
         * @return True if the entity is valid and contains [InternodeInfo] and [Internode], false otherwise.
         */
        bool InternodeCheck(const Entity& target);

        void CollectAssetRef(std::vector<AssetRef> &list) override;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;

        BranchColorMode m_branchColorMode = BranchColorMode::None;
        int m_indexDivider = 512;
        int m_indexRangeMin = 128;
        int m_indexRangeMax = 512;
        void UpdateBranchColors();
        void UpdateBranchCylinder(const float &width = 0.01f);
        void UpdateBranchPointer(const float &length,
                                 const float &width = 0.01f);
    private:
        VoxelSpace m_voxelSpace;

        friend class Internode;

        std::vector<AssetRef> m_internodeBehaviours;


#pragma region Internode debugging camera

        int m_internodeDebuggingCameraResolutionX = 1;
        int m_internodeDebuggingCameraResolutionY = 1;
        float m_lastX = 0;
        float m_lastY = 0;
        float m_lastScrollY = 0;
        bool m_startMouse = false;
        bool m_startScroll = false;
        bool m_rightMouseButtonHold = false;
        EntityRef m_currentFocusingInternode = Entity();

#pragma region Rendering


        float m_connectionWidth = 1.0f;

        float m_pointerLength = 0.4f;
        float m_pointerWidth = 0.02f;

        bool m_drawBranches = true;
        bool m_drawPointers = false;

        float m_transparency = 0.7f;

        bool m_autoUpdate = true;

        float m_branchColorValueMultiplier = 1.0f;
        float m_branchColorValueCompressFactor = 0.0f;
        glm::vec3 m_branchColor = glm::vec3(0, 1, 0);
        std::vector<glm::vec3> m_randomColors;
        glm::vec4 m_pointerColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.5f);

        std::vector<Entity> m_entitiesWithRenderer;
        OpenGLUtils::GLVBO m_internodeColorBuffer;


        void RenderBranchCylinders();
        void RenderBranchPointers();
#pragma endregion
#pragma endregion
    };
    template <typename T>
    void InternodeSystem::PushInternodeBehaviour(const std::shared_ptr<T>& behaviour) {
        if(!behaviour.get()) return;
        bool search = false;
        for (auto &i: m_internodeBehaviours) {
            if (i.Get<IInternodeBehaviour>()->GetTypeName() == std::dynamic_pointer_cast<IInternodeBehaviour>(behaviour)->GetTypeName()) search = true;
        }
        if (!search) {
            m_internodeBehaviours.emplace_back(std::dynamic_pointer_cast<IInternodeBehaviour>(behaviour));
        }
    }

    template<typename T>
    std::shared_ptr<T> InternodeSystem::GetInternodeBehaviour() {
        for (auto &i: m_internodeBehaviours) {
            if (i.Get<IInternodeBehaviour>()->GetTypeName() == SerializationManager::GetSerializableTypeName<T>()) {
                return i.Get<T>();
            }
        }
        return nullptr;
    }
}