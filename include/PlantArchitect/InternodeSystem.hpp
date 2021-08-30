#pragma once

#include <plant_architect_export.h>
#include <Internode.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API InternodeInfo : public IDataComponent {
        float m_thickness = 1.0f;
        float m_length = 0;
    };

    struct PLANT_ARCHITECT_API BranchColor : IDataComponent {
        glm::vec4 m_value;
    };

    struct PLANT_ARCHITECT_API BranchCylinder : IDataComponent {
        glm::mat4 m_value;

        bool operator==(const BranchCylinder &other) const {
            return other.m_value == m_value;
        }
    };

    struct PLANT_ARCHITECT_API BranchCylinderWidth : IDataComponent {
        float m_value;

        bool operator==(const BranchCylinderWidth &other) const {
            return other.m_value == m_value;
        }
    };

    struct PLANT_ARCHITECT_API BranchPointer : IDataComponent {
        glm::mat4 m_value;

        bool operator==(const BranchCylinder &other) const {
            return other.m_value == m_value;
        }
    };

    class PLANT_ARCHITECT_API InternodeSystem : public ISystem {
    public:
        void LateUpdate() override;

        void Simulate(float deltaTime);

        void OnCreate() override;

        void OnInspect() override;

        std::shared_ptr<Camera> m_internodeDebuggingCamera;
    private:
        /**
         * The EntityQuery for filtering all internodes.
         */
        EntityQuery m_internodesQuery;

        friend class Internode;

        std::vector<AssetRef> m_internodeBehaviours;

        void BehaviourSlotButton();

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
        glm::vec3 m_branchColor = glm::vec3(0, 1, 0);
        glm::vec4 m_pointerColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.5f);

        std::vector<Entity> m_entitiesWithRenderer;
        OpenGLUtils::GLVBO m_internodeColorBuffer;

        void UpdateBranchColors();
        void UpdateBranchCylinder(const float &width = 0.01f);
        void UpdateBranchPointer(const float &length,
                                 const float &width = 0.01f);
        void RenderBranchCylinders();
        void RenderBranchPointers();
#pragma endregion
#pragma endregion
    };
}