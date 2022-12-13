
//
// Created by lllll on 8/27/2021.
//
#include "InternodeLayer.hpp"
#include "InternodeModel/Internode.hpp"
#include "EditorLayer.hpp"
#include "DataComponents.hpp"
#include "GeneralTreeBehaviour.hpp"
#include "SpaceColonizationBehaviour.hpp"
#include "LSystemBehaviour.hpp"
#include "CubeVolume.hpp"
#include "RadialBoundingVolume.hpp"
#include "Joint.hpp"
#include "PhysicsLayer.hpp"
#include "DefaultInternodeResource.hpp"
#include "EmptyInternodeResource.hpp"
#include "DefaultInternodeFoliage.hpp"
#include "FBM.hpp"
#include "ClassRegistry.hpp"
#include "VoxelGrid.hpp"
#include "RenderLayer.hpp"
#include "SphereVolume.hpp"
#include "CylinderVolume.hpp"
#include "MeshVolume.hpp"
#include "PointCloudVolume.hpp"
#include "TreeGraph.hpp"
using namespace PlantArchitect;

void InternodeLayer::PreparePhysics(const Entity &entity, const Entity &child,
                                    const BranchPhysicsParameters &branchPhysicsParameters) {
    auto scene = GetScene();
    auto childBranchInfo = scene->GetDataComponent<InternodeBranchInfo>(child);
    if(!scene->HasPrivateComponent<RigidBody>(entity))
    {
        scene->RemovePrivateComponent<RigidBody>(child);
        scene->RemovePrivateComponent<Joint>(child);
	    return;
    }
    if(childBranchInfo.m_thickness < branchPhysicsParameters.m_minimumThickness) return;
    auto rigidBody = scene->GetOrSetPrivateComponent<RigidBody>(child).lock();
    rigidBody->SetEnableGravity(false);
    rigidBody->SetDensityAndMassCenter(branchPhysicsParameters.m_density *
                                       childBranchInfo.m_thickness *
                                       childBranchInfo.m_thickness * childBranchInfo.m_length);
    rigidBody->SetLinearDamping(branchPhysicsParameters.m_linearDamping);
    rigidBody->SetAngularDamping(branchPhysicsParameters.m_angularDamping);
    rigidBody->SetSolverIterations(branchPhysicsParameters.m_positionSolverIteration,
                                   branchPhysicsParameters.m_velocitySolverIteration);
    rigidBody->SetAngularVelocity(glm::vec3(0.0f));
    rigidBody->SetLinearVelocity(glm::vec3(0.0f));

    auto joint = scene->GetOrSetPrivateComponent<Joint>(child).lock();
    joint->Link(entity);
    joint->SetType(JointType::D6);
    joint->SetMotion(MotionAxis::SwingY, MotionType::Free);
    joint->SetMotion(MotionAxis::SwingZ, MotionType::Free);
    joint->SetDrive(DriveType::Swing,
                    glm::pow(childBranchInfo.m_thickness,
                             branchPhysicsParameters.m_jointDriveStiffnessThicknessFactor) *
                    branchPhysicsParameters.m_jointDriveStiffnessFactor,
                    glm::pow(childBranchInfo.m_thickness,
                             branchPhysicsParameters.m_jointDriveDampingThicknessFactor) *
                    branchPhysicsParameters.m_jointDriveDampingFactor,
                    branchPhysicsParameters.m_enableAccelerationForDrive);
}

void InternodeLayer::Simulate(int iterations) {
    auto scene = GetScene();

    for (int iteration = 0; iteration < iterations; iteration++) {
        m_voxelSpace.Clear();
        scene->ForEach<InternodeInfo, GlobalTransform>
                (Jobs::Workers(), m_internodesQuery,
                 [&](int i, Entity entity, InternodeInfo &internodeInfo, GlobalTransform &internodeGlobalTransform) {
                     const auto position = internodeGlobalTransform.GetPosition();
                     m_voxelSpace.Push(position, entity);
                 }, true);
        scene->ForEach<InternodeInfo, GlobalTransform>
                (Jobs::Workers(), m_internodesQuery,
                 [&](int i, Entity entity, InternodeInfo &internodeInfo, GlobalTransform &internodeGlobalTransform) {
                     const auto position = internodeGlobalTransform.GetPosition();
                     internodeInfo.m_neighborsProximity = 0;
                     m_voxelSpace.ForEachInRange(position, 16,
                                                 [&](const glm::vec3 &neighborPosition, const Entity &neighbor) {
                                                     if (neighbor == entity) return;
                                                     auto distance = glm::max(1.0f, glm::distance(position,
                                                                                                  neighborPosition));
                                                     if (distance < 16) {
                                                         internodeInfo.m_neighborsProximity +=
                                                                 1.0f / (distance * distance);
                                                     }
                                                 });
                 }, true);
        for (auto &i: m_plantBehaviours) {
            Preprocess(scene);
            if (i) i->Grow(scene, iteration);
        }
        ObstacleRemoval();
    }
    CalculateStatistics(scene);

    UpdateInternodeColors();
    UpdateInternodeCylinder();
    UpdateInternodePointer(m_pointerLength, m_pointerWidth);
}

void InternodeLayer::PreparePhysics() {
    auto physicsLayer = Application::GetLayer<PhysicsLayer>();
    if (!physicsLayer) return;

    auto scene = GetScene();
    for (auto &behaviour: m_plantBehaviours) {
        if (behaviour) {
            behaviour->GenerateSkinnedMeshes(scene, m_meshGeneratorSettings, false);
            std::vector<Entity> roots;
            scene->GetEntityArray(behaviour->m_rootsQuery, roots);
            for (const auto &root: roots) {
                auto children = scene->GetChildren(root);
                for (const auto &child: children) {
                    if (scene->HasPrivateComponent<Branch>(child)) {
                        auto rootRigidBody = scene->GetOrSetPrivateComponent<RigidBody>(root).lock();
                        if (!rootRigidBody->IsKinematic()) rootRigidBody->SetKinematic(true);
                        rootRigidBody->SetEnableGravity(false);
                        rootRigidBody->SetLinearDamping(m_branchPhysicsParameters.m_linearDamping);
                        rootRigidBody->SetAngularDamping(m_branchPhysicsParameters.m_angularDamping);
                        rootRigidBody->SetSolverIterations(m_branchPhysicsParameters.m_positionSolverIteration,
                                                           m_branchPhysicsParameters.m_velocitySolverIteration);
                        rootRigidBody->SetAngularVelocity(glm::vec3(0.0f));
                        rootRigidBody->SetLinearVelocity(glm::vec3(0.0f));

                        auto childBranchInfo = scene->GetDataComponent<InternodeBranchInfo>(child);
                        auto rigidBody = scene->GetOrSetPrivateComponent<RigidBody>(child).lock();
                        rigidBody->SetEnableGravity(false);
                        rigidBody->SetDensityAndMassCenter(m_branchPhysicsParameters.m_density *
                                                           childBranchInfo.m_thickness *
                                                           childBranchInfo.m_thickness * childBranchInfo.m_length);
                        rigidBody->SetLinearDamping(m_branchPhysicsParameters.m_linearDamping);
                        rigidBody->SetAngularDamping(m_branchPhysicsParameters.m_angularDamping);
                        rigidBody->SetSolverIterations(m_branchPhysicsParameters.m_positionSolverIteration,
                                                       m_branchPhysicsParameters.m_velocitySolverIteration);
                        rigidBody->SetAngularVelocity(glm::vec3(0.0f));
                        rigidBody->SetLinearVelocity(glm::vec3(0.0f));

                        auto joint = scene->GetOrSetPrivateComponent<Joint>(child).lock();
                        joint->Link(root);
                        joint->SetType(JointType::D6);
                        joint->SetMotion(MotionAxis::SwingY, MotionType::Free);
                        joint->SetMotion(MotionAxis::SwingZ, MotionType::Free);
                        joint->SetDrive(DriveType::Swing,
                                        glm::pow(childBranchInfo.m_thickness,
                                                 m_branchPhysicsParameters.m_jointDriveStiffnessThicknessFactor) *
                                        m_branchPhysicsParameters.m_jointDriveStiffnessFactor,
                                        glm::pow(childBranchInfo.m_thickness,
                                                 m_branchPhysicsParameters.m_jointDriveDampingThicknessFactor) *
                                        m_branchPhysicsParameters.m_jointDriveDampingFactor,
                                        m_branchPhysicsParameters.m_enableAccelerationForDrive);

                        behaviour->BranchGraphWalkerRootToEnd(scene, child, [&](Entity parent, Entity child) {
                            PreparePhysics(parent, child, m_branchPhysicsParameters);
                        });
                        break;
                    }
                }
            }
        }
    }
    physicsLayer->UploadRigidBodyShapes(scene);
    physicsLayer->UploadTransforms(scene, true);
    physicsLayer->UploadJointLinks(scene);
}

#pragma region Methods

void InternodeLayer::OnInspect() {
    auto scene = GetScene();
    if (ImGui::Begin("Internode Manager")) {
        static int iterations = 1;
        ImGui::DragInt("Iterations", &iterations);
        static bool autoGenMeshes = false;
        ImGui::Checkbox("Auto gen meshes", &autoGenMeshes);

        if (ImGui::Button("Simulate")) {
            Simulate(iterations);
            if (autoGenMeshes) {
                for (const auto& behaviour : m_plantBehaviours) {
                    behaviour->GenerateSkinnedMeshes(scene, m_meshGeneratorSettings);
                }
            }
        }
        if (ImGui::Button("Prepare physics")) PreparePhysics();
        if (ImGui::TreeNode("Physics")) {
            m_branchPhysicsParameters.OnInspect();
            ImGui::Checkbox("Apply FBM", &m_applyFBMField);
            if (m_applyFBMField) {
                if (ImGui::TreeNodeEx("FBM Settings")) {
                    m_fBMField.OnInspect();
                    ImGui::TreePop();
                }
            }
            ImGui::DragFloat("Force factor", &m_forceFactor, 0.1f);
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Voxels", ImGuiTreeNodeFlags_DefaultOpen)) {
            m_voxelSpace.OnInspect();
            ImGui::TreePop();
        }
        if (m_voxelSpace.m_display) {
            auto editorLayer = Application::GetLayer<EditorLayer>();
            if (editorLayer) {
                Gizmos::DrawGizmoMeshInstanced(
                        DefaultResources::Primitives::Cube, m_visualizationCamera,
                        editorLayer->m_sceneCameraPosition,
                        editorLayer->m_sceneCameraRotation,
                        glm::vec4(1, 1, 1, 0.2),
                        m_voxelSpace.m_frozenVoxels);
            }
            Gizmos::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube,
                                             glm::vec4(1, 1, 1, 0.2),
                                             m_voxelSpace.m_frozenVoxels, glm::mat4(1.0f), 1.0f);
        }

        if (ImGui::TreeNodeEx("Plant Behaviour", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::TreeNodeEx("Meshes", ImGuiTreeNodeFlags_DefaultOpen)) {
                m_meshGeneratorSettings.OnInspect();
                if (ImGui::Button("Generate mesh for all trees")) {
                    for (const auto &behaviour: m_plantBehaviours) {
                        behaviour->GenerateSkinnedMeshes(scene, m_meshGeneratorSettings);
                    }
                }
                if (ImGui::TreeNodeEx("Subtree", ImGuiTreeNodeFlags_DefaultOpen)) {
                    static SubtreeSettings subtreeSettings;
                    subtreeSettings.OnInspect();
                    static EntityRef internodeEntityRef;
                    ImGui::Button("Drop base internode here");
                    if (Editor::Droppable(internodeEntityRef)) {
                        auto internodeEntity = internodeEntityRef.Get();
                        if (scene->IsEntityValid(internodeEntity)) {
                            internodeEntityRef.Clear();
                            for (const auto &behaviour: m_plantBehaviours) {
                                if (behaviour->InternodeCheck(scene, internodeEntity)) {
                                    behaviour->CreateSubtree(scene, internodeEntity, subtreeSettings);
                                    break;
                                }
                            }
                        }
                    }
                    ImGui::TreePop();
                }

                if (ImGui::TreeNodeEx("Branchlets", ImGuiTreeNodeFlags_DefaultOpen)) {
                    static float branchletDistance = 3.0f;
                    ImGui::DragFloat("Branchlet distance", &branchletDistance);
                    static EntityRef rootEntityRef;
                    static std::shared_ptr<IPlantBehaviour> rootBehaviour;
                    static Entity rootEntity = {};
                    static EntityRef branchletCandidateRef[9];
                    if (Editor::DragAndDropButton(rootEntityRef, "InternodePlant")) {
                        rootEntity = rootEntityRef.Get();
                        bool found = false;
                        for (const auto &behaviour: m_plantBehaviours) {
                            if (behaviour->RootCheck(scene, rootEntity)) {
                                rootBehaviour = behaviour;
                                found = true;
                            }
                        }
                        if (!found) {
                            rootEntityRef.Clear();
                            rootBehaviour.reset();
                            rootEntity = {};
                            for (auto &i: branchletCandidateRef) {
                                i.Clear();
                            }
                        }
                    }
                    if (!rootBehaviour) {
                        ImGui::Text("Assign root to continue");
                    } else if (!scene->IsEntityValid(rootEntity)) {
                        rootEntityRef.Clear();
                        rootBehaviour.reset();
                        rootEntity = {};
                        for (auto &i: branchletCandidateRef) {
                            i.Clear();
                        }
                    } else {
                        for (int i = 0; i < 9; i++) {
                            Entity branchletEntity;
                            if (Editor::DragAndDropButton(branchletCandidateRef[i],
                                                          std::string("Internode") + "[" + std::to_string(i) + "]")) {
                                branchletEntity = branchletCandidateRef[i].Get();
                                if (!rootBehaviour->InternodeCheck(scene, branchletEntity)) {
                                    branchletCandidateRef[i].Clear();
                                }
                            }
                        }
                        if (ImGui::Button("Instantiate tree and branchlets")) {
                            scene->ForEach<InternodeColor>(
                                    Jobs::Workers(),
                                    m_internodesQuery,
                                    [=](int i, Entity entity,
                                        InternodeColor &internodeRenderColor) {
                                        internodeRenderColor.m_value = glm::vec4(1.0f);
                                    },
                                    true);
                            auto branchletsEntity = scene->CreateEntity("Branchlets");
                            for (int i = 0; i < 9; i++) {
                                Entity branchletEntity = branchletCandidateRef[i].Get();
                                if (rootBehaviour->InternodeCheck(scene, branchletEntity)) {
                                    InternodeColor internodeColor;

                                    internodeColor.m_value = glm::vec4(m_randomColors[i], 1.0f);
                                    auto children = scene->GetChildren(branchletEntity);
                                    for (auto &child: children) {
                                        scene->SetDataComponent(child, internodeColor);
                                    }
                                    SubtreeSettings branchletSubtreeSettings;
                                    branchletSubtreeSettings.m_enableBaseInternode = false;
                                    branchletSubtreeSettings.m_lineColor = glm::vec3(0, 0, 0);
                                    branchletSubtreeSettings.m_lineLengthFactor = 0.5f;
                                    branchletSubtreeSettings.m_layer = 2;
                                    auto subtree = rootBehaviour->CreateSubtree(scene, branchletEntity,
                                                                                branchletSubtreeSettings);
                                    auto gt = scene->GetDataComponent<GlobalTransform>(branchletEntity);
                                    auto internodeInfo = scene->GetDataComponent<InternodeInfo>(branchletEntity);
                                    GlobalTransform subtreeGT;
                                    subtreeGT.m_value = glm::inverse(gt.m_value);
                                    subtreeGT.SetPosition(subtreeGT.GetPosition() + glm::vec3(0, i / 3, i % 3) * 3.0f);
                                    scene->SetDataComponent(subtree, subtreeGT);
                                    auto branchlet = scene->CreateEntity(std::string("Branchlet " + std::to_string(i)));
                                    scene->SetParent(subtree, branchlet);
                                    scene->SetParent(branchlet, branchletsEntity);
                                }
                            }
                            SubtreeSettings rootSubtreeSettings;
                            rootSubtreeSettings.m_enableBaseInternode = true;
                            rootSubtreeSettings.m_layer = 999;
                            rootSubtreeSettings.m_enableLines = false;
                            auto rootChildren = scene->GetChildren(rootEntity);
                            for (const auto &child: rootChildren) {
                                if (rootBehaviour->InternodeCheck(scene, child))
                                    rootBehaviour->CreateSubtree(scene, child, rootSubtreeSettings);
                            }
                        }
                    }
                    ImGui::TreePop();
                }

                ImGui::TreePop();
            }

            for (const auto &behaviour: m_plantBehaviours) {
                if (ImGui::TreeNodeEx(behaviour->GetTypeName().c_str())) {
                    behaviour->OnInspect();
                    ImGui::TreePop();
                }
            }
            ImGui::TreePop();
        }
    }
    ImGui::End();
#pragma region Internode debugging camera
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    ImVec2 viewPortSize;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0});
    ImGui::Begin("Plant Visual");
    {
        if (ImGui::BeginChild("InternodeCameraRenderer", ImVec2(0, 0), false, ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{5, 5});
                if (ImGui::BeginMenu("Settings")) {
#pragma region Menu
                    ImGui::ColorEdit3("Background color", &m_visualizationCamera->m_clearColor.x);
                    ImGui::Checkbox("Auto update", &m_autoUpdate);
                    ImGui::Checkbox("Internodes", &m_drawInternodes);
                    if (m_drawInternodes) {
                        if (ImGui::TreeNodeEx("Internode settings",
                                              ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::Text("Current Internode amount: %zu",
                                        scene->GetEntityAmount(m_internodesQuery, false));
                            ImGui::ColorEdit3("Base color", &m_internodeColor.x);
                            ImGui::SliderFloat("Transparency", &m_internodeTransparency, 0, 1);
                            DrawColorModeSelectionMenu();
                            ImGui::TreePop();
                        }
                    }
                    ImGui::Checkbox("Branches", &m_drawBranches);
                    if (m_drawBranches) {
                        if (ImGui::TreeNodeEx("Branch settings",
                                              ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::Text("Current Branch amount: %zu",
                                        scene->GetEntityAmount(m_branchesQuery, false));

                            ImGui::SliderFloat("Transparency", &m_branchTransparency, 0, 1);
                            ImGui::TreePop();
                        }
                    }
                    ImGui::Checkbox("Pointers", &m_drawPointers);
                    if (m_drawPointers) {
                        if (ImGui::TreeNodeEx("Pointer settings",
                                              ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::ColorEdit4("Pointer color", &m_pointerColor.x);
                            ImGui::DragFloat("Pointer length", &m_pointerLength, 0.01f, 0.01f, 3.0f);
                            ImGui::DragFloat("Pointer width", &m_pointerWidth, 0.01f, 0.01f, 1.0f);
                            ImGui::TreePop();
                        }
                    }
#pragma endregion
                    ImGui::EndMenu();
                }
                ImGui::PopStyleVar();
                ImGui::EndMenuBar();
            }
            viewPortSize = ImGui::GetWindowSize();
            m_visualizationCameraResolutionX = viewPortSize.x;
            m_visualizationCameraResolutionY = viewPortSize.y - 20;
            ImGui::Image(
                    reinterpret_cast<ImTextureID>(
                            m_visualizationCamera->GetTexture()->UnsafeGetGLTexture()->Id()),
                    ImVec2(viewPortSize.x, viewPortSize.y - 20), ImVec2(0, 1), ImVec2(1, 0));
            glm::vec2 mousePosition = glm::vec2(FLT_MAX, FLT_MIN);
            if (ImGui::IsWindowFocused()) {
                bool valid = true;
                auto mp = ImGui::GetMousePos();
                auto wp = ImGui::GetWindowPos();
                mousePosition = glm::vec2(mp.x - wp.x, mp.y - wp.y - 20);
                float xOffset = 0;
                float yOffset = 0;
                if (valid) {
                    if (!m_startMouse) {
                        m_lastX = mousePosition.x;
                        m_lastY = mousePosition.y;
                        m_startMouse = true;
                    }
                    xOffset = mousePosition.x - m_lastX;
                    yOffset = -mousePosition.y + m_lastY;
                    m_lastX = mousePosition.x;
                    m_lastY = mousePosition.y;
#pragma region Scene Camera Controller
                    if (!m_rightMouseButtonHold &&
                        Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT,
                                                 Windows::GetWindow())) {
                        m_rightMouseButtonHold = true;
                    }
                    if (m_rightMouseButtonHold &&
                        !editorLayer->m_lockCamera) {
                        glm::vec3 front =
                                editorLayer->m_sceneCameraRotation *
                                glm::vec3(0, 0, -1);
                        glm::vec3 right =
                                editorLayer->m_sceneCameraRotation *
                                glm::vec3(1, 0, 0);
                        if (Inputs::GetKeyInternal(GLFW_KEY_W,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition +=
                                    front * static_cast<float>(Application::Time().DeltaTime()) *
                                    editorLayer->m_velocity;
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_S,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition -=
                                    front * static_cast<float>(Application::Time().DeltaTime()) *
                                    editorLayer->m_velocity;
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_A,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition -=
                                    right * static_cast<float>(Application::Time().DeltaTime()) *
                                    editorLayer->m_velocity;
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_D,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition +=
                                    right * static_cast<float>(Application::Time().DeltaTime()) *
                                    editorLayer->m_velocity;
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_LEFT_SHIFT,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition.y +=
                                    editorLayer->m_velocity *
                                    static_cast<float>(Application::Time().DeltaTime());
                        }
                        if (Inputs::GetKeyInternal(GLFW_KEY_LEFT_CONTROL,
                                                   Windows::GetWindow())) {
                            editorLayer->m_sceneCameraPosition.y -=
                                    editorLayer->m_velocity *
                                    static_cast<float>(Application::Time().DeltaTime());
                        }
                        if (xOffset != 0.0f || yOffset != 0.0f) {
                            editorLayer->m_sceneCameraYawAngle +=
                                    xOffset * editorLayer->m_sensitivity;
                            editorLayer->m_sceneCameraPitchAngle +=
                                    yOffset * editorLayer->m_sensitivity;
                            if (editorLayer->m_sceneCameraPitchAngle > 89.0f)
                                editorLayer->m_sceneCameraPitchAngle = 89.0f;
                            if (editorLayer->m_sceneCameraPitchAngle < -89.0f)
                                editorLayer->m_sceneCameraPitchAngle = -89.0f;

                            editorLayer->m_sceneCameraRotation =
                                    Camera::ProcessMouseMovement(
                                            editorLayer->m_sceneCameraYawAngle,
                                            editorLayer->m_sceneCameraPitchAngle,
                                            false);
                        }
                    }
#pragma endregion
                    if (m_drawInternodes) {
#pragma region Ray selection
                        m_currentFocusingInternode = Entity();
                        std::mutex writeMutex;
                        float minDistance = FLT_MAX;
                        GlobalTransform cameraLtw;
                        cameraLtw.m_value =
                                glm::translate(
                                        editorLayer->m_sceneCameraPosition) *
                                glm::mat4_cast(
                                        editorLayer->m_sceneCameraRotation);
                        const Ray cameraRay = m_visualizationCamera->ScreenPointToRay(
                                cameraLtw, mousePosition);
                        scene->ForEach<GlobalTransform, InternodeInfo>(Jobs::Workers(),
                                                                       m_internodesQuery,
                                                                       [&, cameraLtw, cameraRay](int i, Entity entity,
                                                                                                 GlobalTransform &ltw,
                                                                                                 InternodeInfo &internodeInfo) {
                                                                           const glm::vec3 position = ltw.m_value[3];
                                                                           const glm::vec3 position2 = position +
                                                                                                       internodeInfo.m_length *
                                                                                                       glm::normalize(
                                                                                                               ltw.GetRotation() *
                                                                                                               glm::vec3(
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       -1));
                                                                           const auto center =
                                                                                   (position + position2) / 2.0f;
                                                                           auto dir = cameraRay.m_direction;
                                                                           auto pos = cameraRay.m_start;
                                                                           const auto radius = internodeInfo.m_thickness;
                                                                           const auto height = glm::distance(position2,
                                                                                                             position);
                                                                           if (!cameraRay.Intersect(center,
                                                                                                    height / 2.0f))
                                                                               return;

#pragma region Line Line intersection
                                                                           /*
                                                        * http://geomalgorithms.com/a07-_distance.html
                                                        */
                                                                           glm::vec3 u = pos - (pos + dir);
                                                                           glm::vec3 v = position - position2;
                                                                           glm::vec3 w = (pos + dir) - position2;
                                                                           const auto a = dot(u,
                                                                                              u); // always >= 0
                                                                           const auto b = dot(u, v);
                                                                           const auto c = dot(v,
                                                                                              v); // always >= 0
                                                                           const auto d = dot(u, w);
                                                                           const auto e = dot(v, w);
                                                                           const auto dotP =
                                                                                   a * c - b * b; // always >= 0
                                                                           float sc, tc;
                                                                           // compute the line parameters of the two closest points
                                                                           if (dotP <
                                                                               0.001f) { // the lines are almost parallel
                                                                               sc = 0.0f;
                                                                               tc = (b > c ? d / b
                                                                                           : e /
                                                                                             c); // use the largest denominator
                                                                           } else {
                                                                               sc = (b * e - c * d) / dotP;
                                                                               tc = (a * e - b * d) / dotP;
                                                                           }
                                                                           // get the difference of the two closest points
                                                                           glm::vec3 dP = w + sc * u -
                                                                                          tc * v; // =  L1(sc) - L2(tc)
                                                                           if (glm::length(dP) > radius)
                                                                               return;
#pragma endregion

                                                                           const auto distance = glm::distance(
                                                                                   glm::vec3(cameraLtw.m_value[3]),
                                                                                   glm::vec3(center));
                                                                           std::lock_guard<std::mutex> lock(writeMutex);
                                                                           if (distance < minDistance) {
                                                                               minDistance = distance;
                                                                               m_currentFocusingInternode = entity;
                                                                           }
                                                                       });
                        if (Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_LEFT,
                                                     Windows::GetWindow())) {
                            if (m_currentFocusingInternode.Get().GetIndex() != 0) {
                                editorLayer->SetSelectedEntity(
                                        m_currentFocusingInternode.Get());
                            }
                        }
#pragma endregion
                    }
                }
            }
        }
        ImGui::EndChild();
        auto *window = ImGui::FindWindowByName("Plant Visual");
        m_visualizationCamera->SetEnabled(
                !(window->Hidden && !window->Collapsed));
    }
    ImGui::End();
    ImGui::PopStyleVar();

#pragma endregion
}

void InternodeLayer::OnCreate() {
    ClassRegistry::RegisterDataComponent<InternodeCylinder>("InternodeCylinder");
    ClassRegistry::RegisterDataComponent<InternodeCylinderWidth>("InternodeCylinderWidth");
    ClassRegistry::RegisterDataComponent<InternodePointer>("InternodePointer");
    ClassRegistry::RegisterDataComponent<InternodeColor>("InternodeColor");
    ClassRegistry::RegisterDataComponent<InternodeBranchCylinder>("InternodeBranchCylinder");
    ClassRegistry::RegisterDataComponent<InternodeBranchCylinderWidth>("InternodeBranchCylinderWidth");
    ClassRegistry::RegisterDataComponent<InternodeBranchColor>("InternodeBranchColor");

    ClassRegistry::RegisterPrivateComponent<IVolume>("IVolume");
    ClassRegistry::RegisterPrivateComponent<CubeVolume>("CubeVolume");
    ClassRegistry::RegisterPrivateComponent<SphereVolume>("SphereVolume");
    ClassRegistry::RegisterPrivateComponent<CylinderVolume>("CylinderVolume");
    ClassRegistry::RegisterPrivateComponent<MeshVolume>("MeshVolume");
    ClassRegistry::RegisterPrivateComponent<PointCloudVolume>("PointCloudVolume");
    ClassRegistry::RegisterPrivateComponent<RadialBoundingVolume>("RadialBoundingVolume");


    ClassRegistry::RegisterDataComponent<GeneralTreeTag>("GeneralTreeTag");
    ClassRegistry::RegisterAsset<GeneralTreeParameters>("GeneralTreeParameters", {".gtparams"});

    ClassRegistry::RegisterAsset<TreeGraph>("TreeGraph", {".treegraph"});
    ClassRegistry::RegisterAsset<VoxelGrid>("VoxelGrid", {".vg"});

    ClassRegistry::RegisterDataComponent<InternodeStatus>("InternodeStatus");
    ClassRegistry::RegisterDataComponent<InternodeWaterPressure>("InternodeWaterPressure");
    ClassRegistry::RegisterDataComponent<InternodeWater>("InternodeWater");
    ClassRegistry::RegisterDataComponent<InternodeIllumination>("InternodeIllumination");
    ClassRegistry::RegisterPrivateComponent<InternodeWaterFeeder>("InternodeWaterFeeder");

    ClassRegistry::RegisterDataComponent<SpaceColonizationTag>("SpaceColonizationTag");
    ClassRegistry::RegisterAsset<SpaceColonizationParameters>("SpaceColonizationParameters", {".scparams"});
    ClassRegistry::RegisterDataComponent<SpaceColonizationIncentive>("SpaceColonizationIncentive");

    ClassRegistry::RegisterAsset<LSystemString>("LSystemString", {".lstring"});
    ClassRegistry::RegisterDataComponent<LSystemTag>("LSystemTag");

    ClassRegistry::RegisterSerializable<EmptyInternodeResource>("EmptyInternodeResource");
    ClassRegistry::RegisterSerializable<DefaultInternodeResource>("DefaultInternodeResource");
    ClassRegistry::RegisterSerializable<Bud>("LateralBud");
    ClassRegistry::RegisterPrivateComponent<Internode>("Internode");
    ClassRegistry::RegisterPrivateComponent<Branch>("Branch");
    ClassRegistry::RegisterPrivateComponent<InternodePlant>("InternodePlant");

    ClassRegistry::RegisterDataComponent<InternodeInfo>("InternodeInfo");
    ClassRegistry::RegisterDataComponent<InternodeRootInfo>("InternodeRootInfo");
    ClassRegistry::RegisterDataComponent<InternodeBranchInfo>("InternodeBranchInfo");
    ClassRegistry::RegisterDataComponent<InternodeStatistics>("InternodeStatistics");

    ClassRegistry::RegisterAsset<DefaultInternodeFoliage>("DefaultInternodeFoliage", {".defaultip"});

    Editor::RegisterComponentDataInspector<InternodeInfo>([](Entity entity, IDataComponent *data, bool isRoot) {
        auto *ltw = reinterpret_cast<InternodeInfo *>(data);
        ltw->OnInspect();
        return false;
    });

    Editor::RegisterComponentDataInspector<GeneralTreeParameters>(
            [](Entity entity, IDataComponent *data, bool isRoot) {
                auto *ltw = reinterpret_cast<GeneralTreeParameters *>(data);
                ltw->OnInspect();
                return false;
            });

    Editor::RegisterComponentDataInspector<InternodeStatus>(
            [](Entity entity, IDataComponent *data, bool isRoot) {
                auto *ltw = reinterpret_cast<InternodeStatus *>(data);
                ltw->OnInspect();
                return false;
            });

    Editor::RegisterComponentDataInspector<InternodeWaterPressure>(
            [](Entity entity, IDataComponent *data, bool isRoot) {
                auto *ltw = reinterpret_cast<InternodeWaterPressure *>(data);
                ltw->OnInspect();
                return false;
            });

    Editor::RegisterComponentDataInspector<InternodeStatistics>(
            [](Entity entity, IDataComponent *data, bool isRoot) {
                auto *ltw = reinterpret_cast<InternodeStatistics *>(data);
                ltw->OnInspect();
                return false;
            });

    Editor::RegisterComponentDataInspector<InternodeColor>(
            [](Entity entity, IDataComponent *data, bool isRoot) {
                auto *ltw = reinterpret_cast<InternodeColor *>(data);
                ltw->OnInspect();
                return false;
            });

    Editor::RegisterComponentDataInspector<InternodeCylinderWidth>(
            [](Entity entity, IDataComponent *data, bool isRoot) {
                auto *ltw = reinterpret_cast<InternodeCylinderWidth *>(data);
                ltw->OnInspect();
                return false;
            });

    Editor::RegisterComponentDataInspector<InternodeWater>([](Entity entity, IDataComponent *data, bool isRoot) {
        auto *ltw = reinterpret_cast<InternodeWater *>(data);
        ltw->OnInspect();
        return false;
    });

    Editor::RegisterComponentDataInspector<InternodeIllumination>(
            [](Entity entity, IDataComponent *data, bool isRoot) {
                auto *ltw = reinterpret_cast<InternodeIllumination *>(data);
                ltw->OnInspect();
                return false;
            });

    Editor::RegisterComponentDataInspector<SpaceColonizationParameters>(
            [](Entity entity, IDataComponent *data, bool isRoot) {
                auto *ltw = reinterpret_cast<SpaceColonizationParameters *>(data);
                ltw->OnInspect();
                return false;
            });

    m_plantBehaviours.push_back(std::make_shared<GeneralTreeBehaviour>());
    m_plantBehaviours.push_back(std::make_shared<SpaceColonizationBehaviour>());
    m_plantBehaviours.push_back(std::make_shared<LSystemBehaviour>());

    m_randomColors.resize(8192);
    for (int i = 0; i < 8192; i++) {
        m_randomColors[i] = glm::abs(glm::sphericalRand(1.0f));
    }

    m_internodesQuery = Entities::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeInfo());
    m_branchesQuery = Entities::CreateEntityQuery();
    m_branchesQuery.SetAllFilters(InternodeBranchInfo());

#pragma region Internode camera
    m_visualizationCamera =
            Serialization::ProduceSerializable<Camera>();
    m_visualizationCamera->m_useClearColor = true;
    m_visualizationCamera->m_clearColor = glm::vec3(0.1f);
    m_visualizationCamera->OnCreate();
#pragma endregion

    m_voxelSpace.Reset();
}


void InternodeLayer::LateUpdate() {
    UpdateInternodeCamera();
    Application::GetActiveScene()->GetOrCreateSystem<PhysicsSystem>(SystemGroup::SimulationSystemGroup);
}

void InternodeLayer::UpdateInternodeColors() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    auto focusingInternode = Entity();
    auto selectedEntity = Entity();
    auto scene = GetScene();
    if (scene->IsEntityValid(m_currentFocusingInternode.Get())) {
        focusingInternode = m_currentFocusingInternode.Get();
    }
    if (scene->IsEntityValid(editorLayer->GetSelectedEntity())) {
        selectedEntity = editorLayer->GetSelectedEntity();
    }

    scene->ForEach<InternodeColor, InternodeInfo>(
            Jobs::Workers(),
            m_internodesQuery,
            [=](int i, Entity entity, InternodeColor &internodeRenderColor,
                InternodeInfo &internodeInfo) {
                internodeRenderColor.m_value = glm::vec4(m_internodeColor, m_internodeTransparency);
            },
            true);

    switch (m_branchColorMode) {
        case BranchColorMode::SubTree: {
            ColorSubTree(scene, selectedEntity, 0);
        }
            break;
        case BranchColorMode::Branchlet: {
            ColorBranchlet(scene);
        }
            break;
        case BranchColorMode::Order:
            scene->ForEach<InternodeColor, InternodeInfo>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeInfo &internodeInfo) {
                        internodeRenderColor.m_value = glm::vec4(m_randomColors[internodeInfo.m_order],
                                                                 m_internodeTransparency);
                    },
                    true);
            break;
        case BranchColorMode::Level:
            scene->ForEach<InternodeColor, InternodeStatus>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeStatus &internodeStatus) {
                        internodeRenderColor.m_value = glm::vec4(
                                glm::vec3(m_internodeColorValueMultiplier *
                                          glm::pow(
                                                  (float) internodeStatus.m_level,
                                                  m_internodeColorValueCompressFactor)),
                                m_internodeTransparency);
                    },
                    true);
            break;
        case BranchColorMode::ApicalControl:
            scene->ForEach<InternodeColor, InternodeStatus, InternodeInfo, GeneralTreeParameters>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity, InternodeColor &internodeRenderColor,
                        InternodeStatus &internodeStatus, InternodeInfo &internodeInfo,
                        GeneralTreeParameters &parameters) {
                        internodeRenderColor.m_value = glm::vec4(glm::vec3(m_internodeColorValueMultiplier *
                                                                           glm::pow(internodeStatus.m_apicalControl,
                                                                                    m_internodeColorValueCompressFactor)),
                                                                 m_internodeTransparency);
                    },
                    true);
            break;
        case BranchColorMode::Water:
            scene->ForEach<InternodeColor, InternodeWater>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeWater &internodeWater) {
                        internodeRenderColor.m_value = glm::vec4(
                                glm::vec3(m_internodeColorValueMultiplier *
                                          glm::pow(internodeWater.m_value,
                                                   m_internodeColorValueCompressFactor)),
                                m_internodeTransparency);
                    },
                    true);
            break;
        case BranchColorMode::WaterPressure:
            scene->ForEach<InternodeColor, InternodeWaterPressure>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeWaterPressure &internodeWaterPressure) {
                        internodeRenderColor.m_value = glm::vec4(
                                glm::vec3(
                                        m_internodeColorValueMultiplier *
                                        glm::pow(
                                                internodeWaterPressure.m_value,
                                                m_internodeColorValueCompressFactor)),
                                m_internodeTransparency);
                    },
                    true);
            break;
        case BranchColorMode::Proximity:
            scene->ForEach<InternodeColor, InternodeInfo>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeInfo &internodeInfo) {
                        internodeRenderColor.m_value = glm::vec4(
                                glm::vec3(m_internodeColorValueMultiplier *
                                          glm::pow(
                                                  internodeInfo.m_neighborsProximity,
                                                  m_internodeColorValueCompressFactor)),
                                m_internodeTransparency);
                    },
                    true);
            break;
        case BranchColorMode::Inhibitor:
            scene->ForEach<InternodeColor, InternodeStatus>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeStatus &internodeStatus) {
                        internodeRenderColor.m_value = glm::vec4(
                                glm::vec3(m_internodeColorValueMultiplier *
                                          glm::pow(
                                                  internodeStatus.m_inhibitor,
                                                  m_internodeColorValueCompressFactor)),
                                m_internodeTransparency);
                    },
                    true);
            break;
        case BranchColorMode::IndexDivider:
            scene->ForEach<InternodeColor, InternodeStatistics>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        internodeRenderColor.m_value = glm::vec4(
                                glm::vec3(
                                        m_randomColors[
                                                internodeStatistics.m_lSystemStringIndex /
                                                m_indexDivider]),
                                1.0f);
                    },
                    true);
            break;
        case BranchColorMode::IndexRange:
            scene->ForEach<InternodeColor, InternodeStatistics>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        glm::vec3 color = glm::vec3(1.0f);
                        if (internodeStatistics.m_lSystemStringIndex >
                            m_indexRangeMin &&
                            internodeStatistics.m_lSystemStringIndex <
                            m_indexRangeMax) {
                            color = glm::vec3(0.0f, 0.0f, 1.0f);
                        }
                        internodeRenderColor.m_value = glm::vec4(color,
                                                                 1.0f);
                    },
                    true);
            break;
        case BranchColorMode::StrahlerNumber:
            scene->ForEach<InternodeColor, InternodeStatistics>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        internodeRenderColor.m_value = glm::vec4(
                                glm::vec3(
                                        m_randomColors[
                                                internodeStatistics.m_strahlerOrder]),
                                1.0f);
                    },
                    true);
            break;
        case BranchColorMode::ChildCount:
            scene->ForEach<InternodeColor, InternodeStatistics>(
                    Jobs::Workers(),
                    m_internodesQuery,
                    [=](int i, Entity entity,
                        InternodeColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        internodeRenderColor.m_value = m_childCountColors[internodeStatistics.m_childCount];
                    },
                    true);
            break;
        default:
            break;
    }


    InternodeColor color;
    color.m_value = glm::vec4(1, 1, 1, 1);
    if (scene->IsEntityValid(focusingInternode) && scene->HasDataComponent<InternodeColor>(focusingInternode))
        scene->SetDataComponent(focusingInternode, color);
    if (m_branchColorMode != BranchColorMode::Branchlet && m_branchColorMode != BranchColorMode::SubTree) {
        color.m_value = glm::vec4(1, 0, 0, 1);
        if (scene->IsEntityValid(selectedEntity) && scene->HasDataComponent<InternodeColor>(selectedEntity))
            scene->SetDataComponent(selectedEntity, color);
    }


}

void InternodeLayer::UpdateInternodeCylinder() {
    auto scene = GetScene();
    scene->ForEach<GlobalTransform, InternodeCylinder, InternodeCylinderWidth, InternodeInfo>(
            Jobs::Workers(),
            m_internodesQuery,
            [&](int i, Entity entity, GlobalTransform &ltw, InternodeCylinder &c,
                InternodeCylinderWidth &branchCylinderWidth, InternodeInfo &internodeInfo) {
                glm::vec3 scale;
                glm::quat rotation;
                glm::vec3 translation;
                glm::vec3 skew;
                glm::vec4 perspective;
                glm::decompose(ltw.m_value, scale, rotation, translation, skew,
                               perspective);
                const auto direction = glm::normalize(rotation * glm::vec3(0, 0, -1));
                const glm::vec3 position2 =
                        translation + internodeInfo.m_length * direction;
                rotation = glm::quatLookAt(
                        direction, glm::vec3(direction.y, direction.z, direction.x));
                rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
                const glm::mat4 rotationTransform = glm::mat4_cast(rotation);
                float thickness = internodeInfo.m_thickness;
                if (m_overrideThickness) thickness = m_internodeThickness;
                if (m_hideUnnecessaryInternodes && !internodeInfo.m_display) thickness = 0.0f;
                branchCylinderWidth.m_value = thickness;
                c.m_value =
                        glm::translate((translation + position2) / 2.0f) *
                        rotationTransform *
                        glm::scale(glm::vec3(
                                branchCylinderWidth.m_value,
                                glm::distance(translation, position2) / 2.0f,
                                thickness));

            },
            true);
}

void InternodeLayer::UpdateBranchCylinder() {
    auto scene = GetScene();
    scene->ForEach<GlobalTransform, InternodeBranchCylinder, InternodeBranchCylinderWidth, InternodeBranchInfo>(
            Jobs::Workers(),
            m_branchesQuery,
            [](int i, Entity entity, GlobalTransform &ltw, InternodeBranchCylinder &c,
               InternodeBranchCylinderWidth &branchCylinderWidth, InternodeBranchInfo &internodeInfo) {
                glm::vec3 scale;
                glm::quat rotation;
                glm::vec3 translation;
                glm::vec3 skew;
                glm::vec4 perspective;
                glm::decompose(ltw.m_value, scale, rotation, translation, skew,
                               perspective);
                const auto direction = glm::normalize(rotation * glm::vec3(0, 0, -1));
                const glm::vec3 position2 =
                        translation - internodeInfo.m_length * direction;
                rotation = glm::quatLookAt(
                        direction, glm::vec3(-direction.y, -direction.z, -direction.x));
                rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
                const glm::mat4 rotationTransform = glm::mat4_cast(rotation);
                branchCylinderWidth.m_value = internodeInfo.m_thickness;
                c.m_value =
                        glm::translate((translation + position2) / 2.0f) *
                        rotationTransform *
                        glm::scale(glm::vec3(
                                branchCylinderWidth.m_value,
                                glm::distance(translation, position2) / 2.0f,
                                internodeInfo.m_thickness));

            },
            true);
}

void InternodeLayer::UpdateInternodePointer(const float &length, const float &width) {


}

void InternodeLayer::RenderInternodeCylinders() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    std::vector<InternodeCylinder> branchCylinders;
    auto scene = GetScene();
    scene->GetComponentDataArray<InternodeCylinder>(m_internodesQuery, branchCylinders);
    std::vector<InternodeColor> branchColors;
    scene->GetComponentDataArray<InternodeColor>(m_internodesQuery,
                                                 branchColors);
    if (!branchCylinders.empty())
        Gizmos::DrawGizmoMeshInstancedColored(
                DefaultResources::Primitives::Cylinder, m_visualizationCamera,
                editorLayer->m_sceneCameraPosition,
                editorLayer->m_sceneCameraRotation,
                *reinterpret_cast<std::vector<glm::vec4> *>(&branchColors),
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchCylinders),
                glm::mat4(1.0f), 1.0f);
}

void InternodeLayer::RenderInternodePointers() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    std::vector<InternodePointer> branchPointers;
    auto scene = GetScene();
    scene->GetComponentDataArray<InternodePointer>(m_internodesQuery,
                                                   branchPointers);
    if (!branchPointers.empty())
        Gizmos::DrawGizmoMeshInstanced(
                DefaultResources::Primitives::Cylinder, m_visualizationCamera,
                editorLayer->m_sceneCameraPosition,
                editorLayer->m_sceneCameraRotation, m_pointerColor,
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchPointers),
                glm::mat4(1.0f), 1.0f);
}


void InternodeLayer::UpdateInternodeCamera() {
    if (m_rightMouseButtonHold &&
        !Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT,
                                  Windows::GetWindow())) {
        m_rightMouseButtonHold = false;
        m_startMouse = false;
    }
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;

    m_visualizationCamera->ResizeResolution(
            m_visualizationCameraResolutionX,
            m_visualizationCameraResolutionY);
    m_visualizationCamera->Clear();
    auto renderLayer = Application::GetLayer<RenderLayer>();
    renderLayer->ApplyEnvironmentalSettings(m_visualizationCamera);
    GlobalTransform sceneCameraGT;
    sceneCameraGT.SetValue(editorLayer->m_sceneCameraPosition,
                           editorLayer->m_sceneCameraRotation, glm::vec3(1.0f));
    renderLayer->RenderToCamera(m_visualizationCamera, sceneCameraGT);

#pragma region Internode debug camera
    Camera::m_cameraInfoBlock.UploadMatrices(
            editorLayer->m_sceneCamera,
            editorLayer->m_sceneCameraPosition,
            editorLayer->m_sceneCameraRotation);
#pragma endregion

#pragma region Rendering
    if (m_drawInternodes) {
        if (m_autoUpdate) {
            UpdateInternodeColors();
            UpdateInternodeCylinder();
        }
        if (m_visualizationCamera->IsEnabled())
            RenderInternodeCylinders();
    }
    if (m_drawBranches) {
        if (m_autoUpdate) {
            UpdateBranchColors();
            UpdateBranchCylinder();
        }
        if (m_visualizationCamera->IsEnabled())
            RenderBranchCylinders();
    }
    if (m_drawPointers) {
        if (m_autoUpdate) {
            UpdateInternodePointer(m_pointerLength, m_pointerWidth);
        }
        if (m_visualizationCamera->IsEnabled())
            RenderInternodePointers();
    }
#pragma endregion

}

static const char *BranchColorModes[]{"None", "Order", "Level", "Water", "ApicalControl",
                                      "WaterPressure",
                                      "Proximity", "Inhibitor", "IndexDivider", "IndexRange", "StrahlerNumber",
                                      "ChildCount", "SubTree", "Branchlet"};

void InternodeLayer::DrawColorModeSelectionMenu() {
    if (ImGui::TreeNodeEx("Branch Coloring", ImGuiTreeNodeFlags_DefaultOpen)) {
        static int colorModeIndex = 0;
        if (ImGui::Combo("Color mode", &colorModeIndex, BranchColorModes,
                         IM_ARRAYSIZE(BranchColorModes))) {
            m_branchColorMode = (BranchColorMode) colorModeIndex;
        }
        //ImGui::DragFloat("Multiplier", &m_internodeColorValueMultiplier, 0.01f);
        //ImGui::DragFloat("Compress", &m_internodeColorValueCompressFactor, 0.01f);
        ImGui::Checkbox("Hide other branches", &m_hideUnnecessaryInternodes);
        ImGui::Checkbox("Override thickness", &m_overrideThickness);
        if (m_overrideThickness) {
            ImGui::DragFloat("Thickness", &m_internodeThickness, 0.01f);
        }
        switch (m_branchColorMode) {
            case BranchColorMode::IndexDivider:
                ImGui::DragInt("Divider", &m_indexDivider, 1, 2, 1024);
                break;
            case BranchColorMode::IndexRange:
                ImGui::DragInt("Range Min", &m_indexRangeMin, 1, 0, m_indexRangeMax);
                ImGui::DragInt("Range Max", &m_indexRangeMax, 1, m_indexRangeMin, 999999);
                break;
            case BranchColorMode::ChildCount:
                ImGui::ColorEdit4("Color for 0", &m_childCountColors[0].x);
                ImGui::ColorEdit4("Color for 1", &m_childCountColors[1].x);
                ImGui::ColorEdit4("Color for 2", &m_childCountColors[2].x);
                ImGui::ColorEdit4("Color for 3", &m_childCountColors[3].x);
                break;
        }
        ImGui::TreePop();
    }
}

void InternodeLayer::CalculateStatistics(const std::shared_ptr<Scene> &scene) {
    for (auto &behaviour: m_plantBehaviours) {
        if (behaviour) {
            std::vector<Entity> currentRoots;
            scene->GetEntityArray(behaviour->m_rootsQuery, currentRoots);
            for (auto root: currentRoots) {
                scene->ForEachChild(root, [&](Entity child) {
                    if (!behaviour->InternodeCheck(scene, child)) return;
                    behaviour->InternodeGraphWalkerEndToRoot(scene, child,
                                                             [&](Entity parent) {
                                                                 auto parentStat = scene->GetDataComponent<InternodeStatistics>(
                                                                         parent);
                                                                 std::vector<int> indices;
                                                                 parentStat.m_childCount = 0;
                                                                 scene->ForEachChild(parent,
                                                                                     [&](Entity child) {
                                                                                         if (behaviour->InternodeCheck(
                                                                                                 scene,
                                                                                                 child)) {
                                                                                             indices.push_back(
                                                                                                     scene->GetDataComponent<InternodeStatistics>(
                                                                                                             child).m_strahlerOrder);
                                                                                             parentStat.m_childCount++;
                                                                                         }
                                                                                     });
                                                                 if (indices.empty()) { parentStat.m_strahlerOrder = 1; }
                                                                 else if (indices.size() == 1) {
                                                                     parentStat.m_strahlerOrder = indices[0];
                                                                 } else {
                                                                     bool different = false;
                                                                     int maxIndex = indices[0];
                                                                     for (int i = 1; i < indices.size(); i++) {
                                                                         if (indices[i] != maxIndex) {
                                                                             different = true;
                                                                             maxIndex = glm::max(maxIndex, indices[i]);
                                                                         }
                                                                     }
                                                                     if (different) {
                                                                         parentStat.m_strahlerOrder = maxIndex;
                                                                     } else {
                                                                         parentStat.m_strahlerOrder = maxIndex + 1;
                                                                     }
                                                                 }

                                                                 scene->SetDataComponent(parent, parentStat);
                                                             },
                                                             [&](Entity endNode) {
                                                                 auto endNodeStat = scene->GetDataComponent<InternodeStatistics>(
                                                                         endNode);
                                                                 endNodeStat.m_strahlerOrder = 1;
                                                                 endNodeStat.m_childCount = 0;
                                                                 scene->SetDataComponent(endNode, endNodeStat);
                                                             }
                    );
                });

            }
        }
    }

}

void InternodeLayer::FixedUpdate() {
    if (Application::IsPlaying()) {
        if (m_applyFBMField) {
            std::vector<Entity> branches;
            std::vector<glm::vec3> forces;
            branches.clear();
            auto scene = GetScene();
            scene->GetEntityArray(m_branchesQuery, branches, false);
            forces.resize(branches.size());
            scene->ForEach<GlobalTransform>(Jobs::Workers(), m_branchesQuery,
                                            [&](int i, Entity entity, GlobalTransform &globalTransform) {
                                                const auto position = scene->GetDataComponent<GlobalTransform>(
                                                        entity).GetPosition();
                                                forces[i] = m_fBMField.GetT(position,
                                                                            Application::Time().CurrentTime(),
                                                                            10.0f, 0.02f, 6) *
                                                            m_forceFactor;
                                            }, false);

            for (int i = 0; i < branches.size(); i++) {
                auto &branch = branches[i];
                if (scene->IsEntityEnabled(branch) && scene->HasPrivateComponent<RigidBody>(branch)) {
                    auto rigidBody = scene->GetOrSetPrivateComponent<RigidBody>(branch).lock();
                    if (rigidBody->Registered()) rigidBody->AddForce(forces[i]);
                }
            }
        }
    }
}

void InternodeLayer::UpdateBranchColors() {
    auto scene = GetScene();
    scene->ForEach<InternodeBranchColor>(Jobs::Workers(),
                                         m_branchesQuery,
                                         [=](int i, Entity entity, InternodeBranchColor &branchRenderColor) {
                                    branchRenderColor.m_value = glm::vec4(m_branchColor, m_branchTransparency);
                                },
                                         true);
}

void InternodeLayer::RenderBranchCylinders() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    std::vector<InternodeBranchCylinder> branchCylinders;
    auto scene = GetScene();
    scene->GetComponentDataArray<InternodeBranchCylinder>(m_branchesQuery,
                                                          branchCylinders);
    std::vector<InternodeBranchColor> branchColors;
    scene->GetComponentDataArray<InternodeBranchColor>(m_branchesQuery,
                                                       branchColors);
    if (!branchCylinders.empty())
        Gizmos::DrawGizmoMeshInstancedColored(
                DefaultResources::Primitives::Cylinder, m_visualizationCamera,
                editorLayer->m_sceneCameraPosition,
                editorLayer->m_sceneCameraRotation,
                *reinterpret_cast<std::vector<glm::vec4> *>(&branchColors),
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchCylinders),
                glm::mat4(1.0f), 1.0f);
}

void InternodeLayer::ObstacleRemoval() {
    auto scene = Application::GetActiveScene();

    std::vector<std::pair<GlobalTransform, std::shared_ptr<IVolume>>> obstacleVolumes;
    std::vector<std::pair<GlobalTransform, std::shared_ptr<IVolume>>> volumes;
    auto *cubeObstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<CubeVolume>();
    if (cubeObstaclesEntities)
        for (const auto &i: *cubeObstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<CubeVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<CubeVolume>(i).lock());
                if(volume->IsEnabled()) {
                    if (volume->m_asObstacle)
                        obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                    else
                        volumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                }
            }
        }
    auto *sphereObstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<SphereVolume>();
    if (sphereObstaclesEntities)
        for (const auto &i: *sphereObstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<SphereVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<SphereVolume>(i).lock());
                if(volume->IsEnabled()) {
                    if (volume->m_asObstacle)
                        obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                    else
                        volumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                }
            }
        }
    auto *cylinderObstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<CylinderVolume>();
    if (cylinderObstaclesEntities)
        for (const auto &i: *cylinderObstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<CylinderVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<CylinderVolume>(i).lock());
                if(volume->IsEnabled()) {
                    if (volume->m_asObstacle)
                        obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                    else
                        volumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                }
            }
        }
    auto *meshObstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<MeshVolume>();
    if (meshObstaclesEntities)
        for (const auto &i: *meshObstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<MeshVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<MeshVolume>(i).lock());
                if(volume->IsEnabled()) {
                    if (volume->m_asObstacle)
                        obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                    else
                        volumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                }
            }
        }
    auto* rbvObstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<RadialBoundingVolume>();
    if (rbvObstaclesEntities)
        for (const auto& i : *rbvObstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<RadialBoundingVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<RadialBoundingVolume>(i).lock());
                if (volume->IsEnabled()) {
                    if (volume->m_asObstacle)
                        obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                    else
                        volumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
                }
            }
        }
    if(volumes.empty() && obstacleVolumes.empty()) return;
    std::vector<Entity> internodes;
    std::vector<glm::vec3> positions;
    std::vector<bool> obstacleResults;
    std::vector<bool> results;
    internodes.resize(scene->GetEntityAmount(m_internodesQuery));
    positions.resize(internodes.size());
    obstacleResults.resize(internodes.size());
    results.resize(internodes.size());
    scene->ForEach<GlobalTransform>(Jobs::Workers(), m_internodesQuery, [&](int i, Entity entity, GlobalTransform& globalTransform){
        internodes[i] = entity;
        positions[i] = globalTransform.GetPosition();
        obstacleResults[i] = false;
        results[i] = true;
    });
    for(auto& i : obstacleVolumes){
        i.second->InVolume(i.first, positions, obstacleResults);
    }
    for(auto& i : volumes){
        i.second->InVolume(i.first, positions, results);
    }

    for(int i = 0; i < internodes.size(); i++){
        if((obstacleResults[i] || !results[i]) && scene->IsEntityValid(internodes[i])){
            scene->DeleteEntity(internodes[i]);
        }
    }
    /*for (auto &behaviour: m_plantBehaviours) {
        if (behaviour) {
            std::vector<Entity> currentRoots;
            scene->GetEntityArray(behaviour->m_rootsQuery, currentRoots);
            for (auto root: currentRoots) {
                scene->ForEachChild(root, [&](Entity child) {
                    if (!behaviour->InternodeCheck(scene, child)) return;
                    behaviour->InternodeGraphWalkerRootToEnd(scene, child,
                                                             [&](Entity parent, Entity child) {
                                                                 //Remove if obstacle.
                                                                 if (!obstacleVolumes.empty()) {
                                                                     auto position = scene->GetDataComponent<GlobalTransform>(
                                                                             child).GetPosition();
                                                                     for (const auto &i: obstacleVolumes) {
                                                                         if (i.second->InVolume(i.first, position)) {
                                                                             scene->DeleteEntity(child);
                                                                             return;
                                                                         }
                                                                     }
                                                                 }
                                                             }
                    );
                });

            }
        }
    }
    */

}

void InternodeLayer::Preprocess(const std::shared_ptr<Scene> &scene) {
#pragma region PreProcess
#pragma region InternodeStatus
    for (auto &behaviour: m_plantBehaviours) {
        if (behaviour) {
            std::vector<Entity> currentRoots;
            scene->GetEntityArray(behaviour->m_rootsQuery, currentRoots);
            for (auto rootEntity: currentRoots) {
                if (!behaviour->RootCheck(scene, rootEntity)) return;
                auto root = scene->GetOrSetPrivateComponent<InternodePlant>(rootEntity).lock();
                auto center = glm::vec3(0);
                int amount = 1;
                auto internodeGlobalTransform = scene->GetDataComponent<GlobalTransform>(
                        rootEntity);
                center += internodeGlobalTransform.GetPosition();
                scene->ForEachChild(rootEntity, [&](Entity child) {
                    if (!behaviour->InternodeCheck(scene, child)) return;
                    behaviour->InternodeGraphWalker(scene, child,
                                                    [&](Entity parent, Entity child) {
                                                        auto parentInternodeInfo = scene->GetDataComponent<InternodeInfo>(
                                                                parent);
                                                        auto childInternodeInfo = scene->GetDataComponent<InternodeInfo>(
                                                                child);
                                                        auto childGlobalTransform = scene->GetDataComponent<GlobalTransform>(
                                                                child);
                                                        center += childGlobalTransform.GetPosition();
                                                        amount++;
                                                        auto childInternode = scene->GetOrSetPrivateComponent<Internode>(
                                                                child).lock();
                                                        childInternodeInfo.m_rootDistance =
                                                                parentInternodeInfo.m_length +
                                                                parentInternodeInfo.m_rootDistance;
                                                        childInternodeInfo.m_biomass =
                                                                childInternodeInfo.m_length *
                                                                childInternodeInfo.m_thickness;
                                                        if (!childInternode->m_fromApicalBud) {
                                                            childInternodeInfo.m_order =
                                                                    parentInternodeInfo.m_order + 1;
                                                        } else {
                                                            childInternodeInfo.m_order = parentInternodeInfo.m_order;
                                                        }
                                                        scene->SetDataComponent(child, childInternodeInfo);
                                                    },
                                                    [&](Entity parent) {
                                                        auto parentInternodeInfo = scene->GetDataComponent<InternodeInfo>(
                                                                parent);

                                                        parentInternodeInfo.m_endNode = false;
                                                        parentInternodeInfo.m_totalDistanceToAllBranchEnds = parentInternodeInfo.m_childTotalBiomass = 0;
                                                        float maxDistanceToAnyBranchEnd = -1.0f;
                                                        float maxTotalDistanceToAllBranchEnds = -1.0f;
                                                        float maxChildTotalBiomass = -1.0f;
                                                        Entity largestChild;
                                                        Entity longestChild;
                                                        Entity heaviestChild;
                                                        scene->ForEachChild(parent,
                                                                            [&](Entity child) {
                                                                                if (!behaviour->InternodeCheck(scene,
                                                                                                               child))
                                                                                    return;
                                                                                auto childInternodeInfo = scene->GetDataComponent<InternodeInfo>(
                                                                                        child);
                                                                                float childTotalDistanceToAllBranchEnds =
                                                                                        childInternodeInfo.m_totalDistanceToAllBranchEnds +
                                                                                        childInternodeInfo.m_length;
                                                                                float childTotalBiomass =
                                                                                        childInternodeInfo.m_childTotalBiomass +
                                                                                        childInternodeInfo.m_biomass;
                                                                                float childMaxDistanceToAnyBranchEnd =
                                                                                        childInternodeInfo.m_maxDistanceToAnyBranchEnd +
                                                                                        childInternodeInfo.m_length;
                                                                                parentInternodeInfo.m_totalDistanceToAllBranchEnds += childTotalDistanceToAllBranchEnds;
                                                                                parentInternodeInfo.m_childTotalBiomass += childTotalBiomass;
                                                                                if (maxTotalDistanceToAllBranchEnds <
                                                                                    childTotalDistanceToAllBranchEnds) {
                                                                                    maxTotalDistanceToAllBranchEnds = childTotalDistanceToAllBranchEnds;
                                                                                    largestChild = child;
                                                                                }
                                                                                if (maxDistanceToAnyBranchEnd <
                                                                                    childMaxDistanceToAnyBranchEnd) {
                                                                                    maxDistanceToAnyBranchEnd = childMaxDistanceToAnyBranchEnd;
                                                                                    longestChild = child;
                                                                                }
                                                                                if (maxChildTotalBiomass <
                                                                                    childTotalBiomass) {
                                                                                    maxChildTotalBiomass = childTotalBiomass;
                                                                                    heaviestChild = child;
                                                                                }
                                                                            });
                                                        scene->ForEachChild(parent,
                                                                            [&](Entity child) {
                                                                                if (!behaviour->InternodeCheck(scene,
                                                                                                               child))
                                                                                    return;
                                                                                auto childInternodeInfo = scene->GetDataComponent<InternodeInfo>(
                                                                                        child);
                                                                                childInternodeInfo.m_largestChild =
                                                                                        largestChild == child;
                                                                                childInternodeInfo.m_longestChild =
                                                                                        longestChild == child;
                                                                                childInternodeInfo.m_heaviestChild =
                                                                                        heaviestChild ==
                                                                                        child;
                                                                                scene->SetDataComponent(child,
                                                                                                        childInternodeInfo);
                                                                            });
                                                        parentInternodeInfo.m_maxDistanceToAnyBranchEnd = maxDistanceToAnyBranchEnd;
                                                        scene->SetDataComponent(parent, parentInternodeInfo);
                                                    },
                                                    [&](Entity endNode) {
                                                        auto endNodeInternodeInfo = scene->GetDataComponent<InternodeInfo>(
                                                                endNode);
                                                        endNodeInternodeInfo.m_endNode = true;
                                                        endNodeInternodeInfo.m_maxDistanceToAnyBranchEnd = endNodeInternodeInfo.m_totalDistanceToAllBranchEnds = endNodeInternodeInfo.m_childTotalBiomass = 0;
                                                        endNodeInternodeInfo.m_largestChild = endNodeInternodeInfo.m_longestChild = endNodeInternodeInfo.m_heaviestChild = true;
                                                        scene->SetDataComponent(endNode,
                                                                                endNodeInternodeInfo);
                                                    });

                });
                root->m_center = center / static_cast<float>(amount);
            };
        }
    }
}

void InternodeLayer::ColorSubTree(const std::shared_ptr<Scene> &scene, const Entity &entity, int colorIndex) {
    if (scene->HasDataComponent<InternodeInfo>(entity)) {
        InternodeColor internodeColor;
        internodeColor.m_value = glm::vec4(m_randomColors[colorIndex], 1.0f);
        scene->SetDataComponent(entity, internodeColor);
        auto children = scene->GetChildren(entity);
        for (const auto &i: children) {
            ColorSubTree(scene, i, colorIndex + 1);
        }
    }
}

void InternodeLayer::ColorBranchlet(const std::shared_ptr<Scene> &scene) {
    for (auto &behaviour: m_plantBehaviours) {
        if (behaviour) {
            std::vector<Entity> currentRoots;
            scene->GetEntityArray(behaviour->m_rootsQuery, currentRoots);
            for (auto rootEntity: currentRoots) {
                if (!behaviour->RootCheck(scene, rootEntity)) return;
                scene->ForEachChild(rootEntity, [&](Entity child) {
                    if (!behaviour->InternodeCheck(scene, child)) return;
                    int colorIndex = 0;
                    behaviour->InternodeGraphWalkerRootToEnd(scene, child,
                                                    [&](Entity parent, Entity child) {
                                                        auto children = scene->GetChildren(parent);
                                                        InternodeColor internodeColor;
                                                        internodeColor.m_value = {m_randomColors[colorIndex], 1.0};
                                                        for(const auto& i : children){
                                                            scene->SetDataComponent(i, internodeColor);
                                                        }
                                                        colorIndex++;
                                                    });

                });
            };
        }
    }
}


#pragma endregion

void BranchPhysicsParameters::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_density" << YAML::Value << m_density;
    out << YAML::Key << "m_linearDamping" << YAML::Value << m_linearDamping;
    out << YAML::Key << "m_angularDamping" << YAML::Value << m_angularDamping;
    out << YAML::Key << "m_positionSolverIteration" << YAML::Value << m_positionSolverIteration;
    out << YAML::Key << "m_velocitySolverIteration" << YAML::Value << m_velocitySolverIteration;
    out << YAML::Key << "m_jointDriveStiffnessFactor" << YAML::Value << m_jointDriveStiffnessFactor;
    out << YAML::Key << "m_jointDriveStiffnessThicknessFactor" << YAML::Value << m_jointDriveStiffnessThicknessFactor;
    out << YAML::Key << "m_jointDriveDampingFactor" << YAML::Value << m_jointDriveDampingFactor;
    out << YAML::Key << "m_jointDriveDampingThicknessFactor" << YAML::Value << m_jointDriveDampingThicknessFactor;
    out << YAML::Key << "m_enableAccelerationForDrive" << YAML::Value << m_enableAccelerationForDrive;
    out << YAML::Key << "m_minimumThickness" << YAML::Value << m_minimumThickness;
}

void BranchPhysicsParameters::Deserialize(const YAML::Node &in) {
    if (in["m_density"]) m_density = in["m_density"].as<float>();
    if (in["m_linearDamping"]) m_linearDamping = in["m_linearDamping"].as<float>();
    if (in["m_angularDamping"]) m_angularDamping = in["m_angularDamping"].as<float>();
    if (in["m_positionSolverIteration"]) m_positionSolverIteration = in["m_positionSolverIteration"].as<int>();
    if (in["m_velocitySolverIteration"]) m_velocitySolverIteration = in["m_velocitySolverIteration"].as<int>();
    if (in["m_jointDriveStiffnessFactor"]) m_jointDriveStiffnessFactor = in["m_jointDriveStiffnessFactor"].as<float>();
    if (in["m_jointDriveStiffnessThicknessFactor"]) m_jointDriveStiffnessThicknessFactor = in["m_jointDriveStiffnessThicknessFactor"].as<float>();
    if (in["m_jointDriveDampingFactor"]) m_jointDriveDampingFactor = in["m_jointDriveDampingFactor"].as<float>();
    if (in["m_jointDriveDampingThicknessFactor"]) m_jointDriveDampingThicknessFactor = in["m_jointDriveDampingThicknessFactor"].as<float>();
    if (in["m_enableAccelerationForDrive"]) m_enableAccelerationForDrive = in["m_enableAccelerationForDrive"].as<bool>();
    if (in["m_minimumThickness"]) m_minimumThickness = in["m_minimumThickness"].as<float>();
}

void BranchPhysicsParameters::OnInspect() {
    if (ImGui::TreeNodeEx("Physics Parameters")) {
        ImGui::DragFloat("Internode Density", &m_density, 0.1f, 0.01f, 1000.0f);
        ImGui::DragFloat2("RigidBody Damping", &m_linearDamping, 0.1f, 0.01f,
                          1000.0f);
        ImGui::DragFloat2("Drive Stiffness", &m_jointDriveStiffnessFactor, 0.1f,
                          0.01f, 1000000.0f);
        ImGui::DragFloat2("Drive Damping", &m_jointDriveDampingFactor, 0.1f, 0.01f,
                          1000000.0f);
        ImGui::Checkbox("Use acceleration", &m_enableAccelerationForDrive);

        int pi = m_positionSolverIteration;
        int vi = m_velocitySolverIteration;
        if (ImGui::DragInt("Velocity solver iteration", &vi, 1, 1, 100)) {
            m_velocitySolverIteration = vi;
        }
        if (ImGui::DragInt("Position solver iteration", &pi, 1, 1, 100)) {
            m_positionSolverIteration = pi;
        }

        ImGui::DragFloat("Minimum thickness", &m_minimumThickness, 0.001f, 0.001f, 100.0f);
        ImGui::TreePop();
    }
}