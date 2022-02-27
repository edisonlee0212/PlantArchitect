
//
// Created by lllll on 8/27/2021.
//
#include "PlantLayer.hpp"
#include <Internode.hpp>
#include "EditorLayer.hpp"
#include <PlantDataComponents.hpp>
#include "GeneralTreeBehaviour.hpp"
#include "SpaceColonizationBehaviour.hpp"
#include "LSystemBehaviour.hpp"
#include "CubeVolume.hpp"
#include "RadialBoundingVolume.hpp"
#include "Joint.hpp"
#include "PhysicsLayer.hpp"
#include "DefaultInternodeResource.hpp"
#include "EmptyInternodeResource.hpp"
#include "DefaultInternodePhyllotaxis.hpp"
#include "FBM.hpp"

using namespace PlantArchitect;

void PlantLayer::PreparePhysics(const Entity &entity, const Entity &child,
                                const BranchPhysicsParameters &branchPhysicsParameters) {
    auto childBranchInfo = child.GetDataComponent<BranchInfo>();
    auto rigidBody = child.GetOrSetPrivateComponent<RigidBody>().lock();
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

    auto joint = child.GetOrSetPrivateComponent<Joint>().lock();
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

void PlantLayer::Simulate(int iterations) {
    for (int iteration = 0; iteration < iterations; iteration++) {
        m_voxelSpace.Clear();
        Entities::ForEach<InternodeInfo, GlobalTransform>
                (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
                 [&](int i, Entity entity, InternodeInfo &internodeInfo, GlobalTransform &internodeGlobalTransform) {
                     const auto position = internodeGlobalTransform.GetPosition();
                     m_voxelSpace.Push(position, entity);
                 }, true);
        Entities::ForEach<InternodeInfo, GlobalTransform>
                (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
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
        for (auto &i: m_internodeBehaviours) {
            auto behaviour = i.Get<IPlantBehaviour>();
            if (behaviour) behaviour->Grow(iteration);
        }
    }
    if (m_enablePhysics) PreparePhysics();
    CalculateStatistics();
    UpdateInternodeColors();
    UpdateInternodeCylinder();
    UpdateInternodePointer(m_pointerLength, m_pointerWidth);
}

void PlantLayer::PreparePhysics() {
    auto physicsLayer = Application::GetLayer<PhysicsLayer>();
    if (!physicsLayer) return;
    for (auto &i: m_internodeBehaviours) {
        auto behaviour = i.Get<IPlantBehaviour>();
        if (behaviour) {
            std::vector<Entity> roots;
            behaviour->m_rootsQuery.ToEntityArray(Entities::GetCurrentScene(), roots);
            for (const auto &root : roots) {
                auto children = root.GetChildren();
                for(const auto& child : children) {
                    if(child.HasPrivateComponent<Branch>()) {
                        auto branchPhysicsParameters = child.GetOrSetPrivateComponent<Branch>().lock()->m_branchPhysicsParameters;
                        auto rootRigidBody = root.GetOrSetPrivateComponent<RigidBody>().lock();
                        if(!rootRigidBody->IsKinematic()) rootRigidBody->SetKinematic(true);
                        auto childBranchInfo = child.GetDataComponent<BranchInfo>();
                        auto rigidBody = child.GetOrSetPrivateComponent<RigidBody>().lock();
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

                        auto joint = child.GetOrSetPrivateComponent<Joint>().lock();
                        joint->Link(root);
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

                        behaviour->BranchGraphWalkerRootToEnd(child, [&](Entity parent, Entity child) {
                            PreparePhysics(parent, child, branchPhysicsParameters);
                        });
                        break;
                    }
                }
            }
        }
    }
    auto activeScene = Entities::GetCurrentScene();
    physicsLayer->UploadRigidBodyShapes(activeScene);
    physicsLayer->UploadTransforms(activeScene, true);
    physicsLayer->UploadJointLinks(activeScene);
}

#pragma region Methods

void PlantLayer::OnInspect() {
    if (ImGui::Begin("Internode Manager")) {
        static int iterations = 1;
        ImGui::DragInt("Iterations", &iterations);
        if (ImGui::Button("Simulate")) {
            Simulate(iterations);
        }
        ImGui::Checkbox("Enable physics", &m_enablePhysics);
        if (m_enablePhysics) {
            if (ImGui::TreeNode("Physics")) {
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
        }
        if (ImGui::TreeNodeEx("Voxels", ImGuiTreeNodeFlags_DefaultOpen)) {
            m_voxelSpace.OnInspect();
            ImGui::TreePop();
        }
        if (m_voxelSpace.m_display) {
            auto editorLayer = Application::GetLayer<EditorLayer>();
            if (editorLayer) {
                Graphics::DrawGizmoMeshInstanced(
                        DefaultResources::Primitives::Cube, m_visualizationCamera,
                        editorLayer->m_sceneCameraPosition,
                        editorLayer->m_sceneCameraRotation,
                        glm::vec4(1, 1, 1, 0.2),
                        m_voxelSpace.m_frozenVoxels);
            }
            Graphics::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube,
                                             glm::vec4(1, 1, 1, 0.2),
                                             m_voxelSpace.m_frozenVoxels, glm::mat4(1.0f), 1.0f);
        }
        if (ImGui::TreeNodeEx("Internode Behaviour")) {
            ImGui::Text("Add Internode Behaviour");
            ImGui::SameLine();
            static AssetRef temp;
            Editor::DragAndDropButton(temp, "Here",
                                      {"GeneralTreeBehaviour", "SpaceColonizationBehaviour", "LSystemBehaviour"},
                                      false);
            if (temp.Get<IPlantBehaviour>()) {
                PushInternodeBehaviour(temp.Get<IPlantBehaviour>());
                temp.Clear();
            }
            if (ImGui::TreeNodeEx("Internode Behaviours", ImGuiTreeNodeFlags_DefaultOpen)) {
                int index = 0;
                bool skip = false;
                for (auto &i: m_internodeBehaviours) {
                    auto ptr = i.Get<IPlantBehaviour>();
                    ImGui::Button(("Slot " + std::to_string(index) + ": " + ptr->m_name).c_str());
                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                        Editor::GetInstance().m_inspectingAsset = ptr;
                    }
                    const std::string tag = "##" + ptr->GetTypeName() + std::to_string(ptr->GetHandle());
                    if (ImGui::BeginPopupContextItem(tag.c_str())) {
                        if (ImGui::BeginMenu(("Rename" + tag).c_str())) {
                            static char newName[256];
                            ImGui::InputText(("New name" + tag).c_str(), newName, 256);
                            if (ImGui::Button(("Confirm" + tag).c_str()))
                                ptr->m_name = std::string(newName);
                            ImGui::EndMenu();
                        }
                        if (ImGui::Button(("Remove" + tag).c_str())) {
                            i.Clear();
                            skip = true;
                        }
                        ImGui::EndPopup();
                    }
                    if (skip) {
                        break;
                    }
                    index++;
                }
                if (skip) {
                    int index2 = 0;
                    for (auto &i: m_internodeBehaviours) {
                        if (!i.Get<IPlantBehaviour>()) {
                            m_internodeBehaviours.erase(m_internodeBehaviours.begin() + index2);
                            break;
                        }
                        index2++;
                    }
                }
                ImGui::TreePop();
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
        if (ImGui::BeginChild("CameraRenderer", ImVec2(0, 0), false,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Settings")) {
#pragma region Menu
                    ImGui::Checkbox("Auto update", &m_autoUpdate);
                    ImGui::Checkbox("Internodes", &m_drawInternodes);
                    if (m_drawInternodes) {

                        if (ImGui::TreeNodeEx("Internode settings",
                                              ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::SliderFloat("Transparency", &m_internodeTransparency, 0, 1);
                            DrawColorModeSelectionMenu();
                            ImGui::TreePop();
                        }
                    }
                    ImGui::Checkbox("Branches", &m_drawBranches);
                    if (m_drawBranches) {
                        if (ImGui::TreeNodeEx("Branch settings",
                                              ImGuiTreeNodeFlags_DefaultOpen)) {
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
                ImGui::EndMenuBar();
            }
            viewPortSize = ImGui::GetWindowSize();
            viewPortSize.y -= 20;
            if (viewPortSize.y < 0)
                viewPortSize.y = 0;
            m_visualizationCameraResolutionX = viewPortSize.x;
            m_visualizationCameraResolutionY = viewPortSize.y;
            ImGui::Image(
                    reinterpret_cast<ImTextureID>(
                            m_visualizationCamera->GetTexture()->UnsafeGetGLTexture()->Id()),
                    viewPortSize, ImVec2(0, 1), ImVec2(1, 0));
            glm::vec2 mousePosition = glm::vec2(FLT_MAX, FLT_MIN);
            if (ImGui::IsWindowFocused()) {
                bool valid = true;
                mousePosition = Inputs::GetMouseAbsolutePositionInternal(
                        Windows::GetWindow());
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
                        auto windowPos = ImGui::GetWindowPos();
                        auto windowSize = ImGui::GetWindowSize();
                        mousePosition.x -= windowPos.x;
                        mousePosition.x -= windowSize.x;
                        mousePosition.y -= windowPos.y;
                        float minDistance = FLT_MAX;
                        GlobalTransform cameraLtw;
                        cameraLtw.m_value =
                                glm::translate(
                                        editorLayer->m_sceneCameraPosition) *
                                glm::mat4_cast(
                                        editorLayer->m_sceneCameraRotation);
                        const Ray cameraRay = m_visualizationCamera->ScreenPointToRay(
                                cameraLtw, mousePosition);
                        Entities::ForEach<GlobalTransform, InternodeInfo>(
                                Entities::GetCurrentScene(), Jobs::Workers(),
                                m_internodesQuery,
                                [&, cameraLtw, cameraRay](int i, Entity entity,
                                                          GlobalTransform &ltw,
                                                          InternodeInfo &internodeInfo) {
                                    const glm::vec3 position = ltw.m_value[3];
                                    const glm::vec3 position2 = position + internodeInfo.m_length * glm::normalize(
                                            ltw.GetRotation() * glm::vec3(0, 0, -1));
                                    const auto center = (position + position2) / 2.0f;
                                    auto dir = cameraRay.m_direction;
                                    auto pos = cameraRay.m_start;
                                    const auto radius = internodeInfo.m_thickness;
                                    const auto height = glm::distance(position2, position);
                                    if (!cameraRay.Intersect(center, height / 2.0f))
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
                                    const auto dotP = a * c - b * b; // always >= 0
                                    float sc, tc;
                                    // compute the line parameters of the two closest points
                                    if (dotP < 0.001f) { // the lines are almost parallel
                                        sc = 0.0f;
                                        tc = (b > c ? d / b
                                                    : e / c); // use the largest denominator
                                    } else {
                                        sc = (b * e - c * d) / dotP;
                                        tc = (a * e - b * d) / dotP;
                                    }
                                    // get the difference of the two closest points
                                    glm::vec3 dP = w + sc * u - tc * v; // =  L1(sc) - L2(tc)
                                    if (glm::length(dP) > radius)
                                        return;
#pragma endregion

                                    const auto distance = glm::distance(
                                            glm::vec3(cameraLtw.m_value[3]), glm::vec3(center));
                                    std::lock_guard<std::mutex> lock(writeMutex);
                                    if (distance < minDistance) {
                                        minDistance = distance;
                                        m_currentFocusingInternode = entity;
                                    }
                                });
                        if (Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_LEFT,
                                                     Windows::GetWindow())) {
                            if (!m_currentFocusingInternode.Get().IsNull()) {
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

void PlantLayer::OnCreate() {
    ClassRegistry::RegisterDataComponent<InternodeCylinder>("InternodeCylinder");
    ClassRegistry::RegisterDataComponent<InternodeCylinderWidth>("InternodeCylinderWidth");
    ClassRegistry::RegisterDataComponent<InternodePointer>("InternodePointer");
    ClassRegistry::RegisterDataComponent<InternodeColor>("InternodeColor");
    ClassRegistry::RegisterDataComponent<BranchCylinder>("BranchCylinder");
    ClassRegistry::RegisterDataComponent<BranchCylinderWidth>("BranchCylinderWidth");
    ClassRegistry::RegisterDataComponent<BranchColor>("BranchColor");

    ClassRegistry::RegisterPrivateComponent<IVolume>("IVolume");
    ClassRegistry::RegisterPrivateComponent<CubeVolume>("CubeVolume");
    ClassRegistry::RegisterPrivateComponent<RadialBoundingVolume>("RadialBoundingVolume");


    ClassRegistry::RegisterDataComponent<GeneralTreeTag>("GeneralTreeTag");
    ClassRegistry::RegisterDataComponent<GeneralTreeParameters>("GeneralTreeParameters");
    ClassRegistry::RegisterDataComponent<InternodeStatus>("InternodeStatus");
    ClassRegistry::RegisterDataComponent<InternodeWaterPressure>("InternodeWaterPressure");
    ClassRegistry::RegisterDataComponent<InternodeWater>("InternodeWater");
    ClassRegistry::RegisterDataComponent<InternodeIllumination>("InternodeIllumination");
    ClassRegistry::RegisterPrivateComponent<InternodeWaterFeeder>("InternodeWaterFeeder");
    ClassRegistry::RegisterAsset<GeneralTreeBehaviour>("GeneralTreeBehaviour", ".gtbehaviour");

    ClassRegistry::RegisterDataComponent<SpaceColonizationTag>("SpaceColonizationTag");
    ClassRegistry::RegisterDataComponent<SpaceColonizationParameters>("SpaceColonizationParameters");
    ClassRegistry::RegisterDataComponent<SpaceColonizationIncentive>("SpaceColonizationIncentive");
    ClassRegistry::RegisterAsset<SpaceColonizationBehaviour>("SpaceColonizationBehaviour", ".scbehaviour");

    ClassRegistry::RegisterAsset<LString>("LString", ".lstring");
    ClassRegistry::RegisterDataComponent<LSystemTag>("LSystemTag");
    ClassRegistry::RegisterDataComponent<LSystemParameters>("LSystemParameters");
    ClassRegistry::RegisterAsset<LSystemBehaviour>("LSystemBehaviour", ".lsbehaviour");

    ClassRegistry::RegisterSerializable<EmptyInternodeResource>("EmptyInternodeResource");
    ClassRegistry::RegisterSerializable<DefaultInternodeResource>("DefaultInternodeResource");
    ClassRegistry::RegisterSerializable<Bud>("LateralBud");
    ClassRegistry::RegisterPrivateComponent<Internode>("Internode");
    ClassRegistry::RegisterPrivateComponent<Branch>("Branch");
    ClassRegistry::RegisterPrivateComponent<Root>("Root");

    ClassRegistry::RegisterDataComponent<InternodeInfo>("InternodeInfo");
    ClassRegistry::RegisterDataComponent<RootInfo>("RootInfo");
    ClassRegistry::RegisterDataComponent<BranchInfo>("BranchInfo");
    ClassRegistry::RegisterDataComponent<InternodeStatistics>("InternodeStatistics");

    ClassRegistry::RegisterAsset<InternodeFoliage>("InternodeFoliage", ".internodefoliage");
    ClassRegistry::RegisterAsset<DefaultInternodePhyllotaxis>("DefaultInternodePhyllotaxis", ".defaultip");

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

    auto spaceColonizationBehaviour = AssetManager::CreateAsset<SpaceColonizationBehaviour>();
    auto lSystemBehaviour = AssetManager::CreateAsset<LSystemBehaviour>();
    auto generalTreeBehaviour = AssetManager::CreateAsset<GeneralTreeBehaviour>();
    PushInternodeBehaviour(
            std::dynamic_pointer_cast<IPlantBehaviour>(spaceColonizationBehaviour));
    PushInternodeBehaviour(std::dynamic_pointer_cast<IPlantBehaviour>(lSystemBehaviour));
    PushInternodeBehaviour(std::dynamic_pointer_cast<IPlantBehaviour>(generalTreeBehaviour));

    m_randomColors.resize(8192);
    for (int i = 0; i < 8192; i++) {
        m_randomColors[i] = glm::abs(glm::sphericalRand(1.0f));
    }

    m_internodesQuery = Entities::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeInfo());
    m_branchesQuery = Entities::CreateEntityQuery();
    m_branchesQuery.SetAllFilters(BranchInfo());

#pragma region Internode camera
    m_visualizationCamera =
            Serialization::ProduceSerializable<Camera>();
    m_visualizationCamera->m_useClearColor = true;
    m_visualizationCamera->m_clearColor = glm::vec3(0.1f);
    m_visualizationCamera->OnCreate();
#pragma endregion

    m_voxelSpace.Reset();
}


void PlantLayer::LateUpdate() {
    UpdateInternodeCamera();
}

void PlantLayer::UpdateInternodeColors() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    auto focusingInternode = Entity();
    auto selectedEntity = Entity();
    if (m_currentFocusingInternode.Get().IsValid()) {
        focusingInternode = m_currentFocusingInternode.Get();
    }
    if (editorLayer->m_selectedEntity.IsValid()) {
        selectedEntity = editorLayer->m_selectedEntity;
    }

    Entities::ForEach<InternodeColor, InternodeInfo>(
            Entities::GetCurrentScene(), Jobs::Workers(),
            m_internodesQuery,
            [=](int i, Entity entity, InternodeColor &internodeRenderColor,
                InternodeInfo &internodeInfo) {
                internodeRenderColor.m_value = glm::vec4(m_internodeColor, m_internodeTransparency);
            },
            true);

    switch (m_branchColorMode) {
        case BranchColorMode::Order:
            Entities::ForEach<InternodeColor, InternodeStatus>(Entities::GetCurrentScene(),
                                                               Jobs::Workers(),
                                                               m_internodesQuery,
                                                               [=](int i, Entity entity,
                                                                   InternodeColor &internodeRenderColor,
                                                                   InternodeStatus &internodeStatus) {
                                                                   internodeRenderColor.m_value = glm::vec4(
                                                                           glm::vec3(m_internodeColorValueMultiplier *
                                                                                     glm::pow(
                                                                                             (float) internodeStatus.m_order,
                                                                                             m_internodeColorValueCompressFactor)),
                                                                           m_internodeTransparency);
                                                               },
                                                               true);
            break;
        case BranchColorMode::Level:
            Entities::ForEach<InternodeColor, InternodeStatus>(Entities::GetCurrentScene(),
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
            Entities::ForEach<InternodeColor, InternodeStatus, InternodeInfo, GeneralTreeParameters>(
                    Entities::GetCurrentScene(),
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
            Entities::ForEach<InternodeColor, InternodeWater>(Entities::GetCurrentScene(),
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
            Entities::ForEach<InternodeColor, InternodeWaterPressure>(Entities::GetCurrentScene(),
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
            Entities::ForEach<InternodeColor, InternodeInfo>(Entities::GetCurrentScene(),
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
            Entities::ForEach<InternodeColor, InternodeStatus>(Entities::GetCurrentScene(),
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
            Entities::ForEach<InternodeColor, InternodeStatistics>(Entities::GetCurrentScene(),
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
            Entities::ForEach<InternodeColor, InternodeStatistics>(Entities::GetCurrentScene(),
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
            Entities::ForEach<InternodeColor, InternodeStatistics>(Entities::GetCurrentScene(),
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
        default:
            break;
    }


    InternodeColor color;
    color.m_value = glm::vec4(1, 1, 1, 1);
    if (focusingInternode.IsValid() && focusingInternode.HasDataComponent<InternodeColor>())
        focusingInternode.SetDataComponent(color);
    color.m_value = glm::vec4(1, 0, 0, 1);
    if (selectedEntity.IsValid() && selectedEntity.HasDataComponent<InternodeColor>())
        selectedEntity.SetDataComponent(color);
}

void PlantLayer::UpdateInternodeCylinder() {
    Entities::ForEach<GlobalTransform, InternodeCylinder, InternodeCylinderWidth, InternodeInfo>(
            Entities::GetCurrentScene(),
            Jobs::Workers(),
            m_internodesQuery,
            [](int i, Entity entity, GlobalTransform &ltw, InternodeCylinder &c,
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

void PlantLayer::UpdateBranchCylinder() {
    Entities::ForEach<GlobalTransform, BranchCylinder, BranchCylinderWidth, BranchInfo>(
            Entities::GetCurrentScene(),
            Jobs::Workers(),
            m_branchesQuery,
            [](int i, Entity entity, GlobalTransform &ltw, BranchCylinder &c,
               BranchCylinderWidth &branchCylinderWidth, BranchInfo &internodeInfo) {
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

void PlantLayer::UpdateInternodePointer(const float &length, const float &width) {


}

void PlantLayer::RenderInternodeCylinders() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    std::vector<InternodeCylinder> branchCylinders;
    m_internodesQuery.ToComponentDataArray<InternodeCylinder>(Entities::GetCurrentScene(),
                                                              branchCylinders);
    std::vector<InternodeColor> branchColors;
    m_internodesQuery.ToComponentDataArray<InternodeColor>(Entities::GetCurrentScene(),
                                                           branchColors);
    if (!branchCylinders.empty())
        Graphics::DrawGizmoMeshInstancedColored(
                DefaultResources::Primitives::Cylinder, m_visualizationCamera,
                editorLayer->m_sceneCameraPosition,
                editorLayer->m_sceneCameraRotation,
                *reinterpret_cast<std::vector<glm::vec4> *>(&branchColors),
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchCylinders),
                glm::mat4(1.0f), 1.0f);
}

void PlantLayer::RenderInternodePointers() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    std::vector<InternodePointer> branchPointers;
    m_internodesQuery.ToComponentDataArray<InternodePointer>(Entities::GetCurrentScene(),
                                                             branchPointers);
    if (!branchPointers.empty())
        Graphics::DrawGizmoMeshInstanced(
                DefaultResources::Primitives::Cylinder, m_visualizationCamera,
                editorLayer->m_sceneCameraPosition,
                editorLayer->m_sceneCameraRotation, m_pointerColor,
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchPointers),
                glm::mat4(1.0f), 1.0f);
}


bool PlantLayer::InternodeCheck(const Entity &target) {
    return target.IsValid() && target.HasDataComponent<InternodeInfo>() && target.HasPrivateComponent<Internode>();
}


void PlantLayer::UpdateInternodeCamera() {
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

#pragma region Internode debug camera
    Camera::m_cameraInfoBlock.UpdateMatrices(
            editorLayer->m_sceneCamera,
            editorLayer->m_sceneCameraPosition,
            editorLayer->m_sceneCameraRotation);
    Camera::m_cameraInfoBlock.UploadMatrices(
            editorLayer->m_sceneCamera);
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
                                      "Proximity", "Inhibitor", "IndexDivider", "IndexRange", "StrahlerNumber"};

void PlantLayer::DrawColorModeSelectionMenu() {
    if (ImGui::TreeNodeEx("Branch Coloring")) {
        static int colorModeIndex = 0;
        if (ImGui::Combo("Color mode", &colorModeIndex, BranchColorModes,
                         IM_ARRAYSIZE(BranchColorModes))) {
            m_branchColorMode = (BranchColorMode) colorModeIndex;
        }

        ImGui::DragFloat("Multiplier", &m_internodeColorValueMultiplier, 0.01f);
        ImGui::DragFloat("Compress", &m_internodeColorValueCompressFactor, 0.01f);
        switch (m_branchColorMode) {
            case BranchColorMode::IndexDivider:
                ImGui::DragInt("Divider", &m_indexDivider, 1, 2, 1024);
                break;
            case BranchColorMode::IndexRange:
                ImGui::DragInt("Range Min", &m_indexRangeMin, 1, 0, m_indexRangeMax);
                ImGui::DragInt("Range Max", &m_indexRangeMax, 1, m_indexRangeMin, 999999);
                break;
        }
        ImGui::TreePop();
    }
}

void PlantLayer::CalculateStatistics() {
    auto scene = Entities::GetCurrentScene();
    for (auto &i: m_internodeBehaviours) {
        auto behaviour = i.Get<IPlantBehaviour>();
        if (behaviour) {
            std::vector<Entity> currentRoots;
            behaviour->m_rootsQuery.ToEntityArray(Entities::GetCurrentScene(), currentRoots);
            for (auto root: currentRoots) {
                root.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
                    if (!behaviour->InternodeCheck(child)) return;
                    behaviour->InternodeGraphWalker(child,
                                                    [](Entity parent, Entity child) {

                                                    },
                                                    [](Entity parent) {
                                                        auto parentStat = parent.GetDataComponent<InternodeStatistics>();
                                                        std::vector<int> indices;
                                                        parent.ForEachChild(
                                                                [&](const std::shared_ptr<Scene> &scene, Entity child) {
                                                                    indices.push_back(
                                                                            child.GetDataComponent<InternodeStatistics>().m_strahlerOrder);
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

                                                        parent.SetDataComponent(parentStat);
                                                    },
                                                    [](Entity endNode) {
                                                        auto endNodeStat = endNode.GetDataComponent<InternodeStatistics>();
                                                        endNodeStat.m_strahlerOrder = 1;
                                                        endNode.SetDataComponent(endNodeStat);
                                                    }
                    );
                });

            }
        }
    }

}

void PlantLayer::FixedUpdate() {
    if (Application::IsPlaying()) {
        if (m_enablePhysics) {
            if (m_applyFBMField) {
                std::vector<Entity> internodes;
                std::vector<glm::vec3> forces;
                internodes.clear();
                m_internodesQuery.ToEntityArray(Entities::GetCurrentScene(), internodes, false);
                forces.resize(internodes.size());
                Entities::ForEach<GlobalTransform>(Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
                                                   [&](int i, Entity entity, GlobalTransform &globalTransform) {
                                                       const auto position = entity.GetDataComponent<GlobalTransform>().GetPosition();
                                                       forces[i] = m_fBMField.GetT(position,
                                                                                   Application::Time().CurrentTime(),
                                                                                   10.0f, 0.02f, 6) *
                                                                   m_forceFactor;
                                                   }, false);

                for (int i = 0; i < internodes.size(); i++) {
                    auto &internode = internodes[i];
                    const auto position = internode.GetDataComponent<GlobalTransform>().GetPosition();
                    if (internode.IsEnabled() && internode.HasPrivateComponent<RigidBody>()) {
                        auto rigidBody = internode.GetOrSetPrivateComponent<RigidBody>().lock();
                        if (rigidBody->Registered()) rigidBody->AddForce(forces[i]);
                    }
                }
            }
        }
    }
}

void PlantLayer::UpdateBranchColors() {
    Entities::ForEach<BranchColor>(
            Entities::GetCurrentScene(), Jobs::Workers(),
            m_branchesQuery,
            [=](int i, Entity entity, BranchColor &branchRenderColor) {
                branchRenderColor.m_value = glm::vec4(m_branchColor, m_branchTransparency);
            },
            true);
}

void PlantLayer::RenderBranchCylinders() {
    auto editorLayer = Application::GetLayer<EditorLayer>();
    if (!editorLayer) return;
    std::vector<BranchCylinder> branchCylinders;
    m_branchesQuery.ToComponentDataArray<BranchCylinder>(Entities::GetCurrentScene(),
                                                         branchCylinders);
    std::vector<BranchColor> branchColors;
    m_branchesQuery.ToComponentDataArray<BranchColor>(Entities::GetCurrentScene(),
                                                      branchColors);
    if (!branchCylinders.empty())
        Graphics::DrawGizmoMeshInstancedColored(
                DefaultResources::Primitives::Cylinder, m_visualizationCamera,
                editorLayer->m_sceneCameraPosition,
                editorLayer->m_sceneCameraRotation,
                *reinterpret_cast<std::vector<glm::vec4> *>(&branchColors),
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchCylinders),
                glm::mat4(1.0f), 1.0f);
}


#pragma endregion