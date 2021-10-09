//
// Created by lllll on 8/27/2021.
//

#include "InternodeSystem.hpp"
#include "GeneralTreeBehaviour.hpp"
#include "SpaceColonizationBehaviour.hpp"
#include "GeneralTreeParameters.hpp"
#include "LSystemBehaviour.hpp"
#include "CubeVolume.hpp"
#include "RadialBoundingVolume.hpp"
using namespace PlantArchitect;

void InternodeSystem::Simulate(int iterations) {
    for (int iteration = 0; iteration < iterations; iteration++) {
        std::vector<Entity> internodeEntities;
        std::vector<GlobalTransform> internodeGlobalTransforms;
        std::vector<InternodeInfo> internodeInfos;
        std::vector<glm::vec3> internodePositions;
        m_internodesQuery.ToComponentDataArray(EntityManager::GetCurrentScene(), internodeGlobalTransforms);
        m_internodesQuery.ToComponentDataArray(EntityManager::GetCurrentScene(), internodeInfos);
        m_internodesQuery.ToEntityArray(EntityManager::GetCurrentScene(), internodeEntities);
        internodePositions.resize(internodeGlobalTransforms.size());
        for (int i = 0; i < internodeGlobalTransforms.size(); i++) {
            internodePositions[i] = internodeGlobalTransforms[i].GetPosition() + internodeInfos[i].m_length *
                                                                                 (internodeGlobalTransforms[i].GetRotation() *
                                                                                  glm::vec3(0, 0, -1));
        }
        std::vector<float> proximity;
        proximity.resize(internodePositions.size());
        for (int i = 0; i < internodePositions.size(); i++) {
            internodeInfos[i].m_neighborsProximity = 0;
            const auto &position = internodePositions[i];
            for (int j = 0; j < internodePositions.size(); j++) {
                if (i == j) continue;
                auto distance = glm::max(1.0f, glm::distance(internodePositions[i], internodePositions[j]));
                internodeInfos[i].m_neighborsProximity += 1.0f / (distance * distance);
            }
            internodeEntities[i].SetDataComponent(internodeInfos[i]);
        }

        for (auto &i: m_internodeBehaviours) {
            auto behaviour = i.Get<IInternodeBehaviour>();
            if (behaviour) behaviour->Grow(iteration);
        }
    }
}

#pragma region Methods

void InternodeSystem::OnInspect() {
    static int iterations = 1;
    ImGui::DragInt("Iterations", &iterations);
    if (ImGui::Button("Simulate")) {
        Simulate(iterations);
    }

    ImGui::Text("Add Internode Behaviour");
    ImGui::SameLine();
    static AssetRef temp;
    EditorManager::DragAndDropButton(temp, "Here",
                                     {"GeneralTreeBehaviour", "SpaceColonizationBehaviour", "LSystemBehaviour"}, false);
    if (temp.Get<IInternodeBehaviour>()) {
        PushInternodeBehaviour(temp.Get<IInternodeBehaviour>());
        temp.Clear();
    }
    if (ImGui::TreeNodeEx("Internode Behaviours", ImGuiTreeNodeFlags_DefaultOpen)) {
        int index = 0;
        bool skip = false;
        for (auto &i: m_internodeBehaviours) {
            auto ptr = i.Get<IInternodeBehaviour>();
            ImGui::Button(("Slot " + std::to_string(index) + ": " + ptr->m_name).c_str());
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                EditorManager::GetInstance().m_inspectingAsset = ptr;
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
                if (!i.Get<IInternodeBehaviour>()) {
                    m_internodeBehaviours.erase(m_internodeBehaviours.begin() + index2);
                    break;
                }
                index2++;
            }
        }
        ImGui::TreePop();
    }
}

void InternodeSystem::OnCreate() {
    auto spaceColonizationBehaviour = AssetManager::CreateAsset<SpaceColonizationBehaviour>();
    auto lSystemBehaviour = AssetManager::CreateAsset<LSystemBehaviour>();
    auto generalTreeBehaviour = AssetManager::CreateAsset<GeneralTreeBehaviour>();
    PushInternodeBehaviour(
            std::dynamic_pointer_cast<IInternodeBehaviour>(spaceColonizationBehaviour));
    PushInternodeBehaviour(std::dynamic_pointer_cast<IInternodeBehaviour>(lSystemBehaviour));
    PushInternodeBehaviour(std::dynamic_pointer_cast<IInternodeBehaviour>(generalTreeBehaviour));



    m_randomColors.resize(64);
    for (int i = 0; i < 60; i++) {
        m_randomColors[i] = glm::sphericalRand(1.0f);
    }

    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeInfo());

#pragma region Internode camera
    m_internodeDebuggingCamera =
            SerializationManager::ProduceSerializable<Camera>();
    m_internodeDebuggingCamera->m_useClearColor = true;
    m_internodeDebuggingCamera->m_clearColor = glm::vec3(0.1f);
    m_internodeDebuggingCamera->OnCreate();
#pragma endregion
    Enable();
}


void InternodeSystem::LateUpdate() {
    if (m_rightMouseButtonHold &&
        !InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT,
                                        WindowManager::GetWindow())) {
        m_rightMouseButtonHold = false;
        m_startMouse = false;
    }
    m_internodeDebuggingCamera->ResizeResolution(
            m_internodeDebuggingCameraResolutionX,
            m_internodeDebuggingCameraResolutionY);
    m_internodeDebuggingCamera->Clear();

#pragma region Internode debug camera
    Camera::m_cameraInfoBlock.UpdateMatrices(
            EditorManager::GetInstance().m_sceneCamera,
            EditorManager::GetInstance().m_sceneCameraPosition,
            EditorManager::GetInstance().m_sceneCameraRotation);
    Camera::m_cameraInfoBlock.UploadMatrices(
            EditorManager::GetInstance().m_sceneCamera);
#pragma endregion

#pragma region Rendering
    if (m_drawBranches) {
        if (m_autoUpdate) {
            UpdateBranchColors();
            UpdateBranchCylinder(m_connectionWidth);
        }
        if (m_internodeDebuggingCamera->IsEnabled())
            RenderBranchCylinders();
    }
    if (m_drawPointers) {
        if (m_autoUpdate) {
            UpdateBranchPointer(m_pointerLength, m_pointerWidth);
        }
        if (m_internodeDebuggingCamera->IsEnabled())
            RenderBranchPointers();
    }
#pragma endregion

#pragma region Internode debugging camera
    ImVec2 viewPortSize;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0});
    ImGui::Begin("Internodes");
    {
        if (ImGui::BeginChild("CameraRenderer", ImVec2(0, 0), false,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Settings")) {
#pragma region Menu
                    ImGui::Checkbox("Connections", &m_drawBranches);
                    if (m_drawBranches) {

                        if (ImGui::TreeNodeEx("Connection settings",
                                              ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::Checkbox("Auto update", &m_autoUpdate);
                            ImGui::SliderFloat("Alpha", &m_transparency, 0, 1);
                            ImGui::DragFloat("Connection width", &m_connectionWidth, 0.01f, 0.01f, 1.0f);

                            static const char *ColorModes[]{"None", "Order", "Level", "Water", "ApicalControl",
                                                            "WaterPressure",
                                                            "Proximity", "Inhibitor"};
                            static int colorModeIndex = 0;
                            if (ImGui::Combo("Color mode", &colorModeIndex, ColorModes,
                                             IM_ARRAYSIZE(ColorModes))) {
                                m_branchColorMode = (BranchColorMode) colorModeIndex;
                            }
                            ImGui::DragFloat("Multiplier", &m_branchColorValueMultiplier, 0.01f);
                            ImGui::DragFloat("Compress", &m_branchColorValueCompressFactor, 0.01f);
                            switch(m_branchColorMode){
                                case BranchColorMode::IndexDivider:
                                    ImGui::DragInt("Divider", &m_indexDivider, 1, 1, 1024);
                                    break;
                            }
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
            m_internodeDebuggingCameraResolutionX = viewPortSize.x;
            m_internodeDebuggingCameraResolutionY = viewPortSize.y;
            ImGui::Image(
                    reinterpret_cast<ImTextureID>(
                            m_internodeDebuggingCamera->GetTexture()->UnsafeGetGLTexture()->Id()),
                    viewPortSize, ImVec2(0, 1), ImVec2(1, 0));
            glm::vec2 mousePosition = glm::vec2(FLT_MAX, FLT_MIN);
            if (ImGui::IsWindowFocused()) {
                bool valid = true;
                mousePosition = InputManager::GetMouseAbsolutePositionInternal(
                        WindowManager::GetWindow());
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
                        InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT,
                                                       WindowManager::GetWindow())) {
                        m_rightMouseButtonHold = true;
                    }
                    if (m_rightMouseButtonHold &&
                        !EditorManager::GetInstance().m_lockCamera) {
                        glm::vec3 front =
                                EditorManager::GetInstance().m_sceneCameraRotation *
                                glm::vec3(0, 0, -1);
                        glm::vec3 right =
                                EditorManager::GetInstance().m_sceneCameraRotation *
                                glm::vec3(1, 0, 0);
                        if (InputManager::GetKeyInternal(GLFW_KEY_W,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition +=
                                    front * static_cast<float>(Application::Time().DeltaTime()) *
                                    EditorManager::GetInstance().m_velocity;
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_S,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition -=
                                    front * static_cast<float>(Application::Time().DeltaTime()) *
                                    EditorManager::GetInstance().m_velocity;
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_A,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition -=
                                    right * static_cast<float>(Application::Time().DeltaTime()) *
                                    EditorManager::GetInstance().m_velocity;
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_D,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition +=
                                    right * static_cast<float>(Application::Time().DeltaTime()) *
                                    EditorManager::GetInstance().m_velocity;
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_LEFT_SHIFT,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition.y +=
                                    EditorManager::GetInstance().m_velocity *
                                    static_cast<float>(Application::Time().DeltaTime());
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_LEFT_CONTROL,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition.y -=
                                    EditorManager::GetInstance().m_velocity *
                                    static_cast<float>(Application::Time().DeltaTime());
                        }
                        if (xOffset != 0.0f || yOffset != 0.0f) {
                            EditorManager::GetInstance().m_sceneCameraYawAngle +=
                                    xOffset * EditorManager::GetInstance().m_sensitivity;
                            EditorManager::GetInstance().m_sceneCameraPitchAngle +=
                                    yOffset * EditorManager::GetInstance().m_sensitivity;
                            if (EditorManager::GetInstance().m_sceneCameraPitchAngle > 89.0f)
                                EditorManager::GetInstance().m_sceneCameraPitchAngle = 89.0f;
                            if (EditorManager::GetInstance().m_sceneCameraPitchAngle < -89.0f)
                                EditorManager::GetInstance().m_sceneCameraPitchAngle = -89.0f;

                            EditorManager::GetInstance().m_sceneCameraRotation =
                                    Camera::ProcessMouseMovement(
                                            EditorManager::GetInstance().m_sceneCameraYawAngle,
                                            EditorManager::GetInstance().m_sceneCameraPitchAngle,
                                            false);
                        }
                    }
#pragma endregion
                    if (m_drawBranches) {
#pragma region Ray selection
                        m_currentFocusingInternode = Entity();
                        std::mutex writeMutex;
                        auto windowPos = ImGui::GetWindowPos();
                        auto windowSize = ImGui::GetWindowSize();
                        mousePosition.x -= windowPos.x;
                        mousePosition.x -= windowSize.x;
                        mousePosition.y -= windowPos.y + 20;
                        float minDistance = FLT_MAX;
                        GlobalTransform cameraLtw;
                        cameraLtw.m_value =
                                glm::translate(
                                        EditorManager::GetInstance().m_sceneCameraPosition) *
                                glm::mat4_cast(
                                        EditorManager::GetInstance().m_sceneCameraRotation);
                        const Ray cameraRay = m_internodeDebuggingCamera->ScreenPointToRay(
                                cameraLtw, mousePosition);
                        EntityManager::ForEach<GlobalTransform, InternodeInfo>(
                                EntityManager::GetCurrentScene(), JobManager::PrimaryWorkers(),
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
                        if (InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_LEFT,
                                                           WindowManager::GetWindow())) {
                            if (!m_currentFocusingInternode.Get().IsNull()) {
                                EditorManager::SetSelectedEntity(
                                        m_currentFocusingInternode.Get());
                            }
                        }
#pragma endregion
                    }
                }
            }
        }
        ImGui::EndChild();
        auto *window = ImGui::FindWindowByName("Internodes");
        m_internodeDebuggingCamera->SetEnabled(
                !(window->Hidden && !window->Collapsed));
    }
    ImGui::End();
    ImGui::PopStyleVar();

#pragma endregion
}

void InternodeSystem::UpdateBranchColors() {
    auto focusingInternode = Entity();
    auto selectedEntity = Entity();
    if (m_currentFocusingInternode.Get().IsValid()) {
        focusingInternode = m_currentFocusingInternode.Get();
    }
    if (EditorManager::GetSelectedEntity().IsValid()) {
        selectedEntity = EditorManager::GetSelectedEntity();
    }

    EntityManager::ForEach<BranchColor, InternodeInfo>(
            EntityManager::GetCurrentScene(), JobManager::PrimaryWorkers(),
            m_internodesQuery,
            [=](int i, Entity entity, BranchColor &internodeRenderColor,
                InternodeInfo &internodeInfo) {
                internodeRenderColor.m_value = glm::vec4(m_branchColor, m_transparency);
            },
            true);

    switch (m_branchColorMode) {
        case BranchColorMode::Order:
            EntityManager::ForEach<BranchColor, InternodeStatus>(EntityManager::GetCurrentScene(),
                                                                 JobManager::PrimaryWorkers(),
                    m_internodesQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatus &internodeStatus) {
                        internodeRenderColor.m_value = glm::vec4(glm::vec3(m_branchColorValueMultiplier *
                                                                           glm::pow((float) internodeStatus.m_order,
                                                                                    m_branchColorValueCompressFactor)),
                                                                 m_transparency);
                    },
                    true);
            break;
        case BranchColorMode::Level:
            EntityManager::ForEach<BranchColor, InternodeStatus>(EntityManager::GetCurrentScene(),
                                                                 JobManager::PrimaryWorkers(),
                    m_internodesQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatus &internodeStatus) {
                        internodeRenderColor.m_value = glm::vec4(glm::vec3(m_branchColorValueMultiplier *
                                                                           glm::pow((float) internodeStatus.m_level,
                                                                                    m_branchColorValueCompressFactor)),
                                                                 m_transparency);
                    },
                    true);
            break;
        case BranchColorMode::ApicalControl:
            EntityManager::ForEach<BranchColor, InternodeStatus, InternodeInfo, GeneralTreeParameters>(EntityManager::GetCurrentScene(),
                                                                                                       JobManager::PrimaryWorkers(),
                    m_internodesQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatus &internodeStatus, InternodeInfo &internodeInfo,
                        GeneralTreeParameters &parameters) {
                        internodeRenderColor.m_value = glm::vec4(glm::vec3(m_branchColorValueMultiplier *
                                                                           glm::pow(internodeStatus.m_apicalControl,
                                                                                    m_branchColorValueCompressFactor)),
                                                                 m_transparency);
                    },
                    true);
            break;
        case BranchColorMode::Water:
            EntityManager::ForEach<BranchColor, InternodeWater>(EntityManager::GetCurrentScene(),
                                                                JobManager::PrimaryWorkers(),
                    m_internodesQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeWater &internodeWater) {
                        internodeRenderColor.m_value = glm::vec4(glm::vec3(m_branchColorValueMultiplier *
                                                                           glm::pow(internodeWater.m_value,
                                                                                    m_branchColorValueCompressFactor)),
                                                                 m_transparency);
                    },
                    true);
            break;
        case BranchColorMode::WaterPressure:
            EntityManager::ForEach<BranchColor, InternodeWaterPressure>(EntityManager::GetCurrentScene(),
                                                                        JobManager::PrimaryWorkers(),
                    m_internodesQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeWaterPressure &internodeWaterPressure) {
                        internodeRenderColor.m_value = glm::vec4(glm::vec3(m_branchColorValueMultiplier *
                                                                           glm::pow(internodeWaterPressure.m_value,
                                                                                    m_branchColorValueCompressFactor)),
                                                                 m_transparency);
                    },
                    true);
            break;
        case BranchColorMode::Proximity:
            EntityManager::ForEach<BranchColor, InternodeInfo>(EntityManager::GetCurrentScene(),
                                                               JobManager::PrimaryWorkers(),
                    m_internodesQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeInfo &internodeInfo) {
                        internodeRenderColor.m_value = glm::vec4(glm::vec3(m_branchColorValueMultiplier *
                                                                           glm::pow(internodeInfo.m_neighborsProximity,
                                                                                    m_branchColorValueCompressFactor)),
                                                                 m_transparency);
                    },
                    true);
            break;
        case BranchColorMode::Inhibitor:
            EntityManager::ForEach<BranchColor, InternodeStatus>(EntityManager::GetCurrentScene(),
                                                                 JobManager::PrimaryWorkers(),
                    m_internodesQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatus &internodeStatus) {
                        internodeRenderColor.m_value = glm::vec4(glm::vec3(m_branchColorValueMultiplier *
                                                                           glm::pow(internodeStatus.m_inhibitor,
                                                                                    m_branchColorValueCompressFactor)),
                                                                 m_transparency);
                    },
                    true);
            break;
        case BranchColorMode::IndexDivider:
            EntityManager::ForEach<BranchColor, InternodeInfo>(EntityManager::GetCurrentScene(),
                                                                 JobManager::PrimaryWorkers(),
                                                                 m_internodesQuery,
                                                                 [=](int i, Entity entity, BranchColor &internodeRenderColor,
                                                                     InternodeInfo &internodeInfo) {
                                                                     internodeRenderColor.m_value = glm::vec4(glm::vec3(m_randomColors[internodeInfo.m_index / m_indexDivider]),
                                                                                                              1.0f);
                                                                 },
                                                                 true);
            break;
        default:
            break;
    }


    BranchColor color;
    color.m_value = glm::vec4(1, 1, 1, 1);
    if (focusingInternode.IsValid() && focusingInternode.HasDataComponent<BranchColor>())
        focusingInternode.SetDataComponent(color);
    color.m_value = glm::vec4(1, 0, 0, 1);
    if (selectedEntity.IsValid() && selectedEntity.HasDataComponent<BranchColor>())
        selectedEntity.SetDataComponent(color);
}

void InternodeSystem::UpdateBranchCylinder(const float &width) {
    EntityManager::ForEach<GlobalTransform, BranchCylinder, BranchCylinderWidth, InternodeInfo>(EntityManager::GetCurrentScene(),
                                                                                                JobManager::PrimaryWorkers(),
            m_internodesQuery,
            [width](int i, Entity entity, GlobalTransform &ltw, BranchCylinder &c,
                    BranchCylinderWidth &branchCylinderWidth, InternodeInfo &internodeInfo) {
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

void InternodeSystem::UpdateBranchPointer(const float &length, const float &width) {


}

void InternodeSystem::RenderBranchCylinders() {
    std::vector<BranchCylinder> branchCylinders;
    m_internodesQuery.ToComponentDataArray<BranchCylinder>(EntityManager::GetCurrentScene(),
                                                           branchCylinders);
    std::vector<BranchColor> branchColors;
    m_internodesQuery.ToComponentDataArray<BranchColor>(EntityManager::GetCurrentScene(),
                                                        branchColors);
    if (!branchCylinders.empty())
        RenderManager::DrawGizmoMeshInstancedColored(
                DefaultResources::Primitives::Cylinder, m_internodeDebuggingCamera,
                EditorManager::GetInstance().m_sceneCameraPosition,
                EditorManager::GetInstance().m_sceneCameraRotation,
                *reinterpret_cast<std::vector<glm::vec4> *>(&branchColors),
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchCylinders),
                glm::mat4(1.0f), 1.0f);
}

void InternodeSystem::RenderBranchPointers() {
    std::vector<BranchPointer> branchPointers;
    m_internodesQuery.ToComponentDataArray<BranchPointer>(EntityManager::GetCurrentScene(),
                                                          branchPointers);
    if (!branchPointers.empty())
        RenderManager::DrawGizmoMeshInstanced(
                DefaultResources::Primitives::Cylinder, m_internodeDebuggingCamera,
                EditorManager::GetInstance().m_sceneCameraPosition,
                EditorManager::GetInstance().m_sceneCameraRotation, m_pointerColor,
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchPointers),
                glm::mat4(1.0f), 1.0f);
}


bool InternodeSystem::InternodeCheck(const Entity &target) {
    return target.IsValid() && target.HasDataComponent<InternodeInfo>() && target.HasPrivateComponent<Internode>();
}


void InternodeSystem::CollectAssetRef(std::vector<AssetRef> &list) {
    list.insert(list.begin(), m_internodeBehaviours.begin(), m_internodeBehaviours.end());
}

void InternodeSystem::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_internodeBehaviours" << YAML::Value << YAML::BeginSeq;
    for(auto& i : m_internodeBehaviours){
        out << YAML::BeginMap;
        i.Serialize(out);
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
}

void InternodeSystem::Deserialize(const YAML::Node &in) {
    m_internodeBehaviours.clear();
    if(in["m_internodeBehaviours"]){
        for(const auto& i : in["m_internodeBehaviours"]){
            AssetRef behaviour;
            behaviour.Deserialize(i);
            PushInternodeBehaviour(behaviour.Get<IInternodeBehaviour>());
        }
    }
}

#pragma endregion

