#include <RayTracedRenderer.hpp>
#include <RayTracerManager.hpp>
#include <ProjectManager.hpp>

using namespace RayTracerFacility;

void RayTracerManager::UpdateScene() const {
    bool rebuildAccelerationStructure = false;
    bool updateShaderBindingTable = false;
    auto &meshesStorage = CudaModule::GetRayTracer()->m_instances;
    for (auto &i : meshesStorage) {
        i.m_removeTag = true;
    }
    if (const auto *rayTracedEntities =
                EntityManager::UnsafeGetPrivateComponentOwnersList<
                        RayTracedRenderer>();
            rayTracedEntities) {
        for (auto entity : *rayTracedEntities) {
            if (!entity.IsEnabled())
                continue;
            auto rayTracedRenderer =
                    entity.GetOrSetPrivateComponent<RayTracedRenderer>().lock();
            if (!rayTracedRenderer->IsEnabled())
                continue;
            auto mesh = rayTracedRenderer->m_mesh.Get<Mesh>();
            if (!mesh || mesh->UnsafeGetVertices().empty())
                continue;
            auto globalTransform = entity.GetDataComponent<GlobalTransform>().m_value;
            RayTracerInstance newRayTracerInstance;
            RayTracerInstance *rayTracerInstance = &newRayTracerInstance;
            bool needVerticesUpdate = false;
            bool needTransformUpdate = false;
            bool fromNew = true;
            bool needMaterialUpdate = false;
            for (auto &currentRayTracerInstance : meshesStorage) {
                if (currentRayTracerInstance.m_entityId == entity.GetIndex() &&
                    currentRayTracerInstance.m_entityVersion == entity.GetVersion()) {
                    fromNew = false;
                    rayTracerInstance = &currentRayTracerInstance;
                    currentRayTracerInstance.m_removeTag = false;
                    if (globalTransform != currentRayTracerInstance.m_globalTransform) {
                        needTransformUpdate = true;
                    }
                    if (rayTracerInstance->m_version != mesh->GetVersion())
                        needVerticesUpdate = true;
                    if (rayTracerInstance->m_surfaceColor !=
                        rayTracedRenderer->m_surfaceColor ||
                        rayTracerInstance->m_metallic !=
                        (rayTracedRenderer->m_metallic == 1.0f
                         ? -1.0f
                         : 1.0f / glm::pow(1.0f - rayTracedRenderer->m_metallic,
                                           3.0f)) ||
                        rayTracerInstance->m_roughness !=
                        rayTracedRenderer->m_roughness
                            ) {
                        needMaterialUpdate = true;
                    }
                }
            }
            rayTracerInstance->m_version = mesh->GetVersion();
            if (fromNew || needVerticesUpdate || needTransformUpdate ||
                needMaterialUpdate) {
                updateShaderBindingTable = true;
                rayTracerInstance->m_surfaceColor = rayTracedRenderer->m_surfaceColor;
                rayTracerInstance->m_metallic =
                        rayTracedRenderer->m_metallic == 1.0f
                        ? -1.0f
                        : 1.0f / glm::pow(1.0f - rayTracedRenderer->m_metallic, 3.0f);
                rayTracerInstance->m_roughness = rayTracedRenderer->m_roughness;
                rayTracerInstance->m_normalTexture = 0;
                rayTracerInstance->m_albedoTexture = 0;
                rayTracerInstance->m_entityId = entity.GetIndex();
                rayTracerInstance->m_entityVersion = entity.GetVersion();
            }
            if (rayTracedRenderer->m_albedoTexture.Get<Texture2D>() &&
                rayTracedRenderer->m_albedoTexture.Get<Texture2D>()
                        ->Texture()
                        ->Id() != rayTracerInstance->m_albedoTexture) {
                updateShaderBindingTable = true;
                if (rayTracedRenderer->m_albedoTexture.Get<Texture2D>() &&
                    rayTracedRenderer->m_albedoTexture.Get<Texture2D>()
                            ->Texture()
                            ->IsResident()) {
                    UNIENGINE_ERROR("Texture is resident, can't be used!");
                    rayTracedRenderer->m_albedoTexture.Clear();
                    rayTracerInstance->m_albedoTexture = 0;
                } else {
                    rayTracerInstance->m_albedoTexture =
                            rayTracedRenderer->m_albedoTexture.Get<Texture2D>()
                                    ->Texture()
                                    ->Id();
                }
            } else if (!rayTracedRenderer->m_albedoTexture.Get<Texture2D>() &&
                       rayTracerInstance->m_albedoTexture != 0) {
                updateShaderBindingTable = true;
                rayTracerInstance->m_albedoTexture = 0;
            }
            if (rayTracedRenderer->m_normalTexture.Get<Texture2D>() &&
                rayTracedRenderer->m_normalTexture.Get<Texture2D>()
                        ->Texture()
                        ->Id() != rayTracerInstance->m_normalTexture) {
                updateShaderBindingTable = true;
                if (rayTracedRenderer->m_normalTexture.Get<Texture2D>() &&
                    rayTracedRenderer->m_normalTexture.Get<Texture2D>()
                            ->Texture()
                            ->IsResident()) {
                    UNIENGINE_ERROR("Texture is resident, can't be used!");
                    rayTracedRenderer->m_normalTexture.Clear();
                    rayTracerInstance->m_normalTexture = 0;
                } else {
                    rayTracerInstance->m_normalTexture =
                            rayTracedRenderer->m_normalTexture.Get<Texture2D>()
                                    ->Texture()
                                    ->Id();
                }
            } else if (!rayTracedRenderer->m_normalTexture.Get<Texture2D>() &&
                       rayTracerInstance->m_normalTexture != 0) {
                updateShaderBindingTable = true;
                rayTracerInstance->m_normalTexture = 0;
            }
            if (rayTracerInstance->m_diffuseIntensity !=
                rayTracedRenderer->m_diffuseIntensity) {
                updateShaderBindingTable = true;
                rayTracerInstance->m_diffuseIntensity =
                        rayTracedRenderer->m_diffuseIntensity;
            }
            if (fromNew || needVerticesUpdate) {
                rebuildAccelerationStructure = true;
                rayTracerInstance->m_verticesUpdateFlag = true;
                if (fromNew) {
                    rayTracerInstance->m_transformUpdateFlag = true;
                    rayTracerInstance->m_globalTransform = globalTransform;
                }
                rayTracerInstance->m_vertices =
                        reinterpret_cast<std::vector<Vertex> *>(&mesh->UnsafeGetVertices());
                rayTracerInstance->m_triangles = &mesh->UnsafeGetTriangles();
            } else if (needTransformUpdate) {
                rebuildAccelerationStructure = true;
                rayTracerInstance->m_globalTransform = globalTransform;
                rayTracerInstance->m_transformUpdateFlag = true;
            }
            if (fromNew)
                meshesStorage.push_back(newRayTracerInstance);
        }
    } else {
        for (auto &i : meshesStorage) {
            i.m_removeTag = true;
        }
    }
    for (int i = 0; i < meshesStorage.size(); i++) {
        if (meshesStorage[i].m_removeTag) {
            meshesStorage.erase(meshesStorage.begin() + i);
            i--;
            rebuildAccelerationStructure = true;
        }
    }
    if (rebuildAccelerationStructure && !meshesStorage.empty()) {
        CudaModule::GetRayTracer()->BuildAccelerationStructure();
        CudaModule::GetRayTracer()->ClearAccumulate();
    } else if (updateShaderBindingTable) {
        CudaModule::GetRayTracer()->ClearAccumulate();
    }
}

RayTracerManager &RayTracerManager::GetInstance() {
    static RayTracerManager instance;
    return instance;
}

void RayTracerManager::Init() {


    auto &manager = GetInstance();

    CudaModule::Init();

    manager.m_defaultWindow.Init("Ray:Default");
#pragma region Environmental map
    {
        manager.m_environmentalMap = AssetManager::Import<Cubemap>(
                AssetManager::GetResourceFolderPath() /
                "Textures/Cubemaps/GrandCanyon/GCanyon_C_YumaPoint_3k.hdr");
    }
#pragma endregion

    Application::RegisterUpdateFunction([]() {
        Update();
        OnGui();
    });
}

void RayTracerRenderWindow::Init(const std::string &name) {
    m_output =
            std::make_unique<OpenGLUtils::GLTexture2D>(0, GL_RGBA32F, 1, 1, false);
    m_output->SetData(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0);
    m_output->SetInt(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    m_output->SetInt(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    m_output->SetInt(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    m_output->SetInt(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    m_name = name;
}

void RayTracerManager::Update() {
    auto &manager = GetInstance();
    manager.UpdateScene();
    if (manager.m_defaultWindow.m_renderingEnabled) {
        const auto size = manager.m_defaultWindow.Resize();
        manager.m_defaultRenderingProperties.m_camera.Set(
                EditorManager::GetInstance().m_sceneCameraRotation,
                EditorManager::GetInstance().m_sceneCameraPosition,
                EditorManager::GetInstance().m_sceneCamera->m_fov, size);
        manager.m_defaultRenderingProperties.m_environmentalMapId =
                manager.m_environmentalMap->Texture()->Id();
        manager.m_defaultRenderingProperties.m_frameSize = size;
        manager.m_defaultRenderingProperties.m_outputTextureId =
                manager.m_defaultWindow.m_output->Id();
        if (!CudaModule::GetRayTracer()->m_instances.empty()) {
            manager.m_defaultWindow.m_rendered =
                    CudaModule::GetRayTracer()->RenderDefault(
                            manager.m_defaultRenderingProperties);
        }
    }
}

void RayTracerManager::OnGui() {
    auto &manager = GetInstance();
    manager.m_defaultWindow.OnGui();
    manager.m_defaultRenderingProperties.OnGui();
}

void RayTracerManager::End() { CudaModule::Terminate(); }

glm::ivec2 RayTracerRenderWindow::Resize() const {
    glm::ivec2 size = glm::vec2(m_outputSize) * m_resolutionMultiplier;
    if (size.x < 1)
        size.x = 1;
    if (size.y < 1)
        size.y = 1;
    m_output->ReSize(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, nullptr, size.x, size.y);
    return size;
}

void RayTracerRenderWindow::OnGui() {
    if (m_rightMouseButtonHold &&
        !InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT,
                                        WindowManager::GetWindow())) {
        m_rightMouseButtonHold = false;
        m_startMouse = false;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0});
    ImGui::Begin(m_name.c_str());
    {
        if (ImGui::BeginChild("CameraRenderer", ImVec2(0, 0), false,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{5, 5});
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Settings")) {
                    ImGui::DragFloat("Resolution multiplier", &m_resolutionMultiplier,
                                     0.01f, 0.1f, 1.0f);
                    ImGui::DragFloat("FOV",
                                     &EditorManager::GetInstance().m_sceneCamera->m_fov, 1,
                                     1, 120);
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
            ImGui::PopStyleVar();
            ImVec2 viewPortSize = ImGui::GetWindowSize();
            viewPortSize.y -= 20;
            if (viewPortSize.y < 0)
                viewPortSize.y = 0;
            m_outputSize = glm::ivec2(viewPortSize.x, viewPortSize.y);
            if (m_rendered)
                ImGui::Image(reinterpret_cast<ImTextureID>(m_output->Id()),
                             viewPortSize, ImVec2(0, 1), ImVec2(1, 0));
            else
                ImGui::Text("No mesh in the scene!");
            if (ImGui::IsWindowFocused()) {
                const bool valid = true;
                const glm::vec2 mousePosition =
                        InputManager::GetMouseAbsolutePositionInternal(
                                WindowManager::GetWindow());
                if (valid) {
                    if (!m_startMouse) {
                        m_lastX = mousePosition.x;
                        m_lastY = mousePosition.y;
                        m_startMouse = true;
                    }
                    const float xOffset = mousePosition.x - m_lastX;
                    const float yOffset = -mousePosition.y + m_lastY;
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
                        const glm::vec3 front =
                                EditorManager::GetInstance().m_sceneCameraRotation *
                                glm::vec3(0, 0, -1);
                        const glm::vec3 right =
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
                                    UniEngine::Camera::ProcessMouseMovement(
                                            EditorManager::GetInstance().m_sceneCameraYawAngle,
                                            EditorManager::GetInstance().m_sceneCameraPitchAngle,
                                            false);
                        }
                    }
#pragma endregion
                }
            }
        }
        ImGui::EndChild();
        auto *window = ImGui::FindWindowByName(m_name.c_str());
        m_renderingEnabled = !(window->Hidden && !window->Collapsed);
    }
    ImGui::End();
    ImGui::PopStyleVar();
}