#include "MLVQRenderer.hpp"
#include <RayTracerManager.hpp>

using namespace RayTracerFacility;

#include <ProjectManager.hpp>

void RayTracerManager::UpdateMeshesStorage(
    std::vector<RayTracerInstance> &meshesStorage,
    bool &rebuildAccelerationStructure, bool &updateShaderBindingTable) const {
  for (auto &i : meshesStorage) {
    i.m_removeTag = true;
  }
  if (const auto *rayTracedEntities =
          EntityManager::UnsafeGetPrivateComponentOwnersList<MeshRenderer>(
              EntityManager::GetCurrentScene());
      rayTracedEntities) {
    for (auto entity : *rayTracedEntities) {
      if (!entity.IsEnabled())
        continue;
      if (entity.HasPrivateComponent<MLVQRenderer>() &&
          entity.GetOrSetPrivateComponent<MLVQRenderer>().lock()->IsEnabled())
        continue;
      auto meshRenderer =
          entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
      if (!meshRenderer->IsEnabled())
        continue;
      auto mesh = meshRenderer->m_mesh.Get<Mesh>();
      auto material = meshRenderer->m_material.Get<Material>();
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
            currentRayTracerInstance.m_entityVersion == entity.GetVersion() &&
            currentRayTracerInstance.m_materialType == MaterialType::Default) {
          fromNew = false;
          rayTracerInstance = &currentRayTracerInstance;
          currentRayTracerInstance.m_removeTag = false;
          if (globalTransform != currentRayTracerInstance.m_globalTransform) {
            needTransformUpdate = true;
          }
          if (rayTracerInstance->m_version != mesh->GetVersion())
            needVerticesUpdate = true;
          if (rayTracerInstance->m_surfaceColor != material->m_albedoColor ||
              rayTracerInstance->m_metallic !=
                  (material->m_metallic == 1.0f
                       ? -1.0f
                       : 1.0f / glm::pow(1.0f - material->m_metallic, 3.0f)) ||
              rayTracerInstance->m_roughness != material->m_roughness) {
            needMaterialUpdate = true;
          }
        }
      }
      rayTracerInstance->m_materialType = MaterialType::Default;
      rayTracerInstance->m_version = mesh->GetVersion();
      if (fromNew || needVerticesUpdate || needTransformUpdate ||
          needMaterialUpdate) {
        updateShaderBindingTable = true;
        rayTracerInstance->m_surfaceColor = material->m_albedoColor;
        rayTracerInstance->m_metallic =
            material->m_metallic == 1.0f
                ? -1.0f
                : 1.0f / glm::pow(1.0f - material->m_metallic, 3.0f);
        rayTracerInstance->m_roughness = material->m_roughness;
        rayTracerInstance->m_normalTexture = 0;
        rayTracerInstance->m_albedoTexture = 0;
        rayTracerInstance->m_entityId = entity.GetIndex();
        rayTracerInstance->m_entityVersion = entity.GetVersion();
      }
      if (material->m_albedoTexture.Get<Texture2D>() &&
          material->m_albedoTexture.Get<Texture2D>()->UnsafeGetGLTexture()) {
        if (material->m_albedoTexture.Get<Texture2D>()
                ->UnsafeGetGLTexture()
                ->Id() != rayTracerInstance->m_albedoTexture) {
          updateShaderBindingTable = true;
          rayTracerInstance->m_albedoTexture =
              material->m_albedoTexture.Get<Texture2D>()
                  ->UnsafeGetGLTexture()
                  ->Id();
        }
      } else if (rayTracerInstance->m_albedoTexture != 0) {
        updateShaderBindingTable = true;
        rayTracerInstance->m_albedoTexture = 0;
      }

      if (material->m_normalTexture.Get<Texture2D>() &&
          material->m_normalTexture.Get<Texture2D>()->UnsafeGetGLTexture()) {
        if (material->m_normalTexture.Get<Texture2D>()
                ->UnsafeGetGLTexture()
                ->Id() != rayTracerInstance->m_normalTexture) {
          updateShaderBindingTable = true;
          rayTracerInstance->m_normalTexture =
              material->m_normalTexture.Get<Texture2D>()
                  ->UnsafeGetGLTexture()
                  ->Id();
        }
      } else if (rayTracerInstance->m_normalTexture != 0) {
        updateShaderBindingTable = true;
        rayTracerInstance->m_normalTexture = 0;
      }

      if (rayTracerInstance->m_diffuseIntensity != material->m_emission) {
        updateShaderBindingTable = true;
        rayTracerInstance->m_diffuseIntensity = material->m_emission;
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
  }
  if (const auto *rayTracedEntities =
          EntityManager::UnsafeGetPrivateComponentOwnersList<MLVQRenderer>(
              EntityManager::GetCurrentScene());
      rayTracedEntities) {
    for (auto entity : *rayTracedEntities) {
      if (!entity.IsEnabled())
        continue;
      auto mLVQRenderer =
          entity.GetOrSetPrivateComponent<MLVQRenderer>().lock();
      if (!mLVQRenderer->IsEnabled())
        continue;
      auto mesh = mLVQRenderer->m_mesh.Get<Mesh>();
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
            currentRayTracerInstance.m_entityVersion == entity.GetVersion() &&
            currentRayTracerInstance.m_materialType == MaterialType::MLVQ) {
          fromNew = false;
          rayTracerInstance = &currentRayTracerInstance;
          currentRayTracerInstance.m_removeTag = false;
          if (globalTransform != currentRayTracerInstance.m_globalTransform) {
            needTransformUpdate = true;
          }
          if (rayTracerInstance->m_version != mesh->GetVersion())
            needVerticesUpdate = true;
          if (rayTracerInstance->m_MLVQMaterialIndex !=
              mLVQRenderer->m_materialIndex) {
            needMaterialUpdate = true;
          }
        }
      }
      rayTracerInstance->m_materialType = MaterialType::MLVQ;
      rayTracerInstance->m_version = mesh->GetVersion();
      if (fromNew || needVerticesUpdate || needTransformUpdate ||
          needMaterialUpdate) {
        updateShaderBindingTable = true;
        rayTracerInstance->m_MLVQMaterialIndex = mLVQRenderer->m_materialIndex;
        rayTracerInstance->m_normalTexture = 0;
        rayTracerInstance->m_albedoTexture = 0;
        rayTracerInstance->m_entityId = entity.GetIndex();
        rayTracerInstance->m_entityVersion = entity.GetVersion();
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
  }
  for (int i = 0; i < meshesStorage.size(); i++) {
    if (meshesStorage[i].m_removeTag) {
      meshesStorage.erase(meshesStorage.begin() + i);
      i--;
      rebuildAccelerationStructure = true;
    }
  }
}

void RayTracerManager::UpdateSkinnedMeshesStorage(
    std::vector<SkinnedRayTracerInstance> &meshesStorage,
    bool &rebuildAccelerationStructure, bool &updateShaderBindingTable) const {
  for (auto &i : meshesStorage) {
    i.m_removeTag = true;
  }
  if (const auto *rayTracedEntities =
          EntityManager::UnsafeGetPrivateComponentOwnersList<
              SkinnedMeshRenderer>(EntityManager::GetCurrentScene());
      rayTracedEntities) {
    for (auto entity : *rayTracedEntities) {
      if (!entity.IsEnabled())
        continue;
      auto skinnedMeshRenderer =
          entity.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
      if (!skinnedMeshRenderer->IsEnabled())
        continue;
      auto mesh = skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>();
      auto material = skinnedMeshRenderer->m_material.Get<Material>();
      if (!mesh || mesh->UnsafeGetSkinnedVertices().empty())
        continue;
      auto globalTransform =
          skinnedMeshRenderer->RagDoll()
              ? glm::mat4(1.0f)
              : entity.GetDataComponent<GlobalTransform>().m_value;
      SkinnedRayTracerInstance newRayTracerInstance;
      SkinnedRayTracerInstance *rayTracerInstance = &newRayTracerInstance;
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
          if (rayTracerInstance->m_version != mesh->GetVersion() ||
              (skinnedMeshRenderer->RagDoll() &&
               !skinnedMeshRenderer->m_ragDollFreeze) ||
              skinnedMeshRenderer->m_animator.Get<Animator>()
                  ->AnimatedCurrentFrame())
            needVerticesUpdate = true;
          if (rayTracerInstance->m_surfaceColor != material->m_albedoColor ||
              rayTracerInstance->m_metallic !=
                  (material->m_metallic == 1.0f
                       ? -1.0f
                       : 1.0f / glm::pow(1.0f - material->m_metallic, 3.0f)) ||
              rayTracerInstance->m_roughness != material->m_roughness) {
            needMaterialUpdate = true;
          }
        }
      }
      rayTracerInstance->m_version = mesh->GetVersion();
      if (fromNew || needVerticesUpdate || needTransformUpdate ||
          needMaterialUpdate) {
        updateShaderBindingTable = true;
        rayTracerInstance->m_surfaceColor = material->m_albedoColor;
        rayTracerInstance->m_metallic =
            material->m_metallic == 1.0f
                ? -1.0f
                : 1.0f / glm::pow(1.0f - material->m_metallic, 3.0f);
        rayTracerInstance->m_roughness = material->m_roughness;
        rayTracerInstance->m_normalTexture = 0;
        rayTracerInstance->m_albedoTexture = 0;
        rayTracerInstance->m_entityId = entity.GetIndex();
        rayTracerInstance->m_entityVersion = entity.GetVersion();
      }
      if (material->m_albedoTexture.Get<Texture2D>() &&
          material->m_albedoTexture.Get<Texture2D>()->UnsafeGetGLTexture()) {
        if (material->m_albedoTexture.Get<Texture2D>()
                ->UnsafeGetGLTexture()
                ->Id() != rayTracerInstance->m_albedoTexture) {
          updateShaderBindingTable = true;
          rayTracerInstance->m_albedoTexture =
              material->m_albedoTexture.Get<Texture2D>()
                  ->UnsafeGetGLTexture()
                  ->Id();
        }
      } else if (rayTracerInstance->m_albedoTexture != 0) {
        updateShaderBindingTable = true;
        rayTracerInstance->m_albedoTexture = 0;
      }

      if (material->m_normalTexture.Get<Texture2D>() &&
          material->m_normalTexture.Get<Texture2D>()->UnsafeGetGLTexture()) {
        if (material->m_normalTexture.Get<Texture2D>()
                ->UnsafeGetGLTexture()
                ->Id() != rayTracerInstance->m_normalTexture) {
          updateShaderBindingTable = true;
          rayTracerInstance->m_normalTexture =
              material->m_normalTexture.Get<Texture2D>()
                  ->UnsafeGetGLTexture()
                  ->Id();
        }
      } else if (rayTracerInstance->m_normalTexture != 0) {
        updateShaderBindingTable = true;
        rayTracerInstance->m_normalTexture = 0;
      }
      if (rayTracerInstance->m_diffuseIntensity != material->m_emission) {
        updateShaderBindingTable = true;
        rayTracerInstance->m_diffuseIntensity = material->m_emission;
      }
      if (fromNew || needVerticesUpdate) {
        rebuildAccelerationStructure = true;
        rayTracerInstance->m_verticesUpdateFlag = true;
        if (fromNew) {
          rayTracerInstance->m_transformUpdateFlag = true;
          rayTracerInstance->m_globalTransform = globalTransform;
        }
        rayTracerInstance->m_skinnedVertices =
            reinterpret_cast<std::vector<SkinnedVertex> *>(
                &mesh->UnsafeGetSkinnedVertices());
        rayTracerInstance->m_boneMatrices =
            reinterpret_cast<std::vector<glm::mat4> *>(
                &skinnedMeshRenderer->m_finalResults.get()->m_value);
        rayTracerInstance->m_triangles = &mesh->UnsafeGetTriangles();
      } else if (needTransformUpdate) {
        rebuildAccelerationStructure = true;
        rayTracerInstance->m_globalTransform = globalTransform;
        rayTracerInstance->m_transformUpdateFlag = true;
      }
      if (fromNew)
        meshesStorage.push_back(newRayTracerInstance);
    }
  }
  for (int i = 0; i < meshesStorage.size(); i++) {
    if (meshesStorage[i].m_removeTag) {
      meshesStorage.erase(meshesStorage.begin() + i);
      i--;
      rebuildAccelerationStructure = true;
    }
  }
}

void RayTracerManager::UpdateScene() const {
  bool rebuildAccelerationStructure = false;
  bool updateShaderBindingTable = false;
  auto &meshesStorage = CudaModule::GetRayTracer()->m_instances;
  auto &skinnedMeshesStorage = CudaModule::GetRayTracer()->m_skinnedInstances;
  UpdateMeshesStorage(meshesStorage, rebuildAccelerationStructure,
                      updateShaderBindingTable);
  UpdateSkinnedMeshesStorage(skinnedMeshesStorage, rebuildAccelerationStructure,
                             updateShaderBindingTable);

  if (rebuildAccelerationStructure &&
      (!meshesStorage.empty() || !skinnedMeshesStorage.empty())) {
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
  manager.m_defaultWindow.Init("Ray Tracer");
  Application::RegisterUpdateFunction([]() {
    Update();

    OnGui();
  });

  SunlightCalculator::GetInstance().m_database.insert({4, {0, 90}});
  SunlightCalculator::GetInstance().m_database.insert({5, {7.12, 87.78}});
  SunlightCalculator::GetInstance().m_database.insert({6, {79, 77.19}});
  SunlightCalculator::GetInstance().m_database.insert({7, {174.7, 65.33}});
  SunlightCalculator::GetInstance().m_database.insert({8, {267.2, 53.15}});
  SunlightCalculator::GetInstance().m_database.insert({9, {348.05, 40.86}});
  SunlightCalculator::GetInstance().m_database.insert({10, {410.4, 28.77}});
  SunlightCalculator::GetInstance().m_database.insert({11, {449.95, 17.81}});
  SunlightCalculator::GetInstance().m_database.insert({12, {463.62, 12.08}});
  SunlightCalculator::GetInstance().m_database.insert({13, {450.48, 17.8}});
  SunlightCalculator::GetInstance().m_database.insert({14, {411.55, 28.77}});
  SunlightCalculator::GetInstance().m_database.insert({15, {349.6, 40.85}});
  SunlightCalculator::GetInstance().m_database.insert({16, {269.02, 53.14}});
  SunlightCalculator::GetInstance().m_database.insert({17, {175.98, 65.33}});
  SunlightCalculator::GetInstance().m_database.insert({18, {79.4, 77.2}});
  SunlightCalculator::GetInstance().m_database.insert({19, {7.15, 87.79}});
  SunlightCalculator::GetInstance().m_database.insert({20, {0, 90}});
  SunlightCalculator::GetInstance().m_intensityFactor = 0.002f;
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
    auto environmentalMap = manager.m_environmentalMap.Get<Cubemap>();
    if (environmentalMap) {
      manager.m_defaultRenderingProperties.m_environmentalMapId =
          environmentalMap->Texture()->Id();
    }
    manager.m_defaultRenderingProperties.m_frameSize = size;
    manager.m_defaultRenderingProperties.m_outputTextureId =
        manager.m_defaultWindow.m_output->Id();
    if (!CudaModule::GetRayTracer()->m_instances.empty() ||
        !CudaModule::GetRayTracer()->m_skinnedInstances.empty()) {
      manager.m_defaultWindow.m_rendered =
          CudaModule::GetRayTracer()->RenderDefault(
              manager.m_defaultRenderingProperties);
    }
  }
}

void RayTracerManager::OnGui() {
  auto &manager = GetInstance();
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("View")) {
      ImGui::Checkbox("Ray Tracer Manager", &manager.m_enableMenus);
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
  ImGui::Begin("Ray Tracer Manager");
  {
    manager.m_defaultRenderingProperties.OnGui();
    if (manager.m_defaultRenderingProperties.m_environmentalLightingType ==
        EnvironmentalLightingType::CIE) {
      if (ImGui::TreeNodeEx("CIE Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        static bool manualControl = false;
        ImGui::Checkbox("Manual control", &manualControl);
        static glm::vec2 angles = glm::vec2(90, 0);
        if (manualControl) {
          if (ImGui::DragFloat2("Skylight Direction (X/Y axis)", &angles.x,
                                1.0f, 0.0f, 180.0f)) {
            manager.m_defaultRenderingProperties.m_sunDirection =
                glm::quat(glm::radians(glm::vec3(angles.x, angles.y, 0.0f))) *
                glm::vec3(0, 0, -1);
          }
          ImGui::DragFloat(
              "Zenith radiance",
              &manager.m_defaultRenderingProperties.m_skylightIntensity, 0.01f,
              0.0f, 1.0f);
        } else {
          static bool autoUpdate = true;
          ImGui::Checkbox("Auto update", &autoUpdate);
          static int hour = 12;
          static int minute = 0;
          static bool updated = false;
          static float zenithIntensityFactor = 1.0f;
          static bool useDayRange = true;
          ImGui::Checkbox("Use day range", &useDayRange);
          if (useDayRange) {
            static float dayRange = 0.5f;
            if (ImGui::DragFloat("Day range", &dayRange, 0.01f, 0.0f, 0.99f)) {
              dayRange = glm::clamp(dayRange, 0.0f, 0.99f);
              hour = (dayRange * 24.0f);
              minute = (glm::mod(dayRange, 1.0f / 24) * 60.0f);
              updated = true;
            }
          } else {
            if (ImGui::DragInt("Hour", &hour, 1, 0, 23)) {
              hour = glm::clamp(hour, 0, 23);
              updated = true;
            }
            if (ImGui::DragInt("Minute", &minute, 1, 0, 59)) {
              minute = glm::clamp(minute, 0, 59);
              updated = true;
            }
          }
          if (ImGui::DragFloat("Zenith radiance factor", &zenithIntensityFactor,
                               0.01f, 0.0f, 3.0f)) {
            zenithIntensityFactor =
                glm::clamp(zenithIntensityFactor, 0.0f, 3.0f);
            updated = true;
          }
          if (ImGui::Button("Update") || (autoUpdate && updated)) {
            updated = false;
            SunlightCalculator::CalculateSunlightAngle(hour, minute, angles.x);
            SunlightCalculator::CalculateSunlightIntensity(
                hour, minute,
                manager.m_defaultRenderingProperties.m_skylightIntensity);
            manager.m_defaultRenderingProperties.m_skylightIntensity *=
                zenithIntensityFactor;
            manager.m_defaultRenderingProperties.m_sunDirection =
                glm::quat(glm::radians(glm::vec3(angles.x, angles.y, 0.0f))) *
                glm::vec3(0, 0, -1);
          }
          ImGui::Text(
              ("Intensity: " +
               std::to_string(
                   manager.m_defaultRenderingProperties.m_skylightIntensity))
                  .c_str());
          ImGui::Text(("Angle: [" + std::to_string(angles.x)).c_str());
        }
        ImGui::TreePop();
      }
    }
    EditorManager::DragAndDropButton<Cubemap>(
        RayTracerManager::GetInstance().m_environmentalMap,
        "Environmental Map");

    if (ImGui::Button("Load all MLVQ Materials")) {
      std::vector<std::string> pathes;
      std::filesystem::path folder("../Resources/btfs");
      for (auto &entry : std::filesystem::directory_iterator(folder)) {
        pathes.push_back(entry.path().string());
      }
      CudaModule::GetRayTracer()->LoadBtfMaterials(pathes);
    }
  }
  ImGui::End();

  manager.m_defaultWindow.OnGui();
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

float LerpHelper(float x, float y, float t) { return x * (1.f - t) + y * t; }

void SunlightCalculator::CalculateSunlightIntensity(int hour, int minute,
                                                    float &intensity) {
  float actualHour = glm::clamp(hour, 0, 23);
  float actualMinute = glm::clamp(minute, 0, 59) / 60.0f;
  float combinedTime = actualHour + actualMinute;
  auto &sunlightCalculator = GetInstance();

  if (sunlightCalculator.m_database.empty()) {
    intensity = 1.0f;
    return;
  }

  if (combinedTime < sunlightCalculator.m_database.begin()->first) {
    intensity = sunlightCalculator.m_database.begin()->second.first;
  } else {
    float lastVal = sunlightCalculator.m_database.begin()->second.first;
    int index = 0;
    bool found = false;
    for (const auto &i : sunlightCalculator.m_database) {
      if (index != 0) {
        if (combinedTime < i.first) {
          intensity = LerpHelper(lastVal, i.second.first, actualMinute);
          found = true;
          break;
        }
      }
      lastVal = i.second.first;
      index++;
    }
    if (!found)
      intensity = lastVal;
  }
  intensity *= sunlightCalculator.m_intensityFactor;
}
void SunlightCalculator::CalculateSunlightAngle(int hour, int minute,
                                                float &angle) {
  float actualHour = glm::clamp(hour, 0, 23);
  float actualMinute = glm::clamp(minute, 0, 59) / 60.0f;
  float combinedTime = actualHour + actualMinute;
  auto &sunlightCalculator = GetInstance();

  if (sunlightCalculator.m_database.empty()) {
    angle = 90;
    return;
  }
  if (combinedTime < sunlightCalculator.m_database.begin()->first) {
    angle = sunlightCalculator.m_database.begin()->second.second;
  } else {
    float lastVal = sunlightCalculator.m_database.begin()->second.second;
    int index = 0;
    bool found = false;
    for (const auto &i : sunlightCalculator.m_database) {
      if (index != 0) {
        if (combinedTime < i.first) {
          angle = LerpHelper(lastVal, i.second.second, actualMinute);
          found = true;
          break;
        }
      }
      lastVal = i.second.second;
      index++;
    }

    if (!found) {
      angle = lastVal;
    }
  }
  angle = 90.0f - angle;
}
SunlightCalculator &SunlightCalculator::GetInstance() {
  static SunlightCalculator instance;
  return instance;
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
      if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Settings")) {

          ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
      }
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{5, 5});
      if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Settings")) {
          ImGui::DragFloat("Resolution multiplier", &m_resolutionMultiplier,
                           0.01f, 0.1f, 1.0f);
          ImGui::DragFloat("FOV",
                           &EditorManager::GetInstance().m_sceneCamera->m_fov,
                           1, 1, 120);
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