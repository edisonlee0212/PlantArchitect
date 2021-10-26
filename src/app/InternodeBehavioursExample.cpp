// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#ifdef RAYTRACERFACILITY
#include <CUDAModule.hpp>
#include <RayTracerManager.hpp>
#include "MLVQRenderer.hpp"
#endif
#include <EditorManager.hpp>
#include <Utilities.hpp>
#include <ProjectManager.hpp>
#include <PhysicsManager.hpp>
#include <PostProcessing.hpp>

#include <CubeVolume.hpp>
#include <ClassRegistry.hpp>
#include <ObjectRotator.hpp>
#include "GeneralTreeBehaviour.hpp"
#include "DefaultInternodeResource.hpp"
#include "Internode.hpp"
#include <SpaceColonizationBehaviour.hpp>
#include "EmptyInternodeResource.hpp"
#include "LSystemBehaviour.hpp"
#include "AutoTreeGenerationPipeline.hpp"

#include "DefaultInternodePhyllotaxis.hpp"
#include "InternodeFoliage.hpp"
#include "RadialBoundingVolume.hpp"
#include "DepthCamera.hpp"
#include "MultipleAngleCapture.hpp"


#include "InternodeManager.hpp"
using namespace PlantArchitect;
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
using namespace Scripts;
void EngineSetup();

void RegisterDataComponentMenus();

int main() {

    ClassRegistry::RegisterDataComponent<BranchCylinder>("BranchCylinder");
    ClassRegistry::RegisterDataComponent<BranchCylinderWidth>("BranchCylinderWidth");
    ClassRegistry::RegisterDataComponent<BranchPointer>("BranchPointer");
    ClassRegistry::RegisterDataComponent<BranchColor>("BranchColor");

    ClassRegistry::RegisterPrivateComponent<ObjectRotator>("ObjectRotator");
    ClassRegistry::RegisterPrivateComponent<IVolume>("IVolume");
    ClassRegistry::RegisterPrivateComponent<CubeVolume>("CubeVolume");
    ClassRegistry::RegisterPrivateComponent<RadialBoundingVolume>("RadialBoundingVolume");

#ifdef RAYTRACERFACILITY
    ClassRegistry::RegisterPrivateComponent<MLVQRenderer>(
      "MLVQRenderer");
#endif
    ClassRegistry::RegisterPrivateComponent<DepthCamera>("DepthCamera");
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

    ClassRegistry::RegisterDataComponent<InternodeInfo>("InternodeInfo");

    ClassRegistry::RegisterPrivateComponent<AutoTreeGenerationPipeline>("AutoTreeGenerationPipeline");
    ClassRegistry::RegisterAsset<MultipleAngleCapture>("MultipleAngleCapture", ".mulanglecap");

    ClassRegistry::RegisterAsset<InternodeFoliage>("InternodeFoliage", ".internodefoliage");
    ClassRegistry::RegisterAsset<DefaultInternodePhyllotaxis>("DefaultInternodePhyllotaxis", ".defaultip");

    EngineSetup();
    RegisterDataComponentMenus();

    ApplicationConfigs applicationConfigs;
    applicationConfigs.m_projectPath = "InternodeBehavioursExample/InternodeBehavioursExample.ueproj";
    Application::Init(applicationConfigs);
#ifdef RAYTRACERFACILITY
    if (enableRayTracing)
      RayTracerManager::Init();
#endif
    InternodeManager::GetInstance().OnCreate();

    /*
         * Add all internode behaviours for example.
         */

    auto spaceColonizationBehaviour = InternodeManager::GetInternodeBehaviour<SpaceColonizationBehaviour>();
    Entity cubeVolumeEntity = EntityManager::CreateEntity(EntityManager::GetCurrentScene(), "CubeVolume");
    Transform cubeVolumeTransform = cubeVolumeEntity.GetDataComponent<Transform>();
    cubeVolumeTransform.SetPosition(glm::vec3(0, 12.5, 0));
    cubeVolumeEntity.SetDataComponent(cubeVolumeTransform);

    auto cubeVolume = cubeVolumeEntity.GetOrSetPrivateComponent<CubeVolume>().lock();
    cubeVolume->m_minMaxBound.m_min = glm::vec3(-10.0f);
    cubeVolume->m_minMaxBound.m_max = glm::vec3(10.0f);
    spaceColonizationBehaviour->PushVolume(std::dynamic_pointer_cast<IVolume>(cubeVolume));
    /*
     * Add all pipelines
     */
    auto multipleAngleCapturePipelineEntity = EntityManager::CreateEntity(EntityManager::GetCurrentScene(), "MultipleAngleCapturePipeline");
    auto multipleAngleCapturePipeline = multipleAngleCapturePipelineEntity.GetOrSetPrivateComponent<AutoTreeGenerationPipeline>().lock();
    auto multipleAngleCapture = AssetManager::CreateAsset<MultipleAngleCapture>();
    auto mainCamera = EntityManager::GetCurrentScene()->m_mainCamera.Get<UniEngine::Camera>();
    multipleAngleCapture->m_cameraEntity = mainCamera->GetOwner();
    multipleAngleCapturePipeline->m_pipelineBehaviour = multipleAngleCapture;
    multipleAngleCapture->m_volume = cubeVolume;

#pragma region Engine Loop
    Application::Run();
#pragma endregion
#ifdef RAYTRACERFACILITY
    if (enableRayTracing)
    RayTracerManager::End();
#endif
    Application::End();
}

void EngineSetup() {
    ProjectManager::SetScenePostLoadActions([=]() {
#pragma region Engine Setup
#pragma region Global light settings
        RenderManager::GetInstance().m_stableFit = false;
        RenderManager::GetInstance().m_maxShadowDistance = 100;
        RenderManager::SetSplitRatio(0.15f, 0.3f, 0.5f, 1.0f);
#pragma endregion
        Transform transform;
        transform.SetEulerRotation(glm::radians(glm::vec3(150, 30, 0)));
#pragma region Preparations
        Application::Time().SetTimeStep(0.016f);
        transform = Transform();
        transform.SetPosition(glm::vec3(0, 2, 35));
        transform.SetEulerRotation(glm::radians(glm::vec3(15, 0, 0)));
        auto mainCamera = EntityManager::GetCurrentScene()->m_mainCamera.Get<UniEngine::Camera>();
        if (mainCamera) {
            auto postProcessing =
                    mainCamera->GetOwner().GetOrSetPrivateComponent<PostProcessing>().lock();
            auto ssao = postProcessing->GetLayer<SSAO>().lock();
            ssao->m_kernelRadius = 0.1;
            mainCamera->GetOwner().SetDataComponent(transform);
            mainCamera->m_useClearColor = true;
            mainCamera->m_clearColor = glm::vec3(0.5f);
        }
#pragma endregion
#pragma endregion




    });

    Application::RegisterLateUpdateFunction([](){
        InternodeManager::GetInstance().OnInspect();
        InternodeManager::GetInstance().LateUpdate();
    });

}

void RegisterDataComponentMenus() {
    EditorManager::RegisterComponentDataInspector<InternodeInfo>([](Entity entity, IDataComponent *data, bool isRoot) {
        auto *ltw = reinterpret_cast<InternodeInfo *>(data);
        ltw->OnInspect();
    });

    EditorManager::RegisterComponentDataInspector<GeneralTreeParameters>([](Entity entity, IDataComponent *data, bool isRoot) {
        auto *ltw = reinterpret_cast<GeneralTreeParameters *>(data);
        ltw->OnInspect();
    });

    EditorManager::RegisterComponentDataInspector<InternodeStatus>([](Entity entity, IDataComponent *data, bool isRoot) {
        auto *ltw = reinterpret_cast<InternodeStatus *>(data);
        ltw->OnInspect();
    });

    EditorManager::RegisterComponentDataInspector<InternodeWaterPressure>([](Entity entity, IDataComponent *data, bool isRoot) {
        auto *ltw = reinterpret_cast<InternodeWaterPressure *>(data);
        ltw->OnInspect();
    });

    EditorManager::RegisterComponentDataInspector<InternodeWater>([](Entity entity, IDataComponent *data, bool isRoot) {
        auto *ltw = reinterpret_cast<InternodeWater *>(data);
        ltw->OnInspect();
    });

    EditorManager::RegisterComponentDataInspector<InternodeIllumination>([](Entity entity, IDataComponent *data, bool isRoot) {
        auto *ltw = reinterpret_cast<InternodeIllumination *>(data);
        ltw->OnInspect();
    });

    EditorManager::RegisterComponentDataInspector<SpaceColonizationParameters>(
            [](Entity entity, IDataComponent *data, bool isRoot) {
                auto *ltw = reinterpret_cast<SpaceColonizationParameters *>(data);
                ltw->OnInspect();
            });
}
