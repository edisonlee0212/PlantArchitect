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
#include <InternodeSystem.hpp>
#include <SpaceColonizationBehaviour.hpp>
#include "EmptyInternodeResource.hpp"
#include "LSystemBehaviour.hpp"
#include "SpaceColonizationTreeToLString.hpp"
#include "AutoTreeGenerationPipeline.hpp"

#include "DefaultInternodePhyllotaxis.hpp"
#include "InternodeFoliage.hpp"
#include "RadialBoundingVolume.hpp"
#include "GeneralTreeToLString.hpp"
#include "DepthCamera.hpp"
#include "MultipleAngleCapture.hpp"
using namespace PlantArchitect;
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
using namespace Scripts;
void EngineSetup(bool enableRayTracing);

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
    ClassRegistry::RegisterSystem<InternodeSystem>("InternodeSystem");

    ClassRegistry::RegisterPrivateComponent<AutoTreeGenerationPipeline>("AutoTreeGenerationPipeline");
    ClassRegistry::RegisterAsset<SpaceColonizationTreeToLString>("SpaceColonizationTreeToLString", ".sctolstring");
    ClassRegistry::RegisterAsset<GeneralTreeToLString>("GeneralTreeToLString", ".gttolstring");
    ClassRegistry::RegisterAsset<MultipleAngleCapture>("MultipleAngleCapture", ".mulanglecap");

    ClassRegistry::RegisterAsset<InternodeFoliage>("InternodeFoliage", ".internodefoliage");
    ClassRegistry::RegisterAsset<DefaultInternodePhyllotaxis>("DefaultInternodePhyllotaxis", ".defaultip");

    const bool enableRayTracing = true;
    EngineSetup(enableRayTracing);
    RegisterDataComponentMenus();

    ApplicationConfigs applicationConfigs;
    applicationConfigs.m_projectPath = "InternodeBehavioursExample/InternodeBehavioursExample.ueproj";
    Application::Init(applicationConfigs);

#pragma region Engine Loop
    Application::Run();
#pragma endregion
#ifdef RAYTRACERFACILITY
    if (enableRayTracing)
    RayTracerManager::End();
#endif
    Application::End();
}

void EngineSetup(bool enableRayTracing) {
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
        auto mainCamera = RenderManager::GetMainCamera().lock();
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

#ifdef RAYTRACERFACILITY
        if (enableRayTracing)
      RayTracerManager::Init();
#endif
        /*
         * Add all internode behaviours for example.
         */
        auto internodeSystem = EntityManager::GetOrCreateSystem<InternodeSystem>(EntityManager::GetCurrentScene(), 0.0f);
        Entity cubeVolumeEntity = EntityManager::CreateEntity(EntityManager::GetCurrentScene(), "CubeVolume");
        Transform cubeVolumeTransform = cubeVolumeEntity.GetDataComponent<Transform>();
        cubeVolumeTransform.SetPosition(glm::vec3(0, 15, 0));
        cubeVolumeEntity.SetDataComponent(cubeVolumeTransform);
        auto spaceColonizationBehaviour = AssetManager::CreateAsset<SpaceColonizationBehaviour>();
        auto lSystemBehaviour = AssetManager::CreateAsset<LSystemBehaviour>();
        auto generalTreeBehaviour = AssetManager::CreateAsset<GeneralTreeBehaviour>();
        internodeSystem->PushInternodeBehaviour(
                std::dynamic_pointer_cast<IInternodeBehaviour>(spaceColonizationBehaviour));
        internodeSystem->PushInternodeBehaviour(std::dynamic_pointer_cast<IInternodeBehaviour>(lSystemBehaviour));
        internodeSystem->PushInternodeBehaviour(std::dynamic_pointer_cast<IInternodeBehaviour>(generalTreeBehaviour));
        auto cubeVolume = cubeVolumeEntity.GetOrSetPrivateComponent<CubeVolume>().lock();
        cubeVolume->m_minMaxBound.m_min = glm::vec3(-10.0f);
        cubeVolume->m_minMaxBound.m_max = glm::vec3(10.0f);
        spaceColonizationBehaviour->PushVolume(std::dynamic_pointer_cast<IVolume>(cubeVolume));


        /*
         * Add all pipelines
         */
        auto spaceColonizationPipelineEntity = EntityManager::CreateEntity(EntityManager::GetCurrentScene(), "SpaceColonizationTreeToLStringPipeline");
        auto spaceColonizationPipeline = spaceColonizationPipelineEntity.GetOrSetPrivateComponent<AutoTreeGenerationPipeline>().lock();
        spaceColonizationPipeline->m_pipelineBehaviour = AssetManager::CreateAsset<SpaceColonizationTreeToLString>();

        auto generalTreePipelineEntity = EntityManager::CreateEntity(EntityManager::GetCurrentScene(), "GeneralTreeToLStringPipeline");
        auto generalTreePipeline = generalTreePipelineEntity.GetOrSetPrivateComponent<AutoTreeGenerationPipeline>().lock();
        generalTreePipeline->m_pipelineBehaviour = AssetManager::CreateAsset<GeneralTreeToLString>();

        auto multipleAngleCapturePipelineEntity = EntityManager::CreateEntity(EntityManager::GetCurrentScene(), "MultipleAngleCapturePipeline");
        auto multipleAngleCapturePipeline = multipleAngleCapturePipelineEntity.GetOrSetPrivateComponent<AutoTreeGenerationPipeline>().lock();
        auto multipleAngleCapture = AssetManager::CreateAsset<MultipleAngleCapture>();
        multipleAngleCapture->m_cameraEntity = mainCamera->GetOwner();
        multipleAngleCapturePipeline->m_pipelineBehaviour = multipleAngleCapture;

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
