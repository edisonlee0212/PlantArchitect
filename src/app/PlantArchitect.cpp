// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#include <CUDAModule.hpp>
#include <EditorManager.hpp>
#include <Utilities.hpp>
#include <ProjectManager.hpp>
#include <PhysicsManager.hpp>
#include <PlantSystem.hpp>
#include <PostProcessing.hpp>
#include <RayTracerManager.hpp>
#include <TreeSystem.hpp>
#include <TreeLeaves.hpp>
#include <CubeVolume.hpp>
#include <RayTracedRenderer.hpp>
#include <RadialBoundingVolume.hpp>
#include <ClassRegistry.hpp>
#include <ObjectRotator.hpp>
using namespace PlantArchitect;
using namespace RayTracerFacility;

void EngineSetup(bool enableRayTracing);

int main() {
    ClassRegistry::RegisterDataComponent<TreeLeavesTag>("TreeLeavesTag");
    ClassRegistry::RegisterDataComponent<RbvTag>("RbvTag");
    ClassRegistry::RegisterDataComponent<PlantInfo>("PlantInfo");
    ClassRegistry::RegisterDataComponent<BranchCylinder>("BranchCylinder");
    ClassRegistry::RegisterDataComponent<BranchCylinderWidth>("BranchCylinderWidth");
    ClassRegistry::RegisterDataComponent<BranchPointer>("BranchPointer");
    ClassRegistry::RegisterDataComponent<Illumination>("Illumination");
    ClassRegistry::RegisterDataComponent<BranchColor>("BranchColor");
    ClassRegistry::RegisterDataComponent<InternodeInfo>("InternodeInfo");
    ClassRegistry::RegisterDataComponent<InternodeGrowth>("InternodeGrowth");
    ClassRegistry::RegisterDataComponent<InternodeStatistics>("InternodeStatistics");

    ClassRegistry::RegisterPrivateComponent<ObjectRotator>("ObjectRotator");
    ClassRegistry::RegisterPrivateComponent<TreeData>("TreeData");
    ClassRegistry::RegisterPrivateComponent<TreeLeaves>("TreeLeaves");
    ClassRegistry::RegisterPrivateComponent<RadialBoundingVolume>("RadialBoundingVolume");
    ClassRegistry::RegisterPrivateComponent<CubeVolume>("CubeVolume");

    ClassRegistry::RegisterPrivateComponent<InternodeData>("InternodeData");

    ClassRegistry::RegisterSystem<PlantSystem>("PlantSystem");
    ClassRegistry::RegisterSystem<TreeSystem>("TreeSystem");

    ClassRegistry::RegisterPrivateComponent<RayTracedRenderer>(
            "RayTracedRenderer");

    const bool enableRayTracing = true;
    EngineSetup(enableRayTracing);

    Application::Init();

#pragma region Engine Loop
    Application::Run();
#pragma endregion
    if (enableRayTracing)
        RayTracerManager::End();
    Application::End();
}

void EngineSetup(bool enableRayTracing) {
    ProjectManager::SetScenePostLoadActions([=](){
    #pragma region Engine Setup
#pragma region Global light settings
        RenderManager::GetInstance().m_lightSettings.m_ambientLight = 0.2f;
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
        /*
        const Entity lightEntity = EntityManager::CreateEntity("Light source");
        auto pointLight = lightEntity.GetOrSetPrivateComponent<PointLight>().lock();
        pointLight->m_diffuseBrightness = 6;
        pointLight->m_lightSize = 0.25f;
        pointLight->m_quadratic = 0.0001f;
        pointLight->m_linear = 0.01f;
        pointLight->m_lightSize = 0.08f;
        transform.SetPosition(glm::vec3(0, 30, 0));
        transform.SetEulerRotation(glm::radians(glm::vec3(0, 0, 0)));
        lightEntity.SetDataComponent(transform);
        */
        if (enableRayTracing)
            RayTracerManager::Init();

        auto plantSystem =
                EntityManager::GetOrCreateSystem<PlantSystem>(
                        EntityManager::GetCurrentScene(), SystemGroup::SimulationSystemGroup);
        auto treeSystem = EntityManager::GetOrCreateSystem<TreeSystem>(
                EntityManager::GetCurrentScene(), SystemGroup::SimulationSystemGroup + 0.1f);
    });
}
