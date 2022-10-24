// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>

#ifdef RAYTRACERFACILITY

#include <CUDAModule.hpp>
#include <RayTracerLayer.hpp>

#endif

#include "ProjectManager.hpp"
#include "PhysicsLayer.hpp"
#include "PostProcessing.hpp"
#include "ClassRegistry.hpp"
#include "ObjectRotator.hpp"
#include "InternodeModel/Internode.hpp"
#include "AutoTreeGenerationPipeline.hpp"
#include "TreeDataCapturePipeline.hpp"
#include "InternodeLayer.hpp"
#include "StrandLayer.hpp"
#include "PlantGrowth.hpp"

using namespace PlantArchitect;
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
using namespace Scripts;
using namespace Orchards;

void EngineSetup();


int main() {
    ClassRegistry::RegisterPrivateComponent<ObjectRotator>("ObjectRotator");
    ClassRegistry::RegisterPrivateComponent<AutoTreeGenerationPipeline>("AutoTreeGenerationPipeline");
    ClassRegistry::RegisterAsset<TreeDataCapturePipeline>("TreeDataCapturePipeline", {".tdcp"});
    for (int j = 0; j < 3000; j++) {
        TreeGrowthModel model;
        model.m_targetPlant = std::make_shared<Plant<BranchData, InternodeData, BudData>>();
        for (int i = 0; i < 10; i++) {
            model.Grow();
        }
        auto children = model.m_targetPlant->RefInternode(0).m_children;
        for (const auto &i: children) {
            model.m_targetPlant->PruneInternode(i);
        }
    }
    EngineSetup();

    ApplicationConfigs applicationConfigs;
    applicationConfigs.m_applicationName = "Plant Architect";
    Application::Create(applicationConfigs);
#ifdef RAYTRACERFACILITY
    Application::PushLayer<RayTracerLayer>();
#endif
    auto internodeLayer = Application::PushLayer<InternodeLayer>();
    auto strandLayer = Application::PushLayer<StrandLayer>();


#pragma region Engine Loop
    Application::Start();
#pragma endregion
    Application::End();
}

void EngineSetup() {
    ProjectManager::SetScenePostLoadActions([=]() {
        auto scene = Application::GetActiveScene();
#pragma region Engine Setup
        Transform transform;
        transform.SetEulerRotation(glm::radians(glm::vec3(150, 30, 0)));
#pragma region Preparations
        Application::Time().SetTimeStep(0.016f);
        transform = Transform();
        transform.SetPosition(glm::vec3(0, 2, 35));
        transform.SetEulerRotation(glm::radians(glm::vec3(15, 0, 0)));
        auto mainCamera = Application::GetActiveScene()->m_mainCamera.Get<UniEngine::Camera>();
        if (mainCamera) {
            auto postProcessing =
                    scene->GetOrSetPrivateComponent<PostProcessing>(mainCamera->GetOwner()).lock();
            auto ssao = postProcessing->GetLayer<SSAO>().lock();
            ssao->m_kernelRadius = 0.1;
            scene->SetDataComponent(mainCamera->GetOwner(), transform);
            mainCamera->m_useClearColor = true;
            mainCamera->m_clearColor = glm::vec3(0.5f);
        }
#pragma endregion
#pragma endregion

    });
}
