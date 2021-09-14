//
// Created by lllll on 9/1/2021.
//

#include "SpaceColonizationTreeToLString.hpp"
#include "EntityManager.hpp"
#include "InternodeSystem.hpp"
#include "AssetManager.hpp"
#include "LSystemBehaviour.hpp"
#include "IVolume.hpp"

using namespace Scripts;

void SpaceColonizationTreeToLString::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
    if (m_remainingInstanceAmount <= 0) {
        m_remainingInstanceAmount = 0;
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        return;
    }
    pipeline.m_status = AutoTreeGenerationPipelineStatus::Growth;
    if (m_spaceColonizationTreeBehaviour.expired()) {
        m_spaceColonizationTreeBehaviour = EntityManager::GetSystem<InternodeSystem>()->GetInternodeBehaviour<SpaceColonizationBehaviour>();
    }
    auto behaviour = m_spaceColonizationTreeBehaviour.lock();
    for (int i = 0; i < m_attractionPointAmount; i++) {
        behaviour->m_attractionPoints.push_back(behaviour->m_volumes[0].Get<IVolume>()->GetRandomPoint());
    }
    m_currentGrowingTree = behaviour->NewPlant(m_parameters, Transform());
}

void SpaceColonizationTreeToLString::OnGrowth(AutoTreeGenerationPipeline &pipeline) {
    if (m_remainingInstanceAmount == 0) {
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        return;
    }
    auto internodeSystem = EntityManager::GetSystem<InternodeSystem>();
    internodeSystem->Simulate(m_perTreeGrowthIteration);
    pipeline.m_status = AutoTreeGenerationPipelineStatus::AfterGrowth;
    m_spaceColonizationTreeBehaviour.lock()->GenerateSkinnedMeshes();

    auto mainCamera = RenderManager::GetMainCamera().lock();
    auto mainCameraEntity = mainCamera->GetOwner();
    auto mainCameraTransform = mainCameraEntity.GetDataComponent<GlobalTransform>();
    mainCameraTransform.SetPosition(
            glm::vec3(0.0f, 5.0f, 30.0f));
    mainCameraEntity.SetDataComponent(mainCameraTransform);
    mainCamera->m_useClearColor = false;
    mainCamera->m_allowAutoResize = false;
    mainCamera->ResizeResolution(1024, 1024);
    RenderManager::GetInstance().m_lightSettings.m_ambientLight = 1.0f;
    m_imageCapturing = true;
}

void SpaceColonizationTreeToLString::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
    if (m_imageCapturing) {
        m_imageCapturing = false;
    } else {
        auto lString = AssetManager::CreateAsset<LString>();
        m_currentGrowingTree.GetOrSetPrivateComponent<Internode>().lock()->ExportLString(lString);
        auto lstringFolder = m_currentExportFolder / "L-System strings";
        auto imagesFolder = m_currentExportFolder / "Images";
        std::filesystem::create_directories(lstringFolder);
        std::filesystem::create_directories(imagesFolder);
        //path here
        lString->Save(
                lstringFolder / (std::to_string(m_generationAmount - m_remainingInstanceAmount) + ".lstring"));
        auto mainCamera = RenderManager::GetMainCamera().lock();
        mainCamera->GetTexture()->Save(imagesFolder / (std::to_string(m_generationAmount - m_remainingInstanceAmount) +
                                                       ".jpg"));

        auto behaviour = m_spaceColonizationTreeBehaviour.lock();
        behaviour->Recycle(m_currentGrowingTree);
        behaviour->m_attractionPoints.clear();
        m_remainingInstanceAmount--;
        if (m_remainingInstanceAmount == 0) {
            pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        } else {
            pipeline.m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
        }
    }
}

void SpaceColonizationTreeToLString::OnInspect() {
    ImGui::Text("Space colonization parameters");
    m_parameters.OnInspect();
    ImGui::Text("Pipeline Settings:");
    ImGui::DragInt("Generation Amount", &m_generationAmount);
    ImGui::DragInt("Growth iteration", &m_perTreeGrowthIteration);
    ImGui::DragInt("Attraction point per plant", &m_attractionPointAmount);
    if (m_remainingInstanceAmount == 0) {
        if(Application::IsPlaying()) {
            if (ImGui::Button("Start")) {
                std::filesystem::create_directories(m_currentExportFolder);
                m_remainingInstanceAmount = m_generationAmount;
            }
        }else{
            ImGui::Text("Start Engine first!");
        }
    } else {
        ImGui::Text("Task dispatched...");
        ImGui::Text(("Total: " + std::to_string(m_generationAmount) + ", Remaining: " +
                     std::to_string(m_remainingInstanceAmount)).c_str());
        if(ImGui::Button("Force stop")){
            m_remainingInstanceAmount = 1;
        }
    }
}

void SpaceColonizationTreeToLString::OnIdle(AutoTreeGenerationPipeline &pipeline) {
    if (m_remainingInstanceAmount > 0) {
        pipeline.m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
        return;
    }
}

