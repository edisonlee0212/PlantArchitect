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
        m_spaceColonizationTreeBehaviour = EntityManager::GetSystem<InternodeSystem>(EntityManager::GetCurrentScene())->GetInternodeBehaviour<SpaceColonizationBehaviour>();
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
    auto internodeSystem = EntityManager::GetSystem<InternodeSystem>(EntityManager::GetCurrentScene());
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
    EntityManager::GetCurrentScene()->m_environmentalMapSettings.m_environmentalLightingIntensity = 1.0f;
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
        std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / lstringFolder);
        std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / imagesFolder);
        auto objFolder = m_currentExportFolder / "OBJ";
        if (m_exportOBJ) {
            std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / objFolder);
            Entity foliage, branch;
            m_currentGrowingTree.ForEachChild([&](Entity child) {
                if (child.GetName() == "Foliage") foliage = child;
                else if (child.GetName() == "Branch") branch = child;
            });
            if (foliage.IsValid() && foliage.HasPrivateComponent<SkinnedMeshRenderer>()) {
                auto smr = foliage.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
                if (smr->m_skinnedMesh.Get<SkinnedMesh>() && !smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
                    auto exportPath = std::filesystem::absolute(
                            ProjectManager::GetProjectPath().parent_path() / objFolder /
                            (std::to_string(m_generationAmount - m_remainingInstanceAmount) +
                             "_foliage.obj"));
                    UNIENGINE_LOG(exportPath.string());
                    smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
                }
            }
            if (branch.IsValid() && branch.HasPrivateComponent<SkinnedMeshRenderer>()) {
                auto smr = branch.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
                if (smr->m_skinnedMesh.Get<SkinnedMesh>() && !smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
                    auto exportPath = std::filesystem::absolute(
                            ProjectManager::GetProjectPath().parent_path() / objFolder /
                            (std::to_string(m_generationAmount - m_remainingInstanceAmount) +
                             "_branch.obj"));
                    smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
                }
            }
        }

        //path here
        lString->SetPathAndSave(
                lstringFolder / (std::to_string(m_generationAmount - m_remainingInstanceAmount) + ".lstring"));
        auto mainCamera = RenderManager::GetMainCamera().lock();
        mainCamera->GetTexture()->SetPathAndSave(
                imagesFolder / (std::to_string(m_generationAmount - m_remainingInstanceAmount) + ".jpg"));

        auto behaviour = m_spaceColonizationTreeBehaviour.lock();
        behaviour->Recycle(m_currentGrowingTree);
        behaviour->m_attractionPoints.clear();
        m_remainingInstanceAmount--;
        if (m_remainingInstanceAmount == 0) {
            ProjectManager::ScanProjectFolder(true);
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
    ImGui::Checkbox("Export OBJ", &m_exportOBJ);
    if (m_remainingInstanceAmount == 0) {
        if (Application::IsPlaying()) {
            if (ImGui::Button("Start")) {
                m_remainingInstanceAmount = m_generationAmount;
            }
        } else {
            ImGui::Text("Start Engine first!");
        }
    } else {
        ImGui::Text("Task dispatched...");
        ImGui::Text(("Total: " + std::to_string(m_generationAmount) + ", Remaining: " +
                     std::to_string(m_remainingInstanceAmount)).c_str());
        if (ImGui::Button("Force stop")) {
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

