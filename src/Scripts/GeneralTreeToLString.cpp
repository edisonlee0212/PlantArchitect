//
// Created by lllll on 9/1/2021.
//

#include "GeneralTreeToLString.hpp"
#include "EntityManager.hpp"
#include "InternodeSystem.hpp"
#include "AssetManager.hpp"
#include "LSystemBehaviour.hpp"
#include "IVolume.hpp"

using namespace Scripts;

void GeneralTreeToLString::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
    if (m_remainingInstanceAmount <= 0) {
        ProjectManager::ScanProjectFolder(true);
        m_remainingInstanceAmount = 0;
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        return;
    }

    pipeline.m_status = AutoTreeGenerationPipelineStatus::Growth;
    if (m_generalTreeBehaviour.expired()) {
        m_generalTreeBehaviour = EntityManager::GetSystem<InternodeSystem>()->GetInternodeBehaviour<GeneralTreeBehaviour>();
    }
    auto behaviour = m_generalTreeBehaviour.lock();
    m_currentGrowingTree = behaviour->NewPlant(m_parameters, Transform());
}

void GeneralTreeToLString::OnGrowth(AutoTreeGenerationPipeline &pipeline) {
    if (m_remainingInstanceAmount == 0) {
        ProjectManager::ScanProjectFolder(true);
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        return;
    }
    auto internodeSystem = EntityManager::GetSystem<InternodeSystem>();
    internodeSystem->Simulate(m_perTreeGrowthIteration);

    m_generalTreeBehaviour.lock()->GenerateSkinnedMeshes();
    pipeline.m_status = AutoTreeGenerationPipelineStatus::AfterGrowth;

    auto mainCamera = RenderManager::GetMainCamera().lock();
    auto mainCameraEntity = mainCamera->GetOwner();
    auto mainCameraTransform = mainCameraEntity.GetDataComponent<GlobalTransform>();
    mainCameraTransform.SetPosition(
            glm::vec3(0.0f, m_perTreeGrowthIteration / 7.0, 5.0f + 5.0f * m_perTreeGrowthIteration / 7.0f));
    mainCameraEntity.SetDataComponent(mainCameraTransform);
    mainCamera->m_useClearColor = false;
    mainCamera->m_allowAutoResize = false;
    mainCamera->ResizeResolution(1024, 1024);
    EntityManager::GetCurrentScene()->m_environmentalMapSettings.m_environmentalLightingIntensity = 1.0f;
    m_imageCapturing = true;
}

void GeneralTreeToLString::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
    if (m_imageCapturing) {
        m_imageCapturing = false;
    } else {
        auto lString = AssetManager::CreateAsset<LString>();
        m_currentGrowingTree.GetOrSetPrivateComponent<Internode>().lock()->ExportLString(lString);
        auto lstringFolder = m_currentExportFolder / "L-System strings";
        auto imagesFolder = m_currentExportFolder / "Images";
        auto objFolder = m_currentExportFolder / "OBJ";
        std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / lstringFolder);
        std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / imagesFolder);
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
                            (m_parameterFileName + "_" +
                             std::to_string(m_generationAmount - m_remainingInstanceAmount) +
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
                            (m_parameterFileName + "_" +
                             std::to_string(m_generationAmount - m_remainingInstanceAmount) +
                             "_branch.obj"));
                    smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
                }
            }
        }
        //path here
        lString->SetPathAndSave(
                lstringFolder /
                (m_parameterFileName + "_" + std::to_string(m_generationAmount - m_remainingInstanceAmount) +
                 ".lstring"));

        auto mainCamera = RenderManager::GetMainCamera().lock();
        mainCamera->GetTexture()->SetPathAndSave(imagesFolder / (m_parameterFileName + "_" +
                                                                 std::to_string(m_generationAmount -
                                                                                m_remainingInstanceAmount) +
                                                                 ".jpg"));
        auto behaviour = m_generalTreeBehaviour.lock();
        behaviour->Recycle(m_currentGrowingTree);
        m_remainingInstanceAmount--;
        if (m_remainingInstanceAmount == 0) {
            ProjectManager::ScanProjectFolder(true);
            pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        } else {
            pipeline.m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
        }
    }
}

void GeneralTreeToLString::OnInspect() {
    FileUtils::OpenFile("Load parameters", "GeneralTreeParam", {".gtparams"}, [&](const std::filesystem::path &path) {
        m_parameters.Load(path);
        m_parameterFileName = path.stem().string();
        m_perTreeGrowthIteration = m_parameters.m_matureAge;
    }, false);
    FileUtils::SaveFile("Save parameters", "GeneralTreeParam", {".gtparams"}, [&](const std::filesystem::path &path) {
        m_parameters.Save(path);
    }, false);
    if(ImGui::TreeNodeEx("General tree parameters")) {
        m_parameters.OnInspect();
        ImGui::TreePop();
    }
    ImGui::Text("Pipeline Settings:");
    ImGui::DragInt("Generation Amount", &m_generationAmount);
    ImGui::DragInt("Growth iteration", &m_perTreeGrowthIteration);
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

void GeneralTreeToLString::OnIdle(AutoTreeGenerationPipeline &pipeline) {
    if (m_remainingInstanceAmount > 0) {
        pipeline.m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
        return;
    }
}

