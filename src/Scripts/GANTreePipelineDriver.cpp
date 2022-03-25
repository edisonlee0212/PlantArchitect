//
// Created by lllll on 12/5/2021.
//

#include "GANTreePipelineDriver.hpp"
#include "AutoTreeGenerationPipeline.hpp"
#include "MultipleAngleCapture.hpp"

using namespace Scripts;

void GANTreePipelineDriver::OnInspect() {
    Editor::DragAndDropButton<AutoTreeGenerationPipeline>(m_pipeline, "DatasetPipeline");
    FileUtils::OpenFolder("Select folder...", [&](const std::filesystem::path& path){
       m_folderPath = path;
    });
    ImGui::Text(("Folder path: " + m_folderPath.string()).c_str());
    ImGui::DragInt("Amount per specie", &m_instancePerSpecie, 1, 0, 99999);
    if (m_descriptors.empty()) {
        if (ImGui::Button("Start")) {
            if (std::filesystem::exists(m_folderPath) && std::filesystem::is_directory(m_folderPath)) {
                for (const auto &entry: std::filesystem::directory_iterator(m_folderPath)) {
                    if (!std::filesystem::is_directory(entry.path())) {
                        if (entry.path().extension() == ".gtparams") {
                            auto descriptor = AssetManager::CreateAsset<GeneralTreeParameters>();
                            descriptor->SetPathAndLoad(entry.path());
                            m_descriptors.emplace_back(descriptor);
                            break;
                        }
                    }

                }

            }
        }
    } else {
        ImGui::Text("Busy...");
    }
}

void GANTreePipelineDriver::LateUpdate() {
    if (m_descriptors.empty()) return;

    auto pipeline = m_pipeline.Get<AutoTreeGenerationPipeline>();
    if (!pipeline) return;
    if (pipeline->m_status != AutoTreeGenerationPipelineStatus::Idle) return;

    auto pipelineBehaviour = pipeline->m_pipelineBehaviour.Get<MultipleAngleCapture>();
    if (!pipelineBehaviour) return;
    if (pipelineBehaviour->Busy()) return;

    pipeline->UpdateInternodeBehaviour();
    auto generalTreeBehaviour = std::dynamic_pointer_cast<GeneralTreeBehaviour>(pipeline->GetBehaviour());
    if (!generalTreeBehaviour) return;

    auto descriptor = m_descriptors.back().Get<GeneralTreeParameters>();
    pipeline->m_plantDescriptor = descriptor;
    pipelineBehaviour->m_perTreeGrowthIteration = descriptor->m_matureAge;
    pipelineBehaviour->m_generationAmount = m_instancePerSpecie;
    pipelineBehaviour->Start();
    m_descriptors.pop_back();

}

void GANTreePipelineDriver::Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) {
    m_pipeline.Relink(map, scene);
}
