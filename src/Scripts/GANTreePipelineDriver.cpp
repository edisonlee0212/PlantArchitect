//
// Created by lllll on 12/5/2021.
//

#include "GANTreePipelineDriver.hpp"
#include "AutoTreeGenerationPipeline.hpp"
#include "MultipleAngleCapture.hpp"

using namespace Scripts;

void GANTreePipelineDriver::OnInspect() {
    Editor::DragAndDropButton<AutoTreeGenerationPipeline>(m_pipeline, "DatasetPipeline");

    ImGui::DragInt("Amount per specie", &m_instancePerSpecie, 1, 0, 99999);
    if (m_parameterFilePaths.empty()) {
        if (ImGui::Button("Start")) {
            if (std::filesystem::exists(m_folderPath) && std::filesystem::is_directory(m_folderPath)) {
                for (const auto &entry: std::filesystem::directory_iterator(m_folderPath)) {
                    if (!std::filesystem::is_directory(entry.path())) {
                        if (entry.path().extension().string() == ".gtparams") {
                            m_parameterFilePaths.push_back(entry.path());
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
    if (m_parameterFilePaths.empty()) return;

    auto pipeline = m_pipeline.Get<AutoTreeGenerationPipeline>();
    if (!pipeline) return;
    if (pipeline->m_status != AutoTreeGenerationPipelineStatus::Idle) return;

    auto pipelineBehaviour = pipeline->m_pipelineBehaviour.Get<MultipleAngleCapture>();
    if (!pipelineBehaviour) return;
    if (pipelineBehaviour->Busy()) return;

    pipeline->UpdateInternodeBehaviour();
    auto generalTreeBehaviour = std::dynamic_pointer_cast<GeneralTreeBehaviour>(pipeline->GetBehaviour());
    if (!generalTreeBehaviour) return;

    auto path = std::filesystem::path(m_parameterFilePaths.back());
    pipeline->m_generalTreeParameters.Load(path);
    pipeline->m_parameterFileName = path.stem().string();
    pipelineBehaviour->m_perTreeGrowthIteration = pipeline->m_generalTreeParameters.m_matureAge;
    pipelineBehaviour->DisableAllExport();
    pipelineBehaviour->m_exportCSV = true;
    pipelineBehaviour->m_exportImage = false;
    pipelineBehaviour->m_generationAmount = m_instancePerSpecie;
    pipelineBehaviour->Start();

    m_parameterFilePaths.pop_back();

}

void GANTreePipelineDriver::Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) {
    m_pipeline.Relink(map, scene);
}
