//
// Created by lllll on 12/5/2021.
//

#include "GANTreePipelineDriver.hpp"
#include "AutoTreeGenerationPipeline.hpp"
#include "MultipleAngleCapture.hpp"
using namespace Scripts;

void GANTreePipelineDriver::OnInspect() {
    EditorManager::DragAndDropButton<AutoTreeGenerationPipeline>(m_pipeline, "DatasetPipeline");

    ImGui::DragInt("Amount per specie", &m_instancePerSpecie, 1, 0, 99999);

    if(ImGui::Button("Start")){
        m_parameterFileNames.push_back(m_folderPath + "Acacia.gtparams");
        m_parameterFileNames.push_back(m_folderPath + "Birch.gtparams");
        m_parameterFileNames.push_back(m_folderPath + "Elm.gtparams");
        m_parameterFileNames.push_back(m_folderPath + "Maple.gtparams");
        m_parameterFileNames.push_back(m_folderPath + "Oak.gtparams");
        m_parameterFileNames.push_back(m_folderPath + "Pine.gtparams");
        m_parameterFileNames.push_back(m_folderPath + "Tulip.gtparams");
        m_parameterFileNames.push_back(m_folderPath + "Willow.gtparams");
    }
}

void GANTreePipelineDriver::LateUpdate() {
    if(m_parameterFileNames.empty()) return;

    auto pipeline = m_pipeline.Get<AutoTreeGenerationPipeline>();
    if(!pipeline) return;
    if(pipeline->m_status != AutoTreeGenerationPipelineStatus::Idle) return;

    auto pipelineBehaviour = pipeline->m_pipelineBehaviour.Get<MultipleAngleCapture>();
    if(!pipelineBehaviour) return;
    if(pipelineBehaviour->Busy()) return;

    pipeline->UpdateInternodeBehaviour();
    auto generalTreeBehaviour = std::dynamic_pointer_cast<GeneralTreeBehaviour>(pipeline->GetBehaviour());
    if(!generalTreeBehaviour) return;

    auto path = std::filesystem::path(m_parameterFileNames.back());
    pipeline->m_generalTreeParameters.Load(path);
    pipeline->m_parameterFileName = path.stem().string();
    pipelineBehaviour->m_perTreeGrowthIteration = pipeline->m_generalTreeParameters.m_matureAge;
    pipelineBehaviour->DisableAllExport();
    pipelineBehaviour->m_exportCSV = true;

    pipelineBehaviour->m_generationAmount = m_instancePerSpecie;
    pipelineBehaviour->Start();
    m_parameterFileNames.pop_back();
}

void GANTreePipelineDriver::Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) {
    m_pipeline.Relink(map, scene);
}
