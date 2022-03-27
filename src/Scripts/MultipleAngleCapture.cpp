//
// Created by lllll on 9/24/2021.
//

#include "MultipleAngleCapture.hpp"
#include "DepthCamera.hpp"
#include "Entities.hpp"
#include "PlantLayer.hpp"
#include "AssetManager.hpp"
#include "LSystemBehaviour.hpp"
#include "IVolume.hpp"
#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"
#include "TransformLayer.hpp"

using namespace RayTracerFacility;
using namespace Scripts;

void MultipleAngleCapture::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
    Entity rootInternode;
    auto children = pipeline.m_currentGrowingTree.GetChildren();
    for (const auto &i: children) {
        if (i.HasPrivateComponent<Internode>()) rootInternode = i;
    }
    auto internode = rootInternode.GetOrSetPrivateComponent<Internode>().lock();
    if (m_applyPhyllotaxis) pipeline.m_currentGrowingTree.GetOrSetPrivateComponent<Root>().lock()->m_foliagePhyllotaxis = m_foliagePhyllotaxis;
    auto behaviour = pipeline.GetBehaviour();
    auto behaviourType = pipeline.GetBehaviourType();
    switch (behaviourType) {
        case BehaviourType::GeneralTree:
            break;
        case BehaviourType::LSystem:
            break;
        case BehaviourType::SpaceColonization: {
            auto behaviour = std::dynamic_pointer_cast<SpaceColonizationBehaviour>(pipeline.GetBehaviour());
            for (int i = 0; i < 6000; i++) {
                behaviour->m_attractionPoints.push_back(m_volume.Get<IVolume>()->GetRandomPoint());
            }
        }
            break;
    }
    if (m_exportImage || m_exportDepth || m_exportBranchCapture) {
        SetUpCamera(pipeline);
    }
    pipeline.m_status = AutoTreeGenerationPipelineStatus::Growth;
}

void MultipleAngleCapture::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
    auto behaviour = pipeline.GetBehaviour();
    auto camera = pipeline.GetOwner().GetOrSetPrivateComponent<RayTracerCamera>().lock();
    auto internodeLayer = Application::GetLayer<PlantLayer>();
    auto behaviourType = pipeline.GetBehaviourType();
    Entity rootInternode;
    auto children = pipeline.m_currentGrowingTree.GetChildren();
    for (const auto &i: children) {
        if (i.HasPrivateComponent<Internode>()) rootInternode = i;
    }
    auto imagesFolder = m_currentExportFolder / "Image";
    auto objFolder = m_currentExportFolder / "Mesh";
    auto depthFolder = m_currentExportFolder / "Depth";
    auto branchFolder = m_currentExportFolder / "Branch";
    auto graphFolder = m_currentExportFolder / "Graph";
    auto csvFolder = m_currentExportFolder / "CSV";
    auto lStringFolder = m_currentExportFolder / "LSystemString";
    if (m_exportOBJ || m_exportImage || m_exportDepth || m_exportBranchCapture) {
        behaviour->GenerateSkinnedMeshes();
        internodeLayer->UpdateInternodeColors();
    }
    Bound plantBound;
    if (m_exportImage || m_exportDepth || m_exportBranchCapture) {
        pipeline.m_currentGrowingTree.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (!behaviour->InternodeCheck(child)) return;
            plantBound = child.GetOrSetPrivateComponent<Internode>().lock()->CalculateChildrenBound();
        });
    }
#pragma region Export
    if (pipeline.m_behaviourType == BehaviourType::GeneralTree && m_exportGraph) {
        std::filesystem::create_directories(
                ProjectManager::GetProjectPath().parent_path().parent_path() / graphFolder);
        auto exportPath = std::filesystem::absolute(
                ProjectManager::GetProjectPath().parent_path().parent_path() / graphFolder /
                (pipeline.m_prefix + ".yml"));
        ExportGraph(pipeline, behaviour, exportPath);
    }
    if (pipeline.m_behaviourType == BehaviourType::GeneralTree && m_exportCSV) {
        std::filesystem::create_directories(
                ProjectManager::GetProjectPath().parent_path().parent_path() / csvFolder);

        std::filesystem::create_directories(
                ProjectManager::GetProjectPath().parent_path().parent_path() / csvFolder /
                pipeline.m_currentUsingDescriptor.Get<IPlantDescriptor>()->GetName());
        auto exportPath = std::filesystem::absolute(
                ProjectManager::GetProjectPath().parent_path().parent_path() / csvFolder /
                pipeline.m_currentUsingDescriptor.Get<IPlantDescriptor>()->GetName() /
                (pipeline.m_prefix + ".csv"));
        ExportCSV(pipeline, behaviour, exportPath);
    }
    if (m_exportLString) {
        std::filesystem::create_directories(
                ProjectManager::GetProjectPath().parent_path().parent_path() / lStringFolder);
        auto lString = AssetManager::CreateAsset<LSystemString>();

        rootInternode.GetOrSetPrivateComponent<Internode>().lock()->ExportLString(lString);
        //path here
        lString->SetPathAndSave(
                lStringFolder /
                (std::to_string(pipeline.m_generationAmount - pipeline.m_remainingInstanceAmount) +
                 ".lstring"));
    }
    if (m_exportMatrices) {
        for (float turnAngle = m_turnAngleStart; turnAngle < m_turnAngleEnd; turnAngle += m_turnAngleStep) {
            for (float pitchAngle = m_pitchAngleStart; pitchAngle < m_pitchAngleEnd; pitchAngle += m_pitchAngleStep) {
                auto anglePrefix = std::to_string(pitchAngle) + "_" +
                                   std::to_string(turnAngle);
                auto cameraGlobalTransform = TransformCamera(plantBound, turnAngle, pitchAngle);
                m_cameraModels.push_back(cameraGlobalTransform.m_value);
                m_treeModels.push_back(pipeline.m_currentGrowingTree.GetDataComponent<GlobalTransform>().m_value);
                m_projections.push_back(Camera::m_cameraInfoBlock.m_projection);
                m_views.push_back(Camera::m_cameraInfoBlock.m_view);
                m_names.push_back(pipeline.m_prefix + "_" + anglePrefix);
            }
        }
    }
    if (m_exportOBJ) {
        std::filesystem::create_directories(
                ProjectManager::GetProjectPath().parent_path().parent_path() / objFolder);
        Entity foliage, branch;
        pipeline.m_currentGrowingTree.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (child.GetName() == "Foliage") foliage = child;
            else if (child.GetName() == "Branch") branch = child;
        });
        if (foliage.IsValid() && foliage.HasPrivateComponent<SkinnedMeshRenderer>()) {
            auto smr = foliage.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
            if (smr->m_skinnedMesh.Get<SkinnedMesh>() &&
                !smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
                auto exportPath = std::filesystem::absolute(
                        ProjectManager::GetProjectPath().parent_path().parent_path() / objFolder /
                        (pipeline.m_prefix + "_foliage.obj"));
                UNIENGINE_LOG(exportPath.string());
                smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
            }
        }
        if (branch.IsValid() && branch.HasPrivateComponent<SkinnedMeshRenderer>()) {
            auto smr = branch.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
            if (smr->m_skinnedMesh.Get<SkinnedMesh>() &&
                !smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
                auto exportPath = std::filesystem::absolute(
                        ProjectManager::GetProjectPath().parent_path().parent_path() / objFolder /
                        (pipeline.m_prefix + "_branch.obj"));
                smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
            }
        }
    }
    if (m_exportImage) {
        std::filesystem::create_directories(
                ProjectManager::GetProjectPath().parent_path().parent_path() / imagesFolder);
        auto cameraEntity = pipeline.GetOwner();
        auto rayTracerCamera = cameraEntity.GetOrSetPrivateComponent<RayTracerCamera>().lock();
        assert(cameraEntity.IsValid());
        Application::GetLayer<RayTracerLayer>()->UpdateScene();
        for (int turnAngle = m_turnAngleStart; turnAngle < m_turnAngleEnd; turnAngle += m_turnAngleStep) {
            for (int pitchAngle = m_pitchAngleStart; pitchAngle < m_pitchAngleEnd; pitchAngle += m_pitchAngleStep) {
                auto anglePrefix = std::to_string(pitchAngle) + "_" +
                                   std::to_string(turnAngle);
                auto cameraGlobalTransform = TransformCamera(plantBound, turnAngle, pitchAngle);

                cameraEntity.SetDataComponent(cameraGlobalTransform);
                Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(
                        Entities::GetCurrentScene());

                rayTracerCamera->Render(m_rayProperties);
                rayTracerCamera->m_colorTexture->Export(
                        ProjectManager::GetProjectPath().parent_path().parent_path() /
                        imagesFolder / (pipeline.m_prefix + "_" + anglePrefix + "_rgb.png"));
            }
        }
    }
    if (m_exportDepth) {
        std::filesystem::create_directories(
                ProjectManager::GetProjectPath().parent_path().parent_path() / depthFolder);
        auto cameraEntity = pipeline.GetOwner();
        for (int turnAngle = m_turnAngleStart; turnAngle < m_turnAngleEnd; turnAngle += m_turnAngleStep) {
            for (int pitchAngle = m_pitchAngleStart;
                 pitchAngle < m_pitchAngleEnd; pitchAngle += m_pitchAngleStep) {
                auto anglePrefix = std::to_string(pitchAngle) + "_" +
                                   std::to_string(turnAngle);
                auto cameraGlobalTransform = TransformCamera(plantBound, turnAngle, pitchAngle);
                cameraEntity.SetDataComponent(cameraGlobalTransform);
                Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(
                        Entities::GetCurrentScene());
                auto depthCamera = pipeline.GetOwner().GetOrSetPrivateComponent<DepthCamera>().lock();
                depthCamera->Render();
                depthCamera->m_colorTexture->SetPathAndSave(
                        depthFolder / (pipeline.m_prefix + "_" + anglePrefix + "_depth.png"));
            }
        }
    }
    if (m_exportBranchCapture) {
        auto cameraEntity = pipeline.GetOwner();
        auto rayTracerCamera = cameraEntity.GetOrSetPrivateComponent<RayTracerCamera>().lock();
        assert(cameraEntity.IsValid());
        for (int turnAngle = m_turnAngleStart; turnAngle < m_turnAngleEnd; turnAngle += m_turnAngleStep) {
            for (int pitchAngle = m_pitchAngleStart;
                 pitchAngle < m_pitchAngleEnd; pitchAngle += m_pitchAngleStep) {
                auto anglePrefix = std::to_string(pitchAngle) + "_" +
                                   std::to_string(turnAngle);
                auto cameraGlobalTransform = TransformCamera(plantBound, turnAngle, pitchAngle);
                cameraEntity.SetDataComponent(cameraGlobalTransform);
                Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(
                        Entities::GetCurrentScene());

                rayTracerCamera->Render(m_rayProperties);
                rayTracerCamera->m_colorTexture->Export(
                        ProjectManager::GetProjectPath().parent_path().parent_path() /
                        branchFolder / (pipeline.m_prefix + "_" + anglePrefix + "_branch.png"));
            }
        }
    }
#pragma endregion
    pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
}

void MultipleAngleCapture::OnInspect() {
    if (ImGui::Button("Instantiate Pipeline")) {
        auto multipleAngleCapturePipelineEntity = Entities::CreateEntity(Entities::GetCurrentScene(),
                                                                         "GANTree Dataset Pipeline");
        auto multipleAngleCapturePipeline = multipleAngleCapturePipelineEntity.GetOrSetPrivateComponent<AutoTreeGenerationPipeline>().lock();
        multipleAngleCapturePipeline->m_pipelineBehaviour = AssetManager::Get<MultipleAngleCapture>(GetHandle());
    }
    Editor::DragAndDropButton(m_volume, "Volume", {"CubeVolume", "RadialBoundingVolume"}, false);
    if (ImGui::TreeNodeEx("Pipeline Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat("Branch width", &m_branchWidth, 0.01f);

        ImGui::Checkbox("Override phyllotaxis", &m_applyPhyllotaxis);
        if (m_applyPhyllotaxis)
            Editor::DragAndDropButton(m_foliagePhyllotaxis, "Phyllotaxis",
                                      {"EmptyInternodePhyllotaxis", "DefaultInternodePhyllotaxis"}, true);
        ImGui::Text("Data export:");
        ImGui::Checkbox("Export OBJ", &m_exportOBJ);
        ImGui::Checkbox("Export Graph", &m_exportGraph);
        ImGui::Checkbox("Export CSV", &m_exportCSV);
        ImGui::Checkbox("Export LSystemString", &m_exportLString);

        ImGui::Text("Rendering export:");
        ImGui::Checkbox("Export Depth", &m_exportDepth);
        ImGui::Checkbox("Export Image", &m_exportImage);
        ImGui::Checkbox("Export Branch Capture", &m_exportBranchCapture);

        ImGui::TreePop();
    }
    if (m_exportDepth || m_exportImage || m_exportBranchCapture) {
        if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Export Camera matrices", &m_exportMatrices);
            ImGui::Checkbox("Auto adjust camera", &m_autoAdjustCamera);
            if (!m_autoAdjustCamera) {
                ImGui::Text("Position:");
                ImGui::DragFloat3("Focus point", &m_focusPoint.x, 0.1f);
                ImGui::DragFloat("Distance to focus point", &m_distance, 0.1);
            }
            ImGui::Text("Rotation:");
            ImGui::DragInt3("Pitch Angle Start/Step/End", &m_pitchAngleStart, 1);
            ImGui::DragInt3("Turn Angle Start/Step/End", &m_turnAngleStart, 1);

            ImGui::Text("Camera Settings:");
            ImGui::DragFloat("Camera FOV", &m_fov);
            ImGui::DragInt2("Camera Resolution", &m_resolution.x);
            ImGui::DragFloat2("Camera near/far", &m_cameraMin);
            ImGui::Checkbox("Use clear color", &m_useClearColor);
            ImGui::ColorEdit3("Camera Clear Color", &m_backgroundColor.x);
            ImGui::DragFloat("Light Size", &m_lightSize, 0.001f);
            ImGui::DragFloat("Ambient light intensity", &m_ambientLightIntensity, 0.01f);
            ImGui::TreePop();
        }
        if (m_exportBranchCapture) Application::GetLayer<PlantLayer>()->DrawColorModeSelectionMenu();
    }
}

void MultipleAngleCapture::SetUpCamera(AutoTreeGenerationPipeline &pipeline) {
    auto cameraEntity = pipeline.GetOwner();
    assert(cameraEntity.IsValid());

    auto camera = cameraEntity.GetOrSetPrivateComponent<RayTracerCamera>().lock();
    camera->SetFov(m_fov);
    camera->m_allowAutoResize = false;
    camera->m_frameSize = m_resolution;

    auto depthCamera = cameraEntity.GetOrSetPrivateComponent<DepthCamera>().lock();
    depthCamera->m_resX = m_resolution.x;
    depthCamera->m_resY = m_resolution.y;

    if (cameraEntity.HasPrivateComponent<PostProcessing>()) {
        auto postProcessing = cameraEntity.GetOrSetPrivateComponent<PostProcessing>().lock();
        postProcessing->SetEnabled(false);
    }
}

void MultipleAngleCapture::OnCreate() {
}

void MultipleAngleCapture::ExportGraph(AutoTreeGenerationPipeline &pipeline,
                                       const std::shared_ptr<IPlantBehaviour> &behaviour,
                                       const std::filesystem::path &path) {
    try {
        auto directory = path;
        directory.remove_filename();
        std::filesystem::create_directories(directory);
        YAML::Emitter out;
        out << YAML::BeginMap;
        std::vector<std::vector<std::pair<int, Entity>>> internodes;
        internodes.resize(128);
        internodes[0].emplace_back(-1, pipeline.m_currentGrowingTree);
        pipeline.m_currentGrowingTree.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (!behaviour->InternodeCheck(child)) return;
            behaviour->InternodeGraphWalkerRootToEnd(child,
                                                     [&](Entity parent, Entity child) {
                                                         auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                                         internodes[childInternodeInfo.m_layer].emplace_back(
                                                                 parent.GetIndex(),
                                                                 child);
                                                     });
        });

        out << YAML::Key << "Layers" << YAML::Value << YAML::BeginSeq;
        int layerIndex = 0;
        for (const auto &layer: internodes) {
            if (layer.empty()) break;
            out << YAML::BeginMap;
            out << YAML::Key << "Layer Index" << YAML::Value << layerIndex;
            out << YAML::Key << "Nodes" << YAML::Value << YAML::BeginSeq;
            for (const auto &instance: layer) {
                ExportGraphNode(behaviour, out, instance.first, instance.second);
            }
            out << YAML::EndSeq;
            out << YAML::EndMap;
            layerIndex++;
        }


        out << YAML::EndSeq;
        out << YAML::EndMap;
        std::ofstream fout(path.string());
        fout << out.c_str();
        fout.flush();
    }
    catch (std::exception e) {
        UNIENGINE_ERROR("Failed to save!");
    }

}

void MultipleAngleCapture::ExportGraphNode(const std::shared_ptr<IPlantBehaviour> &behaviour, YAML::Emitter &out,
                                           int parentIndex, const Entity &internode) {
    out << YAML::BeginMap;
    out << YAML::Key << "Parent Entity Index" << parentIndex;
    out << YAML::Key << "Entity Index" << internode.GetIndex();

    std::vector<int> indices = {-1, -1, -1};
    internode.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
        if (!behaviour->InternodeCheck(child)) return;
        indices[child.GetDataComponent<InternodeStatus>().m_branchingOrder] = child.GetIndex();
    });

    out << YAML::Key << "Children Entity Indices" << YAML::Key << YAML::BeginSeq;
    for (int i = 0; i < 3; i++) {
        out << YAML::BeginMap;
        out << YAML::Key << "Entity Index" << YAML::Value << indices[i];
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    auto globalTransform = internode.GetDataComponent<GlobalTransform>();
    auto transform = internode.GetDataComponent<Transform>();
    /*
    out << YAML::Key << "Transform"
        << internode.GetDataComponent<Transform>().m_value;
    out << YAML::Key << "GlobalTransform"
        << internode.GetDataComponent<GlobalTransform>().m_value;
        */
    auto position = globalTransform.GetPosition();
    auto globalRotation = globalTransform.GetRotation();
    auto front = globalRotation * glm::vec3(0, 0, -1);
    auto up = globalRotation * glm::vec3(0, 1, 0);
    auto internodeInfo = internode.GetDataComponent<InternodeInfo>();
    auto internodeStatus = internode.GetDataComponent<InternodeStatus>();
    out << YAML::Key << "Branching Order" << internodeStatus.m_branchingOrder;
    out << YAML::Key << "Level" << internodeStatus.m_level;
    out << YAML::Key << "Distance to Root" << internodeStatus.m_rootDistance;
    out << YAML::Key << "Local Rotation" << transform.GetRotation();
    out << YAML::Key << "Global Rotation" << globalRotation;
    out << YAML::Key << "Position" << position + front * internodeInfo.m_length;
    out << YAML::Key << "Front Direction" << front;
    out << YAML::Key << "Up Direction" << up;
    out << YAML::Key << "IsEndNode" << internodeInfo.m_endNode;
    out << YAML::Key << "Thickness" << internodeInfo.m_thickness;
    out << YAML::Key << "Length" << internodeInfo.m_length;
    //out << YAML::Key << "Internode Index" << internodeInfo.m_index;
    out << YAML::Key << "Internode Layer" << internodeInfo.m_layer;
    out << YAML::EndMap;
    out << YAML::EndMap;
}

void MultipleAngleCapture::ExportMatrices(const std::filesystem::path &path) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "Capture Info" << YAML::BeginSeq;
    for (int i = 0; i < m_projections.size(); i++) {
        out << YAML::BeginMap;
        out << YAML::Key << "File Prefix" << YAML::Value << m_names[i];
        out << YAML::Key << "Projection" << YAML::Value << m_projections[i];
        out << YAML::Key << "View" << YAML::Value << m_views[i];
        out << YAML::Key << "Camera Transform" << YAML::Value << m_cameraModels[i];
        out << YAML::Key << "Plant Transform" << YAML::Value << m_treeModels[i];
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
    out << YAML::EndMap;
    std::ofstream fout(path.string());
    fout << out.c_str();
    fout.flush();
}

void
MultipleAngleCapture::ExportCSV(AutoTreeGenerationPipeline &pipeline, const std::shared_ptr<IPlantBehaviour> &behaviour,
                                const std::filesystem::path &path) {
    std::ofstream ofs;
    ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (ofs.is_open()) {
        std::string output;
        std::vector<std::vector<std::pair<int, Entity>>> internodes;
        internodes.resize(128);
        pipeline.m_currentGrowingTree.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (!behaviour->InternodeCheck(child)) return;
            internodes[0].emplace_back(-1, child);
            behaviour->InternodeGraphWalkerRootToEnd(child,
                                                     [&](Entity parent, Entity child) {
                                                         auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                                         if (childInternodeInfo.m_endNode) return;
                                                         internodes[childInternodeInfo.m_layer].emplace_back(
                                                                 parent.GetIndex(),
                                                                 child);
                                                     });
        });
        output += "in_id,in_pos_x,in_pos_y,in_pos_z,in_front_x,in_front_y,in_front_z,in_up_x,in_up_y,in_up_z,in_thickness,in_length,in_root_distance,in_chain_distance,in_distance_to_branch_start,in_level,in_flush_age,in_proximity,in_node_count,in_quat_x,in_quat_y,in_quat_z,in_quat_w,";
        output += "out0_id,out0_pos_x,out0_pos_y,out0_pos_z,out0_front_x,out0_front_y,out0_front_z,out0_up_x,out0_up_y,out0_up_z,out0_thickness,out0_length,out0_root_distance,out0_chain_distance,out0_distance_to_branch_start,out0_level,out0_flush_age,out0_proximity,out0_node_count,out0_quat_x,out0_quat_y,out0_quat_z,out0_quat_w,";
        output += "out1_id,out1_pos_x,out1_pos_y,out1_pos_z,out1_front_x,out1_front_y,out1_front_z,out1_up_x,out1_up_y,out1_up_z,out1_thickness,out1_length,out1_root_distance,out1_chain_distance,out1_distance_to_branch_start,out1_level,out1_flush_age,out1_proximity,out1_node_count,out1_quat_x,out1_quat_y,out1_quat_z,out1_quat_w,";
        output += "out2_id,out2_pos_x,out2_pos_y,out2_pos_z,out2_front_x,out2_front_y,out2_front_z,out2_up_x,out2_up_y,out2_up_z,out2_thickness,out2_length,out2_root_distance,out2_chain_distance,out2_distance_to_branch_start,out2_level,out2_flush_age,out2_proximity,out2_node_count,out2_quat_x,out2_quat_y,out2_quat_z,out2_quat_w\n";
        int layerIndex = 0;
        for (const auto &layer: internodes) {
            if (layer.empty()) break;
            for (const auto &instance: layer) {
                auto internode = instance.second;
                std::vector<Entity> children;
                children.resize(3);
                bool hasChild = false;
                internode.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
                    if (!behaviour->InternodeCheck(child)) return;
                    if (child.GetDataComponent<InternodeInfo>().m_endNode) return;
                    children[child.GetDataComponent<InternodeStatus>().m_branchingOrder] = child;
                    hasChild = true;
                });
                std::string row;

                auto globalTransform = internode.GetDataComponent<GlobalTransform>();
                auto transform = internode.GetDataComponent<Transform>();

                auto position = globalTransform.GetPosition();
                auto globalRotation = globalTransform.GetRotation();
                auto front = globalRotation * glm::vec3(0, 0, -1);
                auto up = globalRotation * glm::vec3(0, 1, 0);
                auto rotation = transform.GetRotation();
                auto internodeInfo = internode.GetDataComponent<InternodeInfo>();
                auto internodeStatus = internode.GetDataComponent<InternodeStatus>();

                position += glm::normalize(front) * internodeInfo.m_length;

                row += std::to_string(internode.GetIndex()) + ",";

                row += std::to_string(position.x) + ",";
                row += std::to_string(position.y) + ",";
                row += std::to_string(position.z) + ",";

                row += std::to_string(front.x) + ",";
                row += std::to_string(front.y) + ",";
                row += std::to_string(front.z) + ",";

                row += std::to_string(up.x) + ",";
                row += std::to_string(up.y) + ",";
                row += std::to_string(up.z) + ",";

                row += std::to_string(internodeInfo.m_thickness) + ",";
                row += std::to_string(internodeInfo.m_length) + ",";
                row += std::to_string(internodeStatus.m_rootDistance) + ",";
                row += std::to_string(internodeStatus.m_chainDistance) + ",";
                row += std::to_string(internodeStatus.m_branchLength) + ",";
                row += std::to_string(internodeStatus.m_level) + ",";

                row += std::to_string(internodeStatus.m_age) + ",";
                //row += std::to_string(internodeStatus.m_recordedProbability) + ",";
                row += std::to_string(internodeStatus.m_startDensity) + ",";
                row += std::to_string(internodeStatus.m_currentTotalNodeCount) + ",";

                row += std::to_string(globalRotation.x) + ",";
                row += std::to_string(globalRotation.y) + ",";
                row += std::to_string(globalRotation.z) + ",";
                row += std::to_string(globalRotation.w) + ",";

                for (int i = 0; i < 3; i++) {
                    auto child = children[i];
                    if (child.IsNull() || child.GetDataComponent<InternodeInfo>().m_endNode) {
                        row += "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A";
                    } else {
                        auto globalTransformChild = child.GetDataComponent<GlobalTransform>();
                        auto transformChild = child.GetDataComponent<Transform>();

                        auto positionChild = globalTransformChild.GetPosition();
                        auto globalRotationChild = globalTransformChild.GetRotation();
                        auto frontChild = globalRotationChild * glm::vec3(0, 0, -1);
                        auto upChild = globalRotationChild * glm::vec3(0, 1, 0);
                        auto rotationChildChild = transformChild.GetRotation();
                        auto internodeInfoChild = child.GetDataComponent<InternodeInfo>();
                        auto internodeStatusChild = child.GetDataComponent<InternodeStatus>();

                        positionChild += glm::normalize(frontChild) * internodeInfoChild.m_length;

                        row += std::to_string(child.GetIndex()) + ",";
                        row += std::to_string(positionChild.x) + ",";
                        row += std::to_string(positionChild.y) + ",";
                        row += std::to_string(positionChild.z) + ",";

                        row += std::to_string(frontChild.x) + ",";
                        row += std::to_string(frontChild.y) + ",";
                        row += std::to_string(frontChild.z) + ",";

                        row += std::to_string(upChild.x) + ",";
                        row += std::to_string(upChild.y) + ",";
                        row += std::to_string(upChild.z) + ",";

                        row += std::to_string(internodeInfoChild.m_thickness) + ",";
                        row += std::to_string(internodeInfoChild.m_length) + ",";
                        row += std::to_string(internodeStatusChild.m_rootDistance) + ",";
                        row += std::to_string(internodeStatusChild.m_chainDistance) + ",";
                        row += std::to_string(internodeStatusChild.m_branchLength) + ",";
                        row += std::to_string(internodeStatusChild.m_level) + ",";

                        row += std::to_string(internodeStatusChild.m_age) + ",";
                        //row += std::to_string(internodeStatusChild.m_recordedProbability) + ",";
                        row += std::to_string(internodeStatusChild.m_startDensity) + ",";
                        row += std::to_string(internodeStatusChild.m_currentTotalNodeCount) + ",";

                        row += std::to_string(globalRotationChild.x) + ",";
                        row += std::to_string(globalRotationChild.y) + ",";
                        row += std::to_string(globalRotationChild.z) + ",";
                        row += std::to_string(globalRotationChild.w);
                    }
                    if (i == 2) row += "\n";
                    else row += ",";
                }
                output += row;
            }
            layerIndex++;
        }
        ofs.write(output.c_str(), output.size());
        ofs.flush();
        ofs.close();
        //UNIENGINE_LOG("Tree group saved: " + path.string() + ".csv");
    } else {
        UNIENGINE_ERROR("Can't open file!");
    }
}

void MultipleAngleCapture::OnStart(AutoTreeGenerationPipeline &pipeline) {
    auto &environment = Application::GetLayer<RayTracerLayer>()->m_environmentProperties;
    environment.m_environmentalLightingType = EnvironmentalLightingType::SingleLightSource;
    environment.m_sunDirection = glm::quat(glm::radians(glm::vec3(60, 160, 0))) * glm::vec3(0, 0, -1);
    environment.m_lightSize = m_lightSize;
    environment.m_ambientLightIntensity = m_ambientLightIntensity;


    m_projections.clear();
    m_views.clear();
    m_names.clear();
    m_cameraModels.clear();
    m_treeModels.clear();
}

void MultipleAngleCapture::OnEnd(AutoTreeGenerationPipeline &pipeline) {
    ExportMatrices(ProjectManager::GetProjectPath().parent_path().parent_path() / m_currentExportFolder /
                   "matrices.yml");
}

void MultipleAngleCapture::DisableAllExport() {
    m_exportOBJ = false;
    m_exportCSV = false;
    m_exportGraph = false;
    m_exportImage = false;
    m_exportDepth = false;
    m_exportMatrices = false;
    m_exportBranchCapture = false;
    m_exportLString = false;
}

GlobalTransform MultipleAngleCapture::TransformCamera(const Bound &bound, float turnAngle, float pitchAngle) {
    GlobalTransform cameraGlobalTransform;
    float distance = m_distance;
    glm::vec3 focusPoint = m_focusPoint;
    if (m_autoAdjustCamera) {
        focusPoint = (bound.m_min + bound.m_max) / 2.0f;
        float halfAngle = (m_fov - 40.0f) / 2.0f;
        float width = bound.m_max.y - bound.m_min.y;
        if(width < bound.m_max.x - bound.m_min.x){
            width = bound.m_max.x - bound.m_min.x;
        }
        if(width < bound.m_max.z - bound.m_min.z){
            width = bound.m_max.z - bound.m_min.z;
        }
        width /= 2.0f;
        distance = width / glm::tan(glm::radians(halfAngle));
    }
    auto height = distance * glm::sin(glm::radians((float) pitchAngle));
    auto groundDistance =
            distance * glm::cos(glm::radians((float) pitchAngle));
    glm::vec3 cameraPosition =
            glm::vec3(glm::sin(glm::radians((float) turnAngle)) * groundDistance,
                      height,
                      glm::cos(glm::radians((float) turnAngle)) * groundDistance);


    cameraGlobalTransform.SetPosition(cameraPosition + focusPoint);
    cameraGlobalTransform.SetRotation(glm::quatLookAt(glm::normalize(-cameraPosition), glm::vec3(0, 1, 0)));
    return cameraGlobalTransform;
}



