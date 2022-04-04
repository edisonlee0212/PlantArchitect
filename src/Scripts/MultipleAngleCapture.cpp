//
// Created by lllll on 9/24/2021.
//

#include "MultipleAngleCapture.hpp"
#include "Entities.hpp"
#include "PlantLayer.hpp"
#include "ProjectManager.hpp"
#include "LSystemBehaviour.hpp"
#ifdef RAYTRACERFACILITY
#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"
using namespace RayTracerFacility;
#endif

#include "TransformLayer.hpp"
#include "DefaultInternodePhyllotaxis.hpp"


using namespace Scripts;

void MultipleAngleCapture::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
    Entity rootInternode;
    auto children = pipeline.m_currentGrowingTree.GetChildren();
    for (const auto &i: children) {
        if (i.HasPrivateComponent<Internode>()) rootInternode = i;
    }
    auto internode = rootInternode.GetOrSetPrivateComponent<Internode>().lock();
    auto root = pipeline.m_currentGrowingTree.GetOrSetPrivateComponent<Root>().lock();
    if (m_applyPhyllotaxis) {
        root->m_foliagePhyllotaxis = m_foliagePhyllotaxis;
    }
    root->m_branchTexture = m_branchTexture;
    root->m_foliageTexture = m_foliageTexture;
    if (m_exportImage || m_exportDepth || m_exportBranchCapture) {
        SetUpCamera(pipeline);
    }
    pipeline.m_status = AutoTreeGenerationPipelineStatus::Growth;
}

void MultipleAngleCapture::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
    auto behaviour = pipeline.GetBehaviour();
#ifdef RAYTRACERFACILITY
    auto camera = pipeline.GetOwner().GetOrSetPrivateComponent<RayTracerCamera>().lock();
#endif
    auto internodeLayer = Application::GetLayer<PlantLayer>();
    auto behaviourType = pipeline.GetBehaviourType();
    Entity rootInternode;
    auto children = pipeline.m_currentGrowingTree.GetChildren();
    for (const auto &i: children) {
        if (i.HasPrivateComponent<Internode>()) rootInternode = i;
    }
    auto treeIOFolder = m_currentExportFolder / "TreeIO";
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
        std::filesystem::create_directories(graphFolder);
        auto exportPath = graphFolder /
                          (pipeline.m_prefix + ".yml");
        ExportGraph(pipeline, behaviour, exportPath);
    }
    if (pipeline.m_behaviourType == BehaviourType::GeneralTree && m_exportCSV) {
        std::filesystem::create_directories(csvFolder);

        std::filesystem::create_directories(csvFolder /
                                            pipeline.m_currentUsingDescriptor.Get<IPlantDescriptor>()->GetAssetRecord().lock()->GetAssetFileName());
        auto exportPath = std::filesystem::absolute(csvFolder /
                                                    pipeline.m_currentUsingDescriptor.Get<IPlantDescriptor>()->GetAssetRecord().lock()->GetAssetFileName() /
                                                    (pipeline.m_prefix + ".csv"));
        ExportCSV(pipeline, behaviour, exportPath);
    }
    if (m_exportLString) {
        std::filesystem::create_directories(lStringFolder);
        auto lString = ProjectManager::CreateTemporaryAsset<LSystemString>();

        rootInternode.GetOrSetPrivateComponent<Internode>().lock()->ExportLString(lString);
        //path here
        lString->Export(
                lStringFolder / (pipeline.m_prefix + "_" +
                                 std::to_string(pipeline.m_generationAmount - pipeline.m_remainingInstanceAmount) +
                                 ".lstring"));
    }
    if (m_exportTreeIOTrees) {
        std::filesystem::create_directories(lStringFolder);
        rootInternode.GetOrSetPrivateComponent<Internode>().lock()->ExportTreeIOTree(
                treeIOFolder / (pipeline.m_prefix + "_" +
                                std::to_string(pipeline.m_generationAmount - pipeline.m_remainingInstanceAmount) +
                                ".tree"));
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
        std::filesystem::create_directories(objFolder);
        Entity foliage, branch;
        pipeline.m_currentGrowingTree.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (child.GetName() == "Foliage") foliage = child;
            else if (child.GetName() == "Branch") branch = child;
        });
        if (foliage.IsValid() && foliage.HasPrivateComponent<SkinnedMeshRenderer>()) {
            auto smr = foliage.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
            if (smr->m_skinnedMesh.Get<SkinnedMesh>() &&
                !smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
                auto exportPath = objFolder /
                                  (pipeline.m_prefix + "_foliage.obj");
                UNIENGINE_LOG(exportPath.string());
                smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
            }
        }
        if (branch.IsValid() && branch.HasPrivateComponent<SkinnedMeshRenderer>()) {
            auto smr = branch.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
            if (smr->m_skinnedMesh.Get<SkinnedMesh>() &&
                !smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
                auto exportPath = objFolder /
                                  (pipeline.m_prefix + "_branch.obj");
                smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
            }
        }
    }

#ifdef RAYTRACERFACILITY
    if (m_exportImage) {
        std::filesystem::create_directories(imagesFolder);
        auto cameraEntity = pipeline.GetOwner();
        auto rayTracerCamera = cameraEntity.GetOrSetPrivateComponent<RayTracerCamera>().lock();
        rayTracerCamera->SetOutputType(OutputType::Color);
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
                        imagesFolder / (pipeline.m_prefix + "_" + anglePrefix + "_rgb.png"));
            }
        }
    }
    if (m_exportDepth) {
        std::filesystem::create_directories(depthFolder);
        auto cameraEntity = pipeline.GetOwner();
        auto rayTracerCamera = cameraEntity.GetOrSetPrivateComponent<RayTracerCamera>().lock();
        rayTracerCamera->SetOutputType(OutputType::Depth);
        rayTracerCamera->SetMaxDistance(m_cameraMax);
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
                        depthFolder / (pipeline.m_prefix + "_" + anglePrefix + "_depth.hdr"));
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

                        branchFolder / (pipeline.m_prefix + "_" + anglePrefix + "_branch.png"));
            }
        }
    }
#endif
#pragma endregion
    pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
}

static const char *DefaultBehaviourTypes[]{"GeneralTree", "LSystem", "SpaceColonization"};

void MultipleAngleCapture::OnInspect() {
    if (ImGui::Button("Instantiate Pipeline")) {
        auto multipleAngleCapturePipelineEntity = Entities::CreateEntity(Entities::GetCurrentScene(),
                                                                         "GANTree Dataset Pipeline");
        auto multipleAngleCapturePipeline = multipleAngleCapturePipelineEntity.GetOrSetPrivateComponent<AutoTreeGenerationPipeline>().lock();
        multipleAngleCapturePipeline->m_pipelineBehaviour = m_self.lock();
        multipleAngleCapturePipeline->SetBehaviourType(m_defaultBehaviourType);
    }
    int behaviourType = (int) m_defaultBehaviourType;
    if (ImGui::Combo(
            "Plant behaviour type",
            &behaviourType,
            DefaultBehaviourTypes,
            IM_ARRAYSIZE(DefaultBehaviourTypes))) {
        m_defaultBehaviourType = (BehaviourType) behaviourType;
    }
    ImGui::Text("Current output folder: %s", m_currentExportFolder.string().c_str());
    FileUtils::OpenFolder("Choose output folder...", [&](const std::filesystem::path &path) {
        m_currentExportFolder = std::filesystem::absolute(path);
    }, false);
    if (ImGui::TreeNodeEx("Pipeline Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat("Branch width", &m_branchWidth, 0.01f);

        ImGui::Checkbox("Override phyllotaxis", &m_applyPhyllotaxis);
        if (m_applyPhyllotaxis) {
            Editor::DragAndDropButton<DefaultInternodePhyllotaxis>(m_foliagePhyllotaxis, "Phyllotaxis", true);
            Editor::DragAndDropButton<Texture2D>(m_foliageTexture, "Foliage texture", true);
            Editor::DragAndDropButton<Texture2D>(m_branchTexture, "Branch texture", true);
        }
        ImGui::Text("Data export:");
        ImGui::Checkbox("Export TreeIO", &m_exportTreeIOTrees);
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
            ImGui::DragFloat("Camera max distance", &m_cameraMax);
            ImGui::Checkbox("Use clear color", &m_useClearColor);
            ImGui::ColorEdit3("Camera Clear Color", &m_backgroundColor.x);
            ImGui::DragFloat("Light Size", &m_lightSize, 0.001f);
            ImGui::DragFloat("Ambient light intensity", &m_ambientLightIntensity, 0.01f);
            ImGui::DragFloat("Environment light intensity", &m_envLightIntensity, 0.01f);
            ImGui::TreePop();
        }
        if (m_exportBranchCapture) Application::GetLayer<PlantLayer>()->DrawColorModeSelectionMenu();
    }
}

void MultipleAngleCapture::SetUpCamera(AutoTreeGenerationPipeline &pipeline) {
    auto cameraEntity = pipeline.GetOwner();
    assert(cameraEntity.IsValid());
#ifdef RAYTRACERFACILITY
    auto camera = cameraEntity.GetOrSetPrivateComponent<RayTracerCamera>().lock();
    camera->SetFov(m_fov);
    camera->m_allowAutoResize = false;
    camera->m_frameSize = m_resolution;
#endif
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
    out << YAML::Key << "Branching Order" << YAML::Value << internodeStatus.m_branchingOrder;
    out << YAML::Key << "Level" << YAML::Value << internodeStatus.m_level;
    out << YAML::Key << "Distance to Root" << YAML::Value << internodeStatus.m_rootDistance;
    out << YAML::Key << "Local Rotation" << YAML::Value << transform.GetRotation();
    out << YAML::Key << "Global Rotation" << YAML::Value << globalRotation;
    out << YAML::Key << "Position" << YAML::Value << position + front * internodeInfo.m_length;
    out << YAML::Key << "Front Direction" << YAML::Value << front;
    out << YAML::Key << "Up Direction" << YAML::Value << up;
    out << YAML::Key << "IsEndNode" << YAML::Value << internodeInfo.m_endNode;
    out << YAML::Key << "Thickness" << YAML::Value << internodeInfo.m_thickness;
    out << YAML::Key << "Length" << YAML::Value << internodeInfo.m_length;
    //out << YAML::Key << "Internode Index" << YAML::Value << internodeInfo.m_index;
    out << YAML::Key << "Internode Layer" << YAML::Value << internodeInfo.m_layer;
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
#ifdef RAYTRACERFACILITY
    auto &environment = Application::GetLayer<RayTracerLayer>()->m_environmentProperties;
    environment.m_environmentalLightingType = EnvironmentalLightingType::SingleLightSource;
    environment.m_sunDirection = glm::quat(glm::radians(glm::vec3(60, 160, 0))) * glm::vec3(0, 0, -1);
    environment.m_lightSize = m_lightSize;
    environment.m_ambientLightIntensity = m_ambientLightIntensity;
    Entities::GetCurrentScene()->m_environmentSettings.m_ambientLightIntensity = m_envLightIntensity;
#endif

    m_projections.clear();
    m_views.clear();
    m_names.clear();
    m_cameraModels.clear();
    m_treeModels.clear();
}

void MultipleAngleCapture::OnEnd(AutoTreeGenerationPipeline &pipeline) {
    ExportMatrices(m_currentExportFolder /
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
        if (width < bound.m_max.x - bound.m_min.x) {
            width = bound.m_max.x - bound.m_min.x;
        }
        if (width < bound.m_max.z - bound.m_min.z) {
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

void MultipleAngleCapture::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_foliageTexture);
    list.push_back(m_branchTexture);
    list.push_back(m_foliagePhyllotaxis);


}

void MultipleAngleCapture::Serialize(YAML::Emitter &out) {
    m_foliageTexture.Save("m_foliageTexture", out);
    m_branchTexture.Save("m_branchTexture", out);
    m_foliagePhyllotaxis.Save("m_foliagePhyllotaxis", out);

    out << YAML::Key << "m_defaultBehaviourType" << YAML::Value << (unsigned) m_defaultBehaviourType;
    out << YAML::Key << "m_autoAdjustCamera" << YAML::Value << m_autoAdjustCamera;
    out << YAML::Key << "m_applyPhyllotaxis" << YAML::Value << m_applyPhyllotaxis;
    out << YAML::Key << "m_currentExportFolder" << YAML::Value << m_currentExportFolder.string();
    out << YAML::Key << "m_branchWidth" << YAML::Value << m_branchWidth;
    out << YAML::Key << "m_nodeSize" << YAML::Value << m_nodeSize;
    out << YAML::Key << "m_focusPoint" << YAML::Value << m_focusPoint;
    out << YAML::Key << "m_pitchAngleStart" << YAML::Value << m_pitchAngleStart;
    out << YAML::Key << "m_pitchAngleStep" << YAML::Value << m_pitchAngleStep;
    out << YAML::Key << "m_pitchAngleEnd" << YAML::Value << m_pitchAngleEnd;
    out << YAML::Key << "m_turnAngleStart" << YAML::Value << m_turnAngleStart;
    out << YAML::Key << "m_turnAngleStep" << YAML::Value << m_turnAngleStep;
    out << YAML::Key << "m_turnAngleEnd" << YAML::Value << m_turnAngleEnd;
    out << YAML::Key << "m_distance" << YAML::Value << m_distance;
    out << YAML::Key << "m_fov" << YAML::Value << m_fov;
    out << YAML::Key << "m_lightSize" << YAML::Value << m_lightSize;
    out << YAML::Key << "m_ambientLightIntensity" << YAML::Value << m_ambientLightIntensity;
    out << YAML::Key << "m_envLightIntensity" << YAML::Value << m_envLightIntensity;
    out << YAML::Key << "m_resolution" << YAML::Value << m_resolution;
    out << YAML::Key << "m_exportTreeIOTrees" << YAML::Value << m_exportTreeIOTrees;
    out << YAML::Key << "m_exportOBJ" << YAML::Value << m_exportOBJ;
    out << YAML::Key << "m_exportCSV" << YAML::Value << m_exportCSV;
    out << YAML::Key << "m_exportGraph" << YAML::Value << m_exportGraph;
    out << YAML::Key << "m_exportImage" << YAML::Value << m_exportImage;
    out << YAML::Key << "m_exportDepth" << YAML::Value << m_exportDepth;
    out << YAML::Key << "m_exportMatrices" << YAML::Value << m_exportMatrices;
    out << YAML::Key << "m_exportBranchCapture" << YAML::Value << m_exportBranchCapture;
    out << YAML::Key << "m_exportLString" << YAML::Value << m_exportLString;
    out << YAML::Key << "m_useClearColor" << YAML::Value << m_useClearColor;
    out << YAML::Key << "m_backgroundColor" << YAML::Value << m_backgroundColor;
    out << YAML::Key << "m_cameraMax" << YAML::Value << m_cameraMax;
}

void MultipleAngleCapture::Deserialize(const YAML::Node &in) {
    m_foliageTexture.Load("m_foliageTexture", in);
    m_branchTexture.Load("m_branchTexture", in);
    m_foliagePhyllotaxis.Load("m_foliagePhyllotaxis", in);


    if (in["m_defaultBehaviourType"]) m_defaultBehaviourType = (BehaviourType) in["m_defaultBehaviourType"].as<unsigned>();
    if (in["m_autoAdjustCamera"]) m_autoAdjustCamera = in["m_autoAdjustCamera"].as<bool>();
    if (in["m_applyPhyllotaxis"]) m_applyPhyllotaxis = in["m_applyPhyllotaxis"].as<bool>();
    if (in["m_currentExportFolder"]) m_currentExportFolder = in["m_currentExportFolder"].as<std::string>();
    if (in["m_branchWidth"]) m_branchWidth = in["m_branchWidth"].as<float>();
    if (in["m_nodeSize"]) m_nodeSize = in["m_nodeSize"].as<float>();
    if (in["m_focusPoint"]) m_focusPoint = in["m_focusPoint"].as<glm::vec3>();
    if (in["m_pitchAngleStart"]) m_pitchAngleStart = in["m_pitchAngleStart"].as<int>();
    if (in["m_pitchAngleStep"]) m_pitchAngleStep = in["m_pitchAngleStep"].as<int>();
    if (in["m_pitchAngleEnd"]) m_pitchAngleEnd = in["m_pitchAngleEnd"].as<int>();
    if (in["m_turnAngleStart"]) m_turnAngleStart = in["m_turnAngleStart"].as<int>();
    if (in["m_turnAngleStep"]) m_turnAngleStep = in["m_turnAngleStep"].as<int>();
    if (in["m_turnAngleEnd"]) m_turnAngleEnd = in["m_turnAngleEnd"].as<int>();
    if (in["m_distance"]) m_distance = in["m_distance"].as<float>();
    if (in["m_fov"]) m_fov = in["m_fov"].as<float>();
    if (in["m_lightSize"]) m_lightSize = in["m_lightSize"].as<float>();
    if (in["m_ambientLightIntensity"]) m_ambientLightIntensity = in["m_ambientLightIntensity"].as<float>();
    if (in["m_envLightIntensity"]) m_envLightIntensity = in["m_envLightIntensity"].as<float>();
    if (in["m_resolution"]) m_resolution = in["m_resolution"].as<glm::ivec2>();
    if (in["m_exportTreeIOTrees"]) m_exportTreeIOTrees = in["m_exportTreeIOTrees"].as<bool>();
    if (in["m_exportOBJ"]) m_exportOBJ = in["m_exportOBJ"].as<bool>();
    if (in["m_exportCSV"]) m_exportCSV = in["m_exportCSV"].as<bool>();
    if (in["m_exportGraph"]) m_exportGraph = in["m_exportGraph"].as<bool>();
    if (in["m_exportImage"]) m_exportImage = in["m_exportImage"].as<bool>();
    if (in["m_exportDepth"]) m_exportDepth = in["m_exportDepth"].as<bool>();
    if (in["m_exportMatrices"]) m_exportMatrices = in["m_exportMatrices"].as<bool>();
    if (in["m_exportBranchCapture"]) m_exportBranchCapture = in["m_exportBranchCapture"].as<bool>();
    if (in["m_exportLString"]) m_exportLString = in["m_exportLString"].as<bool>();
    if (in["m_useClearColor"]) m_useClearColor = in["m_useClearColor"].as<bool>();
    if (in["m_backgroundColor"]) m_backgroundColor = in["m_backgroundColor"].as<glm::vec3>();
    if (in["m_cameraMax"]) m_cameraMax = in["m_cameraMax"].as<float>();
}

MultipleAngleCapture::~MultipleAngleCapture() {
    m_foliagePhyllotaxis.Clear();
    m_foliageTexture.Clear();
    m_branchTexture.Clear();
}



