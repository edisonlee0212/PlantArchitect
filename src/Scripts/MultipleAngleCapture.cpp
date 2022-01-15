//
// Created by lllll on 9/24/2021.
//

#include "MultipleAngleCapture.hpp"
#include "DepthCamera.hpp"
#include "Entities.hpp"
#include "InternodeLayer.hpp"
#include "AssetManager.hpp"
#include "LSystemBehaviour.hpp"
#include "IVolume.hpp"
#include "InternodeFoliage.hpp"

using namespace Scripts;

void MultipleAngleCapture::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
    if (!SetUpCamera()) {
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        return;
    }
    auto camera = m_cameraEntity.Get().GetOrSetPrivateComponent<Camera>().lock();
    camera->SetRequireRendering(true);
    m_currentGrowingTree.GetOrSetPrivateComponent<Internode>().lock()->m_foliage.Get<InternodeFoliage>()->m_foliagePhyllotaxis = m_foliagePhyllotaxis;
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

    m_captureStatus = MultipleAngleCaptureStatus::Info;
    pipeline.m_status = AutoTreeGenerationPipelineStatus::Growth;
    m_rendering = true;
}

void MultipleAngleCapture::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
    auto camera = m_cameraEntity.Get().GetOrSetPrivateComponent<Camera>().lock();
    camera->SetRequireRendering(true);
    auto behaviour = pipeline.GetBehaviour();
    if (m_rendering) {
        behaviour->GenerateSkinnedMeshes();
        m_rendering = false;
        return;
    }
    auto internodeLayer = Application::GetLayer<InternodeLayer>();
    auto behaviourType = pipeline.GetBehaviourType();
    std::string prefix;
    switch (behaviourType) {
        case BehaviourType::GeneralTree:
            prefix = "GeneralTree_" + pipeline.m_parameterFileName + "_";
            break;
        case BehaviourType::LSystem:
            prefix = "LSystem_";
            break;
        case BehaviourType::SpaceColonization:
            prefix = "SpaceColonization_";
            auto spaceColonizationBehaviour = std::dynamic_pointer_cast<SpaceColonizationBehaviour>(behaviour);
            spaceColonizationBehaviour->m_attractionPoints.clear();
            break;
    }
    prefix += std::to_string(m_generationAmount - m_remainingInstanceAmount);
    switch (m_captureStatus) {
        case MultipleAngleCaptureStatus::Info: {
            auto imagesFolder = m_currentExportFolder / "Image";
            auto objFolder = m_currentExportFolder / "Mesh";
            auto depthFolder = m_currentExportFolder / "Depth";
            auto branchFolder = m_currentExportFolder / "Branch";
            auto graphFolder = m_currentExportFolder / "Graph";
            auto csvFolder = m_currentExportFolder / "CSV";
            auto lStringFolder = m_currentExportFolder / "LString";
            if (m_exportImage)
                std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / imagesFolder);
            if (m_exportOBJ)
                std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / objFolder);
            if (m_exportDepth)
                std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / depthFolder);
            if (m_exportBranchCapture)
                std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / branchFolder);
            if (m_exportGraph)
                std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / graphFolder);
            if (m_exportCSV)
                std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / csvFolder);
            if (m_exportLString)
                std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / lStringFolder);

            if (m_exportLString) {
                auto lString = AssetManager::CreateAsset<LString>();
                m_currentGrowingTree.GetOrSetPrivateComponent<Internode>().lock()->ExportLString(lString);
                //path here
                lString->SetPathAndSave(
                        lStringFolder / (std::to_string(m_generationAmount - m_remainingInstanceAmount) + ".lstring"));
            }
            internodeLayer->UpdateBranchColors();
            behaviour->GenerateSkinnedMeshes();
            if (m_exportOBJ) {
                Entity foliage, branch;
                m_currentGrowingTree.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
                    if (child.GetName() == "Foliage") foliage = child;
                    else if (child.GetName() == "Branch") branch = child;
                });
                if (foliage.IsValid() && foliage.HasPrivateComponent<SkinnedMeshRenderer>()) {
                    auto smr = foliage.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
                    if (smr->m_skinnedMesh.Get<SkinnedMesh>() &&
                        !smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
                        auto exportPath = std::filesystem::absolute(
                                ProjectManager::GetProjectPath().parent_path() / objFolder /
                                (prefix + "_foliage.obj"));
                        UNIENGINE_LOG(exportPath.string());
                        smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
                    }
                }
                if (branch.IsValid() && branch.HasPrivateComponent<SkinnedMeshRenderer>()) {
                    auto smr = branch.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
                    if (smr->m_skinnedMesh.Get<SkinnedMesh>() &&
                        !smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
                        auto exportPath = std::filesystem::absolute(
                                ProjectManager::GetProjectPath().parent_path() / objFolder /
                                (prefix + "_branch.obj"));
                        smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
                    }
                }
            }
            if (m_exportGraph) {
                auto exportPath = std::filesystem::absolute(
                        ProjectManager::GetProjectPath().parent_path() / graphFolder /
                        (prefix + ".yml"));
                ExportGraph(behaviour, exportPath);
            }
            if (m_exportCSV) {
                std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / csvFolder / pipeline.m_parameterFileName);
                auto exportPath = std::filesystem::absolute(
                        ProjectManager::GetProjectPath().parent_path() / csvFolder / pipeline.m_parameterFileName /
                        (std::to_string(pipeline.m_generalTreeParameters.m_gravitropism) + "_" + prefix + ".csv"));
                ExportCSV(behaviour, exportPath);
            }
            if (m_exportBranchCapture) RenderBranchCapture();
            m_captureStatus = MultipleAngleCaptureStatus::Image;
        }
            break;
        case MultipleAngleCaptureStatus::Image: {
            auto anglePrefix = std::to_string(m_pitchAngle) + "_" +
                               std::to_string(m_turnAngle);
            auto imagesFolder = m_currentExportFolder / "Image";
            auto depthFolder = m_currentExportFolder / "Depth";
            auto branchFolder = m_currentExportFolder / "Branch";
            if (m_exportImage) {
                camera->GetTexture()->SetPathAndSave(imagesFolder / (prefix + "_" + anglePrefix + "_rgb.png"));
            }
            if (m_exportDepth) {
                auto depthCamera = m_cameraEntity.Get().GetOrSetPrivateComponent<DepthCamera>().lock();
                depthCamera->Render();
                depthCamera->m_colorTexture->SetPathAndSave(depthFolder / (prefix + "_" + anglePrefix + "_depth.png"));
            }
            if (m_exportBranchCapture) {
                m_branchCaptureCamera->GetTexture()->SetPathAndSave(
                        branchFolder / (prefix + "_" + anglePrefix + "_branch.png"));
            }
            m_cameraModels.push_back(m_cameraEntity.Get().GetDataComponent<GlobalTransform>().m_value);
            m_treeModels.push_back(m_currentGrowingTree.GetDataComponent<GlobalTransform>().m_value);
            m_projections.push_back(Camera::m_cameraInfoBlock.m_projection);
            m_views.push_back(Camera::m_cameraInfoBlock.m_view);
            m_names.push_back(prefix + "_" + anglePrefix);
            if (m_pitchAngle + m_pitchAngleStep < m_pitchAngleEnd) {
                m_pitchAngle += m_pitchAngleStep;
                SetUpCamera();
            } else if (m_turnAngle + m_turnAngleStep < 360.0f) {
                m_turnAngle += m_turnAngleStep;
                m_pitchAngle = m_pitchAngleStart;
                SetUpCamera();
            } else {
                m_remainingInstanceAmount--;
                pipeline.m_generalTreeParameters.m_gravitropism = (m_generationAmount - m_remainingInstanceAmount) % 5 * 0.05f - 0.1f;
                m_pitchAngle = m_pitchAngleStart;
                m_turnAngle = 0;
                if (m_remainingInstanceAmount == 0) {
                    if (m_exportMatrices) {
                        ExportMatrices(ProjectManager::GetProjectPath().parent_path() / m_currentExportFolder /
                                       (prefix + "_camera_matrices.yml"));
                    }
                    ProjectManager::ScanProjectFolder(true);
                    pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
                } else {
                    pipeline.m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
                }
            }
        }
            break;
    }
}

void MultipleAngleCapture::OnInspect() {
    Editor::DragAndDropButton(m_volume, "Volume", {"CubeVolume", "RadialBoundingVolume"}, false);
    Editor::DragAndDropButton(m_foliagePhyllotaxis, "Phyllotaxis",
                              {"EmptyInternodePhyllotaxis", "DefaultInternodePhyllotaxis"}, true);
    ImGui::Checkbox("Auto adjust camera", &m_autoAdjustCamera);
    if (ImGui::TreeNodeEx("Pipeline Settings")) {
        ImGui::DragFloat("Branch width", &m_branchWidth, 0.01f);
        ImGui::DragInt("Generation Amount", &m_generationAmount);
        ImGui::DragInt("Growth iteration", &m_perTreeGrowthIteration);

        ImGui::Text("Data export:");
        ImGui::Checkbox("Export OBJ", &m_exportOBJ);
        ImGui::Checkbox("Export Graph", &m_exportGraph);
        ImGui::Checkbox("Export CSV", &m_exportCSV);
        ImGui::Checkbox("Export LString", &m_exportLString);
        Application::GetLayer<InternodeLayer>()->DrawColorModeSelectionMenu();

        ImGui::Text("Rendering export:");
        ImGui::Checkbox("Export Depth", &m_exportDepth);
        ImGui::Checkbox("Export Image", &m_exportImage);
        ImGui::Checkbox("Export Branch Capture", &m_exportBranchCapture);
        ImGui::Checkbox("Export Camera matrices", &m_exportMatrices);
        ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Camera")) {
        Editor::DragAndDropButton(m_cameraEntity, "Attached Camera Entity");
        if (!m_autoAdjustCamera) {
            ImGui::Text("Position:");
            ImGui::DragFloat3("Focus point", &m_focusPoint.x, 0.1f);
            ImGui::DragFloat("Distance to focus point", &m_distance, 0.1);
        }
        ImGui::Text("Rotation:");
        ImGui::DragFloat3("Pitch Angle From/Step/End", &m_pitchAngleStart, 1);
        ImGui::DragFloat("Turn Angle Step", &m_turnAngleStep, 1);

        ImGui::Text("Camera Settings:");
        ImGui::DragFloat("Camera FOV", &m_fov);
        ImGui::DragInt2("Camera Resolution", &m_resolution.x);
        ImGui::DragFloat2("Camera near/far", &m_cameraMin);
        ImGui::Checkbox("Use clear color", &m_useClearColor);
        ImGui::ColorEdit3("Camera Clear Color", &m_backgroundColor.x);

        ImGui::TreePop();
    }
    if (m_remainingInstanceAmount == 0) {
        if (Application::IsPlaying()) {
            if (ImGui::Button("Start")) {
                Start();
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

void MultipleAngleCapture::OnIdle(AutoTreeGenerationPipeline &pipeline) {
    if (m_cameraEntity.Get().IsNull()) {
        m_pitchAngle = m_pitchAngleStart;
        m_turnAngle = -1;
        m_generationAmount = 0;
        return;
    }
    if (m_pitchAngle == m_pitchAngleStart && m_turnAngle == 0 && m_remainingInstanceAmount > 0) {
        pipeline.m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
    }
}

bool MultipleAngleCapture::SetUpCamera() {
    if (m_autoAdjustCamera) {
        m_focusPoint = glm::vec3(0.0f, m_perTreeGrowthIteration / 4.0, 0.0f);
        m_distance = m_perTreeGrowthIteration * 2.0f;
    }
    auto cameraEntity = m_cameraEntity.Get();
    if (cameraEntity.IsNull()) {
        m_pitchAngle = m_pitchAngleStart;
        m_turnAngle = m_remainingInstanceAmount = 0;
        UNIENGINE_ERROR("Camera entity missing!");
        return false;
    }

    auto height = m_distance * glm::sin(glm::radians((float) m_pitchAngle));
    auto groundDistance =
            m_distance * glm::cos(glm::radians((float) m_pitchAngle));
    glm::vec3 cameraPosition =
            glm::vec3(glm::sin(glm::radians((float) m_turnAngle)) * groundDistance,
                      height,
                      glm::cos(glm::radians((float) m_turnAngle)) * groundDistance);
    m_cameraPosition = cameraPosition + m_focusPoint;
    m_cameraRotation = glm::quatLookAt(glm::normalize(-cameraPosition), glm::vec3(0, 1, 0));

    GlobalTransform cameraGlobalTransform;
    cameraGlobalTransform.SetPosition(m_cameraPosition);
    cameraGlobalTransform.SetRotation(m_cameraRotation);
    cameraEntity.SetDataComponent(cameraGlobalTransform);

    auto camera = cameraEntity.GetOrSetPrivateComponent<Camera>().lock();
    camera->m_fov = m_fov;
    camera->m_allowAutoResize = false;
    camera->m_farDistance = m_cameraMax;
    camera->m_nearDistance = m_cameraMin;
    camera->ResizeResolution(m_resolution.x, m_resolution.y);
    camera->m_clearColor = m_backgroundColor;
    camera->m_useClearColor = m_useClearColor;

    auto depthCamera = cameraEntity.GetOrSetPrivateComponent<DepthCamera>().lock();
    depthCamera->m_useCameraResolution = true;

    m_branchCaptureCamera->m_fov = m_fov;
    m_branchCaptureCamera->m_allowAutoResize = false;
    m_branchCaptureCamera->m_farDistance = m_cameraMax;
    m_branchCaptureCamera->m_nearDistance = m_cameraMin;
    m_branchCaptureCamera->ResizeResolution(m_resolution.x, m_resolution.y);
    m_branchCaptureCamera->m_clearColor = m_backgroundColor;
    m_branchCaptureCamera->m_useClearColor = m_useClearColor;

    if (cameraEntity.HasPrivateComponent<PostProcessing>()) {
        auto postProcessing = cameraEntity.GetOrSetPrivateComponent<PostProcessing>().lock();
        postProcessing->SetEnabled(false);
    }
    return true;
}

void MultipleAngleCapture::RenderBranchCapture() {
    auto internodeQuery = Application::GetLayer<InternodeLayer>()->m_internodesQuery;
    /*
    Entities::ForEach<BranchColor, InternodeInfo>(
            Entities::GetCurrentScene(), JobManager::PrimaryWorkers(),
            internodeQuery,
            [=](int i, Entity entity, BranchColor &internodeRenderColor,
                InternodeInfo &internodeInfo) {
                internodeRenderColor.m_value = glm::vec4(((entity.GetIndex() / 4096) % 64) / 64.0f,
                                                         ((entity.GetIndex() / 64) % 64) / 64.0f,
                                                         (entity.GetIndex() % 64) / 64.0f, 1.0f);
            },
            true);
    */
    Entities::ForEach<GlobalTransform, BranchCylinder, InternodeInfo>(
            Entities::GetCurrentScene(), Jobs::Workers(),
            internodeQuery,
            [=](int i, Entity entity, GlobalTransform &ltw, BranchCylinder &c,
                InternodeInfo &internodeInfo) {
                glm::vec3 scale;
                glm::quat rotation;
                glm::vec3 translation;
                glm::vec3 skew;
                glm::vec4 perspective;
                glm::decompose(ltw.m_value, scale, rotation, translation, skew,
                               perspective);
                const auto direction = glm::normalize(rotation * glm::vec3(0, 0, -1));
                const glm::vec3 position2 =
                        translation + internodeInfo.m_length * direction;
                rotation = glm::quatLookAt(
                        direction, glm::vec3(direction.y, direction.z, direction.x));
                rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
                const glm::mat4 rotationTransform = glm::mat4_cast(rotation);
                c.m_value =
                        glm::translate((translation + position2) / 2.0f) *
                        rotationTransform *
                        glm::scale(glm::vec3(
                                m_branchWidth,
                                glm::distance(translation, position2) / 2.0f,
                                m_branchWidth));

            },
            true);

    m_branchCaptureCamera->Clear();
    std::vector<BranchCylinder> branchCylinders;
    internodeQuery.ToComponentDataArray<BranchCylinder>(Entities::GetCurrentScene(),
                                                        branchCylinders);
    std::vector<BranchColor> branchColors;
    internodeQuery.ToComponentDataArray<BranchColor>(Entities::GetCurrentScene(),
                                                     branchColors);
    std::vector<GlobalTransform> branchGlobalTransforms;
    internodeQuery.ToComponentDataArray<GlobalTransform>(Entities::GetCurrentScene(),
                                                         branchGlobalTransforms);
    Graphics::DrawGizmoMeshInstancedColored(
            DefaultResources::Primitives::Cylinder, m_branchCaptureCamera,
            m_cameraPosition,
            m_cameraRotation,
            *reinterpret_cast<std::vector<glm::vec4> *>(&branchColors),
            *reinterpret_cast<std::vector<glm::mat4> *>(&branchCylinders),
            glm::mat4(1.0f), 1.0f);

    Graphics::DrawGizmoMeshInstanced(
            DefaultResources::Primitives::Sphere, m_branchCaptureCamera,
            m_cameraPosition,
            m_cameraRotation,
            glm::vec4(1.0f),
            *reinterpret_cast<std::vector<glm::mat4> *>(&branchGlobalTransforms),
            glm::mat4(1.0f), m_nodeSize);
}

void MultipleAngleCapture::OnCreate() {
    m_branchCaptureCamera = Serialization::ProduceSerializable<Camera>();
    m_branchCaptureCamera->OnCreate();
}

void MultipleAngleCapture::ExportGraph(const std::shared_ptr<IInternodeBehaviour> &behaviour,
                                       const std::filesystem::path &path) {
    try {
        auto directory = path;
        directory.remove_filename();
        std::filesystem::create_directories(directory);
        YAML::Emitter out;
        out << YAML::BeginMap;
        std::vector<std::vector<std::pair<int, Entity>>> internodes;
        internodes.resize(128);
        internodes[0].emplace_back(-1, m_currentGrowingTree);
        behaviour->TreeGraphWalkerRootToEnd(m_currentGrowingTree, m_currentGrowingTree,
                                            [&](Entity parent, Entity child) {
                                                if (!behaviour->InternodeCheck(child)) return;
                                                auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                                internodes[childInternodeInfo.m_layer].emplace_back(parent.GetIndex(),
                                                                                                    child);
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

void MultipleAngleCapture::ExportGraphNode(const std::shared_ptr<IInternodeBehaviour> &behaviour, YAML::Emitter &out,
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
    out << YAML::Key << "Distance to Root" << internodeStatus.m_distanceToRoot;
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

void MultipleAngleCapture::ExportCSV(const std::shared_ptr<IInternodeBehaviour> &behaviour,
                                     const std::filesystem::path &path) {
    std::ofstream ofs;
    ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (ofs.is_open()) {
        std::string output;

        std::vector<std::vector<std::pair<int, Entity>>> internodes;
        internodes.resize(128);
        internodes[0].emplace_back(-1, m_currentGrowingTree);
        behaviour->TreeGraphWalkerRootToEnd(m_currentGrowingTree, m_currentGrowingTree,
                                            [&](Entity parent, Entity child) {
                                                if (!behaviour->InternodeCheck(child)) return;
                                                auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                                if(childInternodeInfo.m_endNode) return;
                                                internodes[childInternodeInfo.m_layer].emplace_back(parent.GetIndex(),
                                                                                                    child);
                                            });
        output += "in_id, in_pos_x, in_pos_y, in_pos_z, in_front_x, in_front_y, in_front_z, in_up_x, in_up_y, in_up_z, in_thickness, in_length, in_root_distance, in_chain_distance, in_level, in_flush_age, in_quat_x, in_quat_y, in_quat_z, in_quat_w, ";
        output += "out0_id, out0_pos_x, out0_pos_y, out0_pos_z, out0_front_x, out0_front_y, out0_front_z, out0_up_x, out0_up_y, out0_up_z, out0_thickness, out0_length, out0_root_distance, out0_chain_distance, out0_level, out0_flush_age, out0_quat_x, out0_quat_y, out0_quat_z, out0_quat_w, ";
        output += "out1_id, out1_pos_x, out1_pos_y, out1_pos_z, out1_front_x, out1_front_y, out1_front_z, out1_up_x, out1_up_y, out1_up_z, out1_thickness, out1_length, out1_root_distance, out1_chain_distance, out1_level, out1_flush_age, out1_quat_x, out1_quat_y, out1_quat_z, out1_quat_w, ";
        output += "out2_id, out2_pos_x, out2_pos_y, out2_pos_z, out2_front_x, out2_front_y, out2_front_z, out2_up_x, out2_up_y, out2_up_z, out2_thickness, out2_length, out2_root_distance, out2_chain_distance, out2_level, out2_flush_age, out2_quat_x, out2_quat_y, out2_quat_z, out2_quat_w\n";
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
                row += std::to_string(internode.GetIndex()) + ", ";

                row += std::to_string(position.x) + ", ";
                row += std::to_string(position.y) + ", ";
                row += std::to_string(position.z) + ", ";

                row += std::to_string(front.x) + ", ";
                row += std::to_string(front.y) + ", ";
                row += std::to_string(front.z) + ", ";

                row += std::to_string(up.x) + ", ";
                row += std::to_string(up.y) + ", ";
                row += std::to_string(up.z) + ", ";

                row += std::to_string(internodeInfo.m_thickness) + ", ";
                row += std::to_string(internodeInfo.m_length) + ", ";
                row += std::to_string(internodeStatus.m_distanceToRoot) + ", ";
                row += std::to_string(internodeStatus.m_chainDistance) + ", ";
                row += std::to_string(internodeStatus.m_level) + ", ";

                row += std::to_string(internodeStatus.m_age) + ", ";

                row += std::to_string(globalRotation.x) + ", ";
                row += std::to_string(globalRotation.y) + ", ";
                row += std::to_string(globalRotation.z) + ", ";
                row += std::to_string(globalRotation.w) + ", ";

                for (int i = 0; i < 3; i++) {
                    auto child = children[i];
                    if (child.IsNull() || child.GetDataComponent<InternodeInfo>().m_endNode) {
                        row += "N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A";
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
                        row += std::to_string(child.GetIndex()) + ", ";
                        row += std::to_string(positionChild.x) + ", ";
                        row += std::to_string(positionChild.y) + ", ";
                        row += std::to_string(positionChild.z) + ", ";

                        row += std::to_string(frontChild.x) + ", ";
                        row += std::to_string(frontChild.y) + ", ";
                        row += std::to_string(frontChild.z) + ", ";

                        row += std::to_string(upChild.x) + ", ";
                        row += std::to_string(upChild.y) + ", ";
                        row += std::to_string(upChild.z) + ", ";

                        row += std::to_string(internodeInfoChild.m_thickness) + ", ";
                        row += std::to_string(internodeInfoChild.m_length) + ", ";
                        row += std::to_string(internodeStatusChild.m_distanceToRoot) + ", ";
                        row += std::to_string(internodeStatusChild.m_chainDistance) + ", ";
                        row += std::to_string(internodeStatusChild.m_level) + ", ";

                        row += std::to_string(internodeStatusChild.m_age) + ", ";

                        row += std::to_string(globalRotationChild.x) + ", ";
                        row += std::to_string(globalRotationChild.y) + ", ";
                        row += std::to_string(globalRotationChild.z) + ", ";
                        row += std::to_string(globalRotationChild.w);
                    }
                    if (i == 2) row += "\n";
                    else row += ", ";
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

void MultipleAngleCapture::Start() {
    if (!m_cameraEntity.Get().IsNull()) {
        m_cameraEntity.Get().GetOrSetPrivateComponent<DepthCamera>();
        m_projections.clear();
        m_views.clear();
        m_names.clear();
        m_cameraModels.clear();
        m_treeModels.clear();
        m_pitchAngle = m_pitchAngleStart;
        m_turnAngle = 0;
        m_remainingInstanceAmount = m_generationAmount;
    }
}

bool MultipleAngleCapture::Busy() const {
    return m_remainingInstanceAmount != 0;
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



