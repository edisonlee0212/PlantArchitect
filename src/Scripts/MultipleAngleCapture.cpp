//
// Created by lllll on 9/24/2021.
//

#include "MultipleAngleCapture.hpp"
#include "DepthCamera.hpp"
#include "EntityManager.hpp"
#include "InternodeSystem.hpp"
#include "AssetManager.hpp"
#include "LSystemBehaviour.hpp"
#include "IVolume.hpp"

using namespace Scripts;

void MultipleAngleCapture::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
    if (!SetUpCamera()) {
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        return;
    }

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
}

void MultipleAngleCapture::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
    auto behaviour = pipeline.GetBehaviour();
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
            if (m_exportLString)
                std::filesystem::create_directories(ProjectManager::GetProjectPath().parent_path() / lStringFolder);

            if (m_exportLString) {
                auto lString = AssetManager::CreateAsset<LString>();
                m_currentGrowingTree.GetOrSetPrivateComponent<Internode>().lock()->ExportLString(lString);
                //path here
                lString->SetPathAndSave(
                        lStringFolder / (std::to_string(m_generationAmount - m_remainingInstanceAmount) + ".lstring"));
                auto internodeSystem = EntityManager::GetCurrentScene()->GetSystem<InternodeSystem>();
                internodeSystem->m_branchColorMode = BranchColorMode::IndexDivider;
                internodeSystem->m_indexDivider = m_targetDivider;
                internodeSystem->UpdateBranchColors();
            }
            behaviour->GenerateSkinnedMeshes();
            if (m_exportOBJ) {
                Entity foliage, branch;
                m_currentGrowingTree.ForEachChild([&](const std::shared_ptr<Scene>& scene, Entity child) {
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
            if (m_exportBranchCapture) RenderBranchCapture();

            m_skipCurrentFrame = true;
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
                auto camera = m_cameraEntity.Get().GetOrSetPrivateComponent<Camera>().lock();
                camera->GetTexture()->SetPathAndSave(imagesFolder / (prefix + "_" + anglePrefix + "_rgb.png"));
            }
            if (m_exportDepth) {
                auto depthCamera = m_cameraEntity.Get().GetOrSetPrivateComponent<DepthCamera>().lock();
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
                m_skipCurrentFrame = true;
            } else if (m_turnAngle + m_turnAngleStep < 360.0f) {
                m_turnAngle += m_turnAngleStep;
                m_pitchAngle = m_pitchAngleStart;
                SetUpCamera();
                m_skipCurrentFrame = true;
            } else {
                m_remainingInstanceAmount--;
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
    EditorManager::DragAndDropButton(m_volume, "Volume", {"CubeVolume", "RadialBoundingVolume"}, false);

    ImGui::Checkbox("Auto adjust camera", &m_autoAdjustCamera);
    if (ImGui::TreeNodeEx("Pipeline Settings")) {
        ImGui::DragFloat("Branch width", &m_branchWidth, 0.01f);
        ImGui::DragInt("Generation Amount", &m_generationAmount);
        ImGui::DragInt("Growth iteration", &m_perTreeGrowthIteration);

        ImGui::Text("Data export:");
        ImGui::Checkbox("Export OBJ", &m_exportOBJ);
        ImGui::Checkbox("Export Graph", &m_exportGraph);
        ImGui::Checkbox("Export LString", &m_exportLString);
        if(m_exportLString){
            ImGui::DragInt("Character div", &m_targetDivider, 1, 1, 1024);
        }
        ImGui::Text("Rendering export:");
        ImGui::Checkbox("Export Depth", &m_exportDepth);
        ImGui::Checkbox("Export Image", &m_exportImage);
        ImGui::Checkbox("Export Branch Capture", &m_exportBranchCapture);
        ImGui::Checkbox("Export Camera matrices", &m_exportMatrices);
        ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Camera")) {
        EditorManager::DragAndDropButton(m_cameraEntity, "Attached Camera Entity");
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
    auto internodeQuery = EntityManager::GetCurrentScene()->GetSystem<InternodeSystem>()->m_internodesQuery;
    /*
    EntityManager::ForEach<BranchColor, InternodeInfo>(
            EntityManager::GetCurrentScene(), JobManager::PrimaryWorkers(),
            internodeQuery,
            [=](int i, Entity entity, BranchColor &internodeRenderColor,
                InternodeInfo &internodeInfo) {
                internodeRenderColor.m_value = glm::vec4(((entity.GetIndex() / 4096) % 64) / 64.0f,
                                                         ((entity.GetIndex() / 64) % 64) / 64.0f,
                                                         (entity.GetIndex() % 64) / 64.0f, 1.0f);
            },
            true);
    */
    EntityManager::ForEach<GlobalTransform, BranchCylinder, InternodeInfo>(
            EntityManager::GetCurrentScene(), JobManager::PrimaryWorkers(),
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
    internodeQuery.ToComponentDataArray<BranchCylinder>(EntityManager::GetCurrentScene(),
                                                        branchCylinders);
    std::vector<BranchColor> branchColors;
    internodeQuery.ToComponentDataArray<BranchColor>(EntityManager::GetCurrentScene(),
                                                     branchColors);
    std::vector<GlobalTransform> branchGlobalTransforms;
    internodeQuery.ToComponentDataArray<GlobalTransform>(EntityManager::GetCurrentScene(),
                                                         branchGlobalTransforms);
    RenderManager::DrawGizmoMeshInstancedColored(
            DefaultResources::Primitives::Cylinder, m_branchCaptureCamera,
            m_cameraPosition,
            m_cameraRotation,
            *reinterpret_cast<std::vector<glm::vec4> *>(&branchColors),
            *reinterpret_cast<std::vector<glm::mat4> *>(&branchCylinders),
            glm::mat4(1.0f), 1.0f);

    RenderManager::DrawGizmoMeshInstanced(
            DefaultResources::Primitives::Sphere, m_branchCaptureCamera,
            m_cameraPosition,
            m_cameraRotation,
            glm::vec4(1.0f),
            *reinterpret_cast<std::vector<glm::mat4> *>(&branchGlobalTransforms),
            glm::mat4(1.0f), m_nodeSize);
}

void MultipleAngleCapture::OnCreate() {
    m_branchCaptureCamera = SerializationManager::ProduceSerializable<Camera>();
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
        out << YAML::Key << "Nodes" << YAML::Value << YAML::BeginSeq;
        {
            out << YAML::BeginMap;
            out << YAML::Key << "Parent Index" << -1;
            out << YAML::Key << "Index" << m_currentGrowingTree.GetIndex();
            out << YAML::Key << "Transform" << m_currentGrowingTree.GetDataComponent<Transform>().m_value;
            out << YAML::Key << "GlobalTransform" << m_currentGrowingTree.GetDataComponent<GlobalTransform>().m_value;
            auto internodeInfo = m_currentGrowingTree.GetDataComponent<InternodeInfo>();
            out << YAML::Key << "Thickness" << internodeInfo.m_thickness;
            out << YAML::Key << "Length" << internodeInfo.m_length;
            out << YAML::Key << "Internode Index" << internodeInfo.m_index;

            out << YAML::EndMap;
        }
        behaviour->TreeGraphWalkerRootToEnd(m_currentGrowingTree, m_currentGrowingTree,
                                            [&](Entity parent, Entity child) {
                                                if (!behaviour->InternodeCheck(child)) return;
                                                out << YAML::BeginMap;
                                                out << YAML::Key << "Parent Index" << parent.GetIndex();
                                                out << YAML::Key << "Index" << child.GetIndex();
                                                out << YAML::Key << "Transform"
                                                    << child.GetDataComponent<Transform>().m_value;
                                                out << YAML::Key << "GlobalTransform"
                                                    << child.GetDataComponent<GlobalTransform>().m_value;
                                                auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                                out << YAML::Key << "Thickness" << childInternodeInfo.m_thickness;
                                                out << YAML::Key << "Length" << childInternodeInfo.m_length;
                                                out << YAML::Key << "Internode Index" << childInternodeInfo.m_index;
                                                out << YAML::EndMap;
                                            });
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

