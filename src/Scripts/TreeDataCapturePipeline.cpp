//
// Created by lllll on 9/24/2021.
//

#include "TreeDataCapturePipeline.hpp"
#include "Entities.hpp"
#include "PlantLayer.hpp"
#include "ProjectManager.hpp"
#include "LSystemBehaviour.hpp"
#include "CubeVolume.hpp"
#include "Prefab.hpp"

#ifdef RAYTRACERFACILITY

#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"

using namespace RayTracerFacility;
#endif

#include "TransformLayer.hpp"
#include "DefaultInternodeFoliage.hpp"


using namespace Scripts;

void TreeDataCapturePipeline::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
    Entity rootInternode;
    auto scene = pipeline.GetScene();
    auto children = scene->GetChildren(pipeline.m_currentGrowingTree);
    for (const auto &i: children) {
        if (scene->HasPrivateComponent<Internode>(i)) rootInternode = i;
    }
    auto internode = scene->GetOrSetPrivateComponent<Internode>(rootInternode).lock();
    auto root = scene->GetOrSetPrivateComponent<Root>(pipeline.m_currentGrowingTree).lock();
    if (m_applyPhyllotaxis) {
        root->m_plantDescriptor.Get<IPlantDescriptor>()->m_foliagePhyllotaxis = m_foliagePhyllotaxis;
        root->m_plantDescriptor.Get<IPlantDescriptor>()->m_branchTexture = m_branchTexture;
    }

    if (m_exportImage || m_exportDepth || m_exportBranchCapture) {
        SetUpCamera(pipeline);
    }
    pipeline.m_status = AutoTreeGenerationPipelineStatus::Growth;

    if (m_enableRandomObstacle) {
        m_obstacle = scene->CreateEntity("Obstacle");
        float distance = glm::linearRand(glm::min(m_obstacleDistanceRange.x, m_obstacleDistanceRange.y),
                                         glm::max(m_obstacleDistanceRange.x, m_obstacleDistanceRange.y));


        auto angle = glm::linearRand(0.0f, 360.0f);
        if (!m_randomRotation) angle = 0;
        if (m_lShapedWall) {
            float wallY = glm::linearRand(glm::min(m_wallSize.y, m_wallSize.z), glm::max(m_wallSize.y, m_wallSize.z));
            GlobalTransform obstacleGT1, obstacleGT2;
            obstacleGT1.SetValue({glm::cos(glm::radians(angle)) * (distance + m_wallSize.x / 2.0f), 0.0f,
                                  -glm::sin(glm::radians(angle)) * (distance + m_wallSize.x / 2.0f)},
                                 glm::vec3(0, glm::radians(angle), 0),
                                 {m_wallSize.x, wallY, distance + m_wallSize.x / 2.0f});
            obstacleGT2.SetValue({glm::cos(glm::radians(angle + 90.0f)) * (distance + m_wallSize.x / 2.0f), 0.0f,
                                  -glm::sin(glm::radians(angle + 90.0f)) * (distance + m_wallSize.x / 2.0f)},
                                 glm::vec3(0, glm::radians(angle + 90.0f), 0),
                                 {m_wallSize.x, wallY, distance + m_wallSize.x / 2.0f});
            auto wall1 = scene->CreateEntity("Wall1");
            auto wall2 = scene->CreateEntity("Wall2");
            scene->SetDataComponent<GlobalTransform>(wall1, obstacleGT1);
            scene->SetDataComponent<GlobalTransform>(wall2, obstacleGT2);

            auto cubeVolume1 = scene->GetOrSetPrivateComponent<CubeVolume>(wall1).lock();
            cubeVolume1->m_minMaxBound.m_min = glm::vec3(-1.0f);
            cubeVolume1->m_minMaxBound.m_max = glm::vec3(1.0f);
            cubeVolume1->m_asObstacle = true;
            auto cubeVolume2 = scene->GetOrSetPrivateComponent<CubeVolume>(wall2).lock();
            cubeVolume2->m_minMaxBound.m_min = glm::vec3(-1.0f);
            cubeVolume2->m_minMaxBound.m_max = glm::vec3(1.0f);
            cubeVolume2->m_asObstacle = true;
            if (m_renderObstacle) {
                auto obstacleMeshRenderer1 = scene->GetOrSetPrivateComponent<MeshRenderer>(wall1).lock();
                obstacleMeshRenderer1->m_material = ProjectManager::CreateTemporaryAsset<Material>();
                obstacleMeshRenderer1->m_material.Get<Material>()->m_albedoColor = glm::vec3(0.7f);
                obstacleMeshRenderer1->m_mesh = DefaultResources::Primitives::Cube;
                auto obstacleMeshRenderer2 = scene->GetOrSetPrivateComponent<MeshRenderer>(wall2).lock();
                obstacleMeshRenderer2->m_material = ProjectManager::CreateTemporaryAsset<Material>();
                obstacleMeshRenderer2->m_material.Get<Material>()->m_albedoColor = glm::vec3(0.7f);
                obstacleMeshRenderer2->m_mesh = DefaultResources::Primitives::Cube;
            }

            scene->SetParent(wall1, m_obstacle);
            scene->SetParent(wall2, m_obstacle);
        } else {
            GlobalTransform obstacleGT;
            float wallYZ = glm::linearRand(glm::min(m_wallSize.y, m_wallSize.z), glm::max(m_wallSize.y, m_wallSize.z));
            obstacleGT.SetValue({glm::cos(glm::radians(angle)) * (distance + m_wallSize.x / 2.0f), 0.0f,
                                 -glm::sin(glm::radians(angle)) * (distance + m_wallSize.x / 2.0f)},
                                glm::vec3(0, glm::radians(angle), 0),
                                {m_wallSize.x, wallYZ, wallYZ});

            scene->SetDataComponent<GlobalTransform>(m_obstacle, obstacleGT);
            auto cubeVolume = scene->GetOrSetPrivateComponent<CubeVolume>(m_obstacle).lock();
            cubeVolume->m_minMaxBound.m_min = glm::vec3(-1.0f);
            cubeVolume->m_minMaxBound.m_max = glm::vec3(1.0f);
            cubeVolume->m_asObstacle = true;

            if (m_renderObstacle) {
                auto obstacleMeshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(m_obstacle).lock();
                obstacleMeshRenderer->m_material = ProjectManager::CreateTemporaryAsset<Material>();
                obstacleMeshRenderer->m_material.Get<Material>()->m_albedoColor = glm::vec3(0.7f);
                obstacleMeshRenderer->m_mesh = DefaultResources::Primitives::Cube;
            }
        }
    }

    auto prefabPath = pipeline.m_currentDescriptorPath;
    prefabPath.replace_extension(".ueprefab");
    auto prefabABPath = ProjectManager::GetProjectPath().parent_path() / prefabPath;
    if (std::filesystem::exists(prefabABPath)) {
        auto prefab = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset(prefabPath));
        m_prefabEntity = prefab->ToEntity(scene);
    }
}

void TreeDataCapturePipeline::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
    auto scene = pipeline.GetScene();
    auto behaviour = pipeline.GetBehaviour();
#ifdef RAYTRACERFACILITY
    auto camera = scene->GetOrSetPrivateComponent<RayTracerCamera>(pipeline.GetOwner()).lock();
#endif
    auto internodeLayer = Application::GetLayer<PlantLayer>();
    auto behaviourType = pipeline.GetBehaviourType();
    Entity rootInternode;
    auto children = scene->GetChildren(pipeline.m_currentGrowingTree);
    for (const auto &i: children) {
        if (scene->HasPrivateComponent<Internode>(i)) rootInternode = i;
    }
    auto treeIOFolder = m_currentExportFolder;
    auto imagesFolder = m_currentExportFolder / "Image";
    auto objFolder = m_currentExportFolder / "Mesh";
    auto depthFolder = m_currentExportFolder / "Depth";
    auto branchFolder = m_currentExportFolder / "Branch";
    auto graphFolder = m_currentExportFolder / "Graph";
    auto csvFolder = m_currentExportFolder / "CSV";
    auto envGridFolder = m_currentExportFolder / "EnvGrid";
    auto lStringFolder = m_currentExportFolder / "LSystemString";
    auto wallPrefabFolder = m_currentExportFolder / "WallPrefab";
    if (m_enableRandomObstacle && m_exportWallPrefab) {
        std::filesystem::create_directories(wallPrefabFolder);
        auto exportPath = wallPrefabFolder /
                          (pipeline.m_prefix + ".ueprefab");
        auto wallPrefab = ProjectManager::CreateTemporaryAsset<Prefab>();
        wallPrefab->FromEntity(m_obstacle);
        wallPrefab->Export(exportPath);
    }

    if (m_exportOBJ || m_exportImage || m_exportDepth || m_exportBranchCapture) {
        behaviour->GenerateSkinnedMeshes(scene, m_meshGeneratorSettings);
        internodeLayer->UpdateInternodeColors();
    }
    Bound plantBound;
    if (m_exportImage || m_exportDepth || m_exportBranchCapture) {
        scene->ForEachChild(pipeline.m_currentGrowingTree, [&](Entity child) {
            if (!behaviour->InternodeCheck(scene, child)) return;
            plantBound = scene->GetOrSetPrivateComponent<Internode>(child).lock()->CalculateChildrenBound();
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
        auto exportPath = std::filesystem::absolute(csvFolder / (pipeline.m_prefix + ".csv"));
        ExportCSV(pipeline, behaviour, exportPath);
    }
    if (m_enableRandomObstacle && m_exportEnvironmentalGrid) {
        std::filesystem::create_directories(envGridFolder);
        auto exportPath = std::filesystem::absolute(envGridFolder / (pipeline.m_prefix + ".vg"));
        ExportEnvironmentalGrid(pipeline, exportPath);
    }
    if (m_exportLString) {
        std::filesystem::create_directories(lStringFolder);
        auto lString = ProjectManager::CreateTemporaryAsset<LSystemString>();
        scene->GetOrSetPrivateComponent<Internode>(rootInternode).lock()->ExportLString(lString);
        //path here
        lString->Export(
                lStringFolder / (pipeline.m_prefix + ".lstring"));
    }
    if (m_exportTreeIOTrees) {
        std::filesystem::create_directories(treeIOFolder);
        scene->GetOrSetPrivateComponent<Internode>(rootInternode).lock()->ExportTreeIOTree(
                treeIOFolder /
                ("tree" + std::to_string(pipeline.m_descriptorPaths.size() + 1) + ".tree"));
    }
    if (m_exportMatrices) {
        for (float turnAngle = m_turnAngleStart; turnAngle < m_turnAngleEnd; turnAngle += m_turnAngleStep) {
            for (float pitchAngle = m_pitchAngleStart; pitchAngle < m_pitchAngleEnd; pitchAngle += m_pitchAngleStep) {
                auto anglePrefix = std::to_string(pitchAngle) + "_" +
                                   std::to_string(turnAngle);
                auto cameraGlobalTransform = TransformCamera(plantBound, turnAngle, pitchAngle);
                m_cameraModels.push_back(cameraGlobalTransform.m_value);
                m_treeModels.push_back(scene->GetDataComponent<GlobalTransform>(pipeline.m_currentGrowingTree).m_value);
                m_projections.push_back(Camera::m_cameraInfoBlock.m_projection);
                m_views.push_back(Camera::m_cameraInfoBlock.m_view);
                m_names.push_back(pipeline.m_prefix + "_" + anglePrefix);
            }
        }
    }
    if (m_exportOBJ) {
        std::filesystem::create_directories(objFolder);
        Entity foliage, branch;
        scene->ForEachChild(pipeline.m_currentGrowingTree, [&](Entity child) {
            if (scene->GetEntityName(child) == "FoliageMesh") foliage = child;
            else if (scene->GetEntityName(child) == "BranchMesh") branch = child;
        });
        if (scene->IsEntityValid(foliage) && scene->HasPrivateComponent<SkinnedMeshRenderer>(foliage)) {
            auto smr = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(foliage).lock();
            if (smr->m_skinnedMesh.Get<SkinnedMesh>() &&
                !smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
                auto exportPath = objFolder /
                                  (pipeline.m_prefix + "_foliage.obj");
                UNIENGINE_LOG(exportPath.string());
                smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
            }
        }
        if (scene->IsEntityValid(branch) && scene->HasPrivateComponent<SkinnedMeshRenderer>(branch)) {
            auto smr = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(branch).lock();
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
        auto rayTracerCamera = scene->GetOrSetPrivateComponent<RayTracerCamera>(cameraEntity).lock();
        rayTracerCamera->SetOutputType(OutputType::Color);
        for (int turnAngle = m_turnAngleStart; turnAngle < m_turnAngleEnd; turnAngle += m_turnAngleStep) {
            for (int pitchAngle = m_pitchAngleStart; pitchAngle < m_pitchAngleEnd; pitchAngle += m_pitchAngleStep) {
                auto anglePrefix = std::to_string(pitchAngle) + "_" +
                                   std::to_string(turnAngle);
                auto cameraGlobalTransform = TransformCamera(plantBound, turnAngle, pitchAngle);

                scene->SetDataComponent(cameraEntity, cameraGlobalTransform);
                Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
                Application::GetLayer<RayTracerLayer>()->UpdateScene();
                rayTracerCamera->Render(m_rayProperties);
                rayTracerCamera->m_colorTexture->Export(
                        imagesFolder / (pipeline.m_prefix + "_" + anglePrefix + "_rgb.png"));
            }
        }
    }
    if (m_exportDepth) {
        std::filesystem::create_directories(depthFolder);
        auto cameraEntity = pipeline.GetOwner();
        auto rayTracerCamera = scene->GetOrSetPrivateComponent<RayTracerCamera>(cameraEntity).lock();
        rayTracerCamera->SetOutputType(OutputType::Depth);
        rayTracerCamera->SetMaxDistance(m_cameraMax);
        for (int turnAngle = m_turnAngleStart; turnAngle < m_turnAngleEnd; turnAngle += m_turnAngleStep) {
            for (int pitchAngle = m_pitchAngleStart;
                 pitchAngle < m_pitchAngleEnd; pitchAngle += m_pitchAngleStep) {
                auto anglePrefix = std::to_string(pitchAngle) + "_" +
                                   std::to_string(turnAngle);
                auto cameraGlobalTransform = TransformCamera(plantBound, turnAngle, pitchAngle);
                scene->SetDataComponent(cameraEntity, cameraGlobalTransform);
                Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
                Application::GetLayer<RayTracerLayer>()->UpdateScene();
                rayTracerCamera->Render(m_rayProperties);
                rayTracerCamera->m_colorTexture->Export(
                        depthFolder / (pipeline.m_prefix + "_" + anglePrefix + "_depth.hdr"));
            }
        }
    }
    if (m_exportBranchCapture) {
        std::filesystem::create_directories(branchFolder);
        auto cameraEntity = pipeline.GetOwner();
        auto rayTracerCamera = scene->GetOrSetPrivateComponent<RayTracerCamera>(cameraEntity).lock();
        auto rootChildren = scene->GetChildren(pipeline.m_currentGrowingTree);
        for (const auto &child: rootChildren) {
            if (scene->GetEntityName(child) == "FoliageMesh") {
                scene->DeleteEntity(child);
            }
        }
        rayTracerCamera->SetOutputType(OutputType::Color);
        for (int turnAngle = m_turnAngleStart; turnAngle < m_turnAngleEnd; turnAngle += m_turnAngleStep) {
            for (int pitchAngle = m_pitchAngleStart; pitchAngle < m_pitchAngleEnd; pitchAngle += m_pitchAngleStep) {
                auto anglePrefix = std::to_string(pitchAngle) + "_" +
                                   std::to_string(turnAngle);
                auto cameraGlobalTransform = TransformCamera(plantBound, turnAngle, pitchAngle);

                scene->SetDataComponent(cameraEntity, cameraGlobalTransform);
                Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
                Application::GetLayer<RayTracerLayer>()->UpdateScene();

                rayTracerCamera->Render(m_rayProperties);
                rayTracerCamera->m_colorTexture->Export(
                        branchFolder / (pipeline.m_prefix + "_" + anglePrefix + "_branch.png"));
            }
        }
    }
#endif
#pragma endregion

    if (m_enableRandomObstacle)scene->DeleteEntity(m_obstacle);
    if (scene->IsEntityValid(m_prefabEntity)) scene->DeleteEntity(m_prefabEntity);
    pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
}

static const char *DefaultBehaviourTypes[]{"GeneralTree", "LSystem", "SpaceColonization", "TreeGraph"};

void TreeDataCapturePipeline::OnInspect() {
    auto scene = Application::GetActiveScene();
    if (ImGui::Button("Instantiate Pipeline")) {
        auto multipleAngleCapturePipelineEntity = scene->CreateEntity(
                m_self.lock()->GetAssetRecord().lock()->GetAssetFileName());
        auto multipleAngleCapturePipeline = scene->GetOrSetPrivateComponent<AutoTreeGenerationPipeline>(
                multipleAngleCapturePipelineEntity).lock();
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
        ImGui::Checkbox("Enable ground", &m_enableGround);
        Editor::DragAndDropButton<VoxelGrid>(m_obstacleGrid, "Voxel Grid", true);
        ImGui::Checkbox("Random obstacle", &m_enableRandomObstacle);
        if (m_enableRandomObstacle) {
            ImGui::Checkbox("Render obstacle", &m_renderObstacle);
            ImGui::Checkbox("L-Shaped obstacle", &m_lShapedWall);
            ImGui::Checkbox("Random rotation obstacle", &m_randomRotation);
            ImGui::DragFloat2("Obstacle distance (min/max)", &m_obstacleDistanceRange.x, 0.01f);
            ImGui::DragFloat3("Wall size", &m_wallSize.x, 0.01f);
        }
        ImGui::Checkbox("Override phyllotaxis", &m_applyPhyllotaxis);
        if (m_applyPhyllotaxis) {
            Editor::DragAndDropButton<DefaultInternodeFoliage>(m_foliagePhyllotaxis, "Phyllotaxis", true);
            Editor::DragAndDropButton<Texture2D>(m_branchTexture, "Branch texture", true);
        }
        if (ImGui::TreeNodeEx("Export settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Data:");
            if (m_enableRandomObstacle) {
                ImGui::Checkbox("Export Voxel Grid", &m_exportEnvironmentalGrid);
                ImGui::Checkbox("Export Obstacle as Prefab", &m_exportWallPrefab);
            }
            ImGui::Checkbox("Export TreeIO", &m_exportTreeIOTrees);
            ImGui::Checkbox("Export OBJ", &m_exportOBJ);
            ImGui::Checkbox("Export Graph", &m_exportGraph);
            ImGui::Checkbox("Export CSV", &m_exportCSV);
            ImGui::Checkbox("Export LSystemString", &m_exportLString);
            ImGui::Text("Rendering:");
            ImGui::Checkbox("Export Depth", &m_exportDepth);
            ImGui::Checkbox("Export Image", &m_exportImage);
            ImGui::Checkbox("Export Branch Capture", &m_exportBranchCapture);
            ImGui::TreePop();
        }
        ImGui::TreePop();
    }
    m_meshGeneratorSettings.OnInspect();
    if (m_exportDepth || m_exportImage || m_exportBranchCapture) {
        if (ImGui::TreeNodeEx("Camera settings")) {
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

#ifdef RAYTRACERFACILITY
            ImGui::Text("Ray tracer Settings");
            ImGui::DragInt("Bounce", &m_rayProperties.m_bounces);
            ImGui::DragInt("Sample", &m_rayProperties.m_samples);
#endif
            ImGui::TreePop();
        }
        if (m_exportBranchCapture) Application::GetLayer<PlantLayer>()->DrawColorModeSelectionMenu();
    }
}

void TreeDataCapturePipeline::SetUpCamera(AutoTreeGenerationPipeline &pipeline) {
    auto scene = pipeline.GetScene();
    auto cameraEntity = pipeline.GetOwner();
#ifdef RAYTRACERFACILITY
    auto camera = scene->GetOrSetPrivateComponent<RayTracerCamera>(cameraEntity).lock();
    camera->SetFov(m_fov);
    camera->m_allowAutoResize = false;
    camera->m_frameSize = m_resolution;
#endif
    if (scene->HasPrivateComponent<PostProcessing>(cameraEntity)) {
        auto postProcessing = scene->GetOrSetPrivateComponent<PostProcessing>(cameraEntity).lock();
        postProcessing->SetEnabled(false);
    }
}

void TreeDataCapturePipeline::OnCreate() {
}

void TreeDataCapturePipeline::ExportGraph(AutoTreeGenerationPipeline &pipeline,
                                          const std::shared_ptr<IPlantBehaviour> &behaviour,
                                          const std::filesystem::path &path) {
    auto scene = pipeline.GetScene();
    try {
        auto directory = path;
        directory.remove_filename();
        std::filesystem::create_directories(directory);
        YAML::Emitter out;
        out << YAML::BeginMap;
        std::vector<std::vector<std::pair<int, Entity>>> internodes;
        internodes.resize(128);
        internodes[0].emplace_back(-1, pipeline.m_currentGrowingTree);
        scene->ForEachChild(pipeline.m_currentGrowingTree, [&](Entity child) {
            if (!behaviour->InternodeCheck(scene, child)) return;
            behaviour->InternodeGraphWalkerRootToEnd(scene, child,
                                                     [&](Entity parent, Entity child) {
                                                         auto childInternodeInfo = scene->GetDataComponent<InternodeInfo>(
                                                                 child);
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
                ExportGraphNode(pipeline, behaviour, out, instance.first, instance.second);
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

void TreeDataCapturePipeline::ExportGraphNode(AutoTreeGenerationPipeline &pipeline,
                                              const std::shared_ptr<IPlantBehaviour> &behaviour, YAML::Emitter &out,
                                              int parentIndex, const Entity &internode) {
    auto scene = pipeline.GetScene();
    out << YAML::BeginMap;
    out << YAML::Key << "Parent Entity Index" << parentIndex;
    out << YAML::Key << "Entity Index" << internode.GetIndex();

    std::vector<int> indices = {-1, -1, -1};
    scene->ForEachChild(internode, [&](Entity child) {
        if (!behaviour->InternodeCheck(scene, child)) return;
        indices[scene->GetDataComponent<InternodeStatus>(child).m_branchingOrder] = child.GetIndex();
    });

    out << YAML::Key << "Children Entity Indices" << YAML::Key << YAML::BeginSeq;
    for (int i = 0; i < 3; i++) {
        out << YAML::BeginMap;
        out << YAML::Key << "Entity Index" << YAML::Value << indices[i];
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    auto globalTransform = scene->GetDataComponent<GlobalTransform>(internode);
    auto transform = scene->GetDataComponent<Transform>(internode);
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
    auto internodeInfo = scene->GetDataComponent<InternodeInfo>(internode);
    auto internodeStatus = scene->GetDataComponent<InternodeStatus>(internode);
    out << YAML::Key << "Branching Order" << YAML::Value << internodeStatus.m_branchingOrder;
    out << YAML::Key << "Level" << YAML::Value << internodeStatus.m_level;
    out << YAML::Key << "Distance to Root" << YAML::Value << internodeInfo.m_rootDistance;
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

void TreeDataCapturePipeline::ExportMatrices(const std::filesystem::path &path) {
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
TreeDataCapturePipeline::ExportCSV(AutoTreeGenerationPipeline &pipeline,
                                   const std::shared_ptr<IPlantBehaviour> &behaviour,
                                   const std::filesystem::path &path) {
    auto scene = pipeline.GetScene();
    std::ofstream ofs;
    ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (ofs.is_open()) {
        std::string output;
        std::vector<std::vector<std::pair<int, Entity>>> internodes;
        internodes.resize(128);
        scene->ForEachChild(pipeline.m_currentGrowingTree, [&](Entity child) {
            if (!behaviour->InternodeCheck(scene, child)) return;
            internodes[0].emplace_back(-1, child);
            behaviour->InternodeGraphWalkerRootToEnd(scene, child,
                                                     [&](Entity parent, Entity child) {
                                                         auto childInternodeInfo = scene->GetDataComponent<InternodeInfo>(
                                                                 child);
                                                         if (childInternodeInfo.m_endNode) return;
                                                         internodes[childInternodeInfo.m_layer].emplace_back(
                                                                 parent.GetIndex(),
                                                                 child);
                                                     });
        });
        output += "in_id,in_pos_x,in_pos_y,in_pos_z,in_front_x,in_front_y,in_front_z,in_up_x,in_up_y,in_up_z,in_thickness,in_length,in_root_distance,in_chain_distance,in_distance_to_branch_start,in_order,in_quat_x,in_quat_y,in_quat_z,in_quat_w,";
        output += "out0_id,out0_pos_x,out0_pos_y,out0_pos_z,out0_front_x,out0_front_y,out0_front_z,out0_up_x,out0_up_y,out0_up_z,out0_thickness,out0_length,out0_root_distance,out0_chain_distance,out0_distance_to_branch_start,out0_order,out0_quat_x,out0_quat_y,out0_quat_z,out0_quat_w,";
        output += "out1_id,out1_pos_x,out1_pos_y,out1_pos_z,out1_front_x,out1_front_y,out1_front_z,out1_up_x,out1_up_y,out1_up_z,out1_thickness,out1_length,out1_root_distance,out1_chain_distance,out1_distance_to_branch_start,out1_order,out1_quat_x,out1_quat_y,out1_quat_z,out1_quat_w,";
        output += "out2_id,out2_pos_x,out2_pos_y,out2_pos_z,out2_front_x,out2_front_y,out2_front_z,out2_up_x,out2_up_y,out2_up_z,out2_thickness,out2_length,out2_root_distance,out2_chain_distance,out2_distance_to_branch_start,out2_order,out2_quat_x,out2_quat_y,out2_quat_z,out2_quat_w\n";
        int layerIndex = 0;
        for (const auto &layer: internodes) {
            if (layer.empty()) break;
            for (const auto &instance: layer) {
                auto internode = instance.second;
                std::vector<Entity> children;
                children.resize(3);
                bool hasChild = false;
                scene->ForEachChild(internode, [&](Entity child) {
                    if (!behaviour->InternodeCheck(scene, child)) return;
                    if (scene->GetDataComponent<InternodeInfo>(child).m_endNode) return;
                    children[scene->GetDataComponent<InternodeStatus>(child).m_branchingOrder] = child;
                    hasChild = true;
                });
                std::string row;

                auto globalTransform = scene->GetDataComponent<GlobalTransform>(internode);
                auto transform = scene->GetDataComponent<Transform>(internode);

                auto position = globalTransform.GetPosition();
                auto globalRotation = globalTransform.GetRotation();
                auto front = globalRotation * glm::vec3(0, 0, -1);
                auto up = globalRotation * glm::vec3(0, 1, 0);
                auto rotation = transform.GetRotation();
                auto internodeInfo = scene->GetDataComponent<InternodeInfo>(internode);
                auto internodeStatus = scene->GetDataComponent<InternodeStatus>(internode);

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
                row += std::to_string(internodeInfo.m_rootDistance) + ",";
                row += std::to_string(internodeStatus.m_chainDistance) + ",";
                row += std::to_string(internodeStatus.m_branchLength) + ",";
                row += std::to_string(internodeInfo.m_order) + ",";

                row += std::to_string(globalRotation.x) + ",";
                row += std::to_string(globalRotation.y) + ",";
                row += std::to_string(globalRotation.z) + ",";
                row += std::to_string(globalRotation.w) + ",";

                for (int i = 0; i < 3; i++) {
                    auto child = children[i];
                    if (child.GetIndex() == 0 || scene->GetDataComponent<InternodeInfo>(child).m_endNode) {
                        row += "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A";
                    } else {
                        auto globalTransformChild = scene->GetDataComponent<GlobalTransform>(child);
                        auto transformChild = scene->GetDataComponent<Transform>(child);

                        auto positionChild = globalTransformChild.GetPosition();
                        auto globalRotationChild = globalTransformChild.GetRotation();
                        auto frontChild = globalRotationChild * glm::vec3(0, 0, -1);
                        auto upChild = globalRotationChild * glm::vec3(0, 1, 0);
                        auto rotationChildChild = transformChild.GetRotation();
                        auto internodeInfoChild = scene->GetDataComponent<InternodeInfo>(child);
                        auto internodeStatusChild = scene->GetDataComponent<InternodeStatus>(child);

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
                        row += std::to_string(internodeInfoChild.m_rootDistance) + ",";
                        row += std::to_string(internodeStatusChild.m_chainDistance) + ",";
                        row += std::to_string(internodeStatusChild.m_branchLength) + ",";
                        row += std::to_string(internodeInfoChild.m_order) + ",";

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
    } else {
        UNIENGINE_ERROR("Can't open file!");
    }
}

void TreeDataCapturePipeline::OnStart(AutoTreeGenerationPipeline &pipeline) {
    auto scene = pipeline.GetScene();
#ifdef RAYTRACERFACILITY
    auto &environment = Application::GetLayer<RayTracerLayer>()->m_environmentProperties;
    environment.m_environmentalLightingType = EnvironmentalLightingType::SingleLightSource;
    environment.m_sunDirection = glm::quat(glm::radians(glm::vec3(120, 0, 0))) * glm::vec3(0, 0, -1);
    environment.m_lightSize = m_lightSize;
    environment.m_ambientLightIntensity = m_ambientLightIntensity;
    scene->m_environmentSettings.m_ambientLightIntensity = m_envLightIntensity;
#endif

    m_projections.clear();
    m_views.clear();
    m_names.clear();
    m_cameraModels.clear();
    m_treeModels.clear();

    if (m_enableGround) {
        m_ground = scene->CreateEntity("Ground");
        auto groundMeshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(m_ground).lock();
        groundMeshRenderer->m_material = ProjectManager::CreateTemporaryAsset<Material>();
        groundMeshRenderer->m_mesh = DefaultResources::Primitives::Quad;
        GlobalTransform groundGT;
        groundGT.SetScale({1000, 1, 1000});
        scene->SetDataComponent<GlobalTransform>(m_ground, groundGT);
    }

}

void TreeDataCapturePipeline::OnEnd(AutoTreeGenerationPipeline &pipeline) {
    auto scene = pipeline.GetScene();
    if ((m_exportDepth || m_exportImage || m_exportBranchCapture) && m_exportMatrices)
        ExportMatrices(m_currentExportFolder /
                       "matrices.yml");
    if (m_enableGround) scene->DeleteEntity(m_ground);
}

void TreeDataCapturePipeline::DisableAllExport() {
    m_exportTreeIOTrees = false;
    m_exportOBJ = false;
    m_exportCSV = false;
    m_exportEnvironmentalGrid = false;
    m_exportWallPrefab = false;
    m_exportGraph = false;
    m_exportImage = false;
    m_exportDepth = false;
    m_exportMatrices = false;
    m_exportBranchCapture = false;
    m_exportLString = false;
}

GlobalTransform TreeDataCapturePipeline::TransformCamera(const Bound &bound, float turnAngle, float pitchAngle) {
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

void TreeDataCapturePipeline::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_branchTexture);
    list.push_back(m_foliagePhyllotaxis);
    list.push_back(m_obstacleGrid);
}

void TreeDataCapturePipeline::Serialize(YAML::Emitter &out) {
    m_branchTexture.Save("m_branchTexture", out);
    m_foliagePhyllotaxis.Save("m_foliagePhyllotaxis", out);
    m_obstacleGrid.Save("m_obstacleGrid", out);

    m_meshGeneratorSettings.Save("m_meshGeneratorSettings", out);
    out << YAML::Key << "m_enableGround" << YAML::Value << m_enableGround;
    out << YAML::Key << "m_exportEnvironmentalGrid" << YAML::Value << m_exportEnvironmentalGrid;
    out << YAML::Key << "m_enableRandomObstacle" << YAML::Value << m_enableRandomObstacle;
    out << YAML::Key << "m_renderObstacle" << YAML::Value << m_renderObstacle;
    out << YAML::Key << "m_lShapedWall" << YAML::Value << m_lShapedWall;
    out << YAML::Key << "m_obstacleDistanceRange" << YAML::Value << m_obstacleDistanceRange;
    out << YAML::Key << "m_wallSize" << YAML::Value << m_wallSize;
    out << YAML::Key << "m_exportWallPrefab" << YAML::Value << m_exportWallPrefab;
    out << YAML::Key << "m_randomRotation" << YAML::Value << m_randomRotation;
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

void TreeDataCapturePipeline::Deserialize(const YAML::Node &in) {
    m_branchTexture.Load("m_branchTexture", in);
    m_foliagePhyllotaxis.Load("m_foliagePhyllotaxis", in);
    m_obstacleGrid.Load("m_obstacleGrid", in);

    m_meshGeneratorSettings.Load("m_meshGeneratorSettings", in);
    if (in["m_enableGround"]) m_enableGround = in["m_enableGround"].as<bool>();
    if (in["m_randomRotation"]) m_randomRotation = in["m_randomRotation"].as<bool>();
    if (in["m_lShapedWall"]) m_lShapedWall = in["m_lShapedWall"].as<bool>();
    if (in["m_exportWallPrefab"]) m_exportWallPrefab = in["m_exportWallPrefab"].as<bool>();
    if (in["m_exportEnvironmentalGrid"]) m_exportEnvironmentalGrid = in["m_exportEnvironmentalGrid"].as<bool>();
    if (in["m_enableRandomObstacle"]) m_enableRandomObstacle = in["m_enableRandomObstacle"].as<bool>();
    if (in["m_renderObstacle"]) m_renderObstacle = in["m_renderObstacle"].as<bool>();
    if (in["m_obstacleDistanceRange"]) m_obstacleDistanceRange = in["m_obstacleDistanceRange"].as<glm::vec2>();
    if (in["m_wallSize"]) m_wallSize = in["m_wallSize"].as<glm::vec3>();

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

void TreeDataCapturePipeline::ExportEnvironmentalGrid(AutoTreeGenerationPipeline &pipeline,
                                                      const std::filesystem::path &path) {
    auto grid = m_obstacleGrid.Get<VoxelGrid>();
    if (grid) {
        grid->FillObstacle(pipeline.GetScene());
    }
    grid->Export(path);
}

