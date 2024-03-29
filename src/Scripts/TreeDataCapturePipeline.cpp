//
// Created by lllll on 9/24/2021.
//

#include "TreeDataCapturePipeline.hpp"
#include "Entities.hpp"
#include "InternodeLayer.hpp"
#include "ProjectManager.hpp"
#include "LSystemBehaviour.hpp"
#include "CubeVolume.hpp"
#include "Prefab.hpp"
#include <Tinyply.hpp>
#include <unordered_set>

#ifdef RAYTRACERFACILITY

#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"

using namespace RayTracerFacility;
using namespace tinyply;
#endif

#include "TransformLayer.hpp"
#include "DefaultInternodeFoliage.hpp"
#include "RadialBoundingVolume.hpp"

using namespace Scripts;

void TreeDataCapturePipeline::OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) {
	Entity rootInternode;
	auto scene = pipeline.GetScene();

	for (const auto& tree : pipeline.m_currentGrowingTrees) {
		auto children = scene->GetChildren(tree);
		for (const auto& i : children) {
			if (scene->HasPrivateComponent<Internode>(i)) rootInternode = i;
		}
		auto internode = scene->GetOrSetPrivateComponent<Internode>(rootInternode).lock();
		auto root = scene->GetOrSetPrivateComponent<InternodePlant>(tree).lock();
		if (m_appearanceSettings.m_applyPhyllotaxis) {
			root->m_plantDescriptor.Get<IPlantDescriptor>()->m_foliagePhyllotaxis = m_appearanceSettings.m_foliagePhyllotaxis;
			root->m_plantDescriptor.Get<IPlantDescriptor>()->m_branchTexture = m_appearanceSettings.m_branchTexture;
		}
	}
	if (m_exportOptions.m_exportImage || m_exportOptions.m_exportDepth || m_exportOptions.m_exportBranchCapture) {
		SetUpCamera(pipeline);
	}
	pipeline.m_status = AutoTreeGenerationPipelineStatus::Growth;

	if (m_obstacleSettings.m_enableRandomObstacle) {
		m_obstacle = scene->CreateEntity("Obstacle");
		float distance = glm::linearRand(
			glm::min(m_obstacleSettings.m_obstacleDistanceRange.x, m_obstacleSettings.m_obstacleDistanceRange.y),
			glm::max(m_obstacleSettings.m_obstacleDistanceRange.x, m_obstacleSettings.m_obstacleDistanceRange.y));


		auto angle = glm::linearRand(0.0f, 360.0f);
		if (!m_obstacleSettings.m_randomRotation) angle = 0;
		if (m_obstacleSettings.m_lShapedWall) {
			float wallY = glm::linearRand(glm::min(m_obstacleSettings.m_wallSize.y, m_obstacleSettings.m_wallSize.z),
				glm::max(m_obstacleSettings.m_wallSize.y, m_obstacleSettings.m_wallSize.z));
			GlobalTransform obstacleGT1, obstacleGT2;
			obstacleGT1.SetValue(
				{ glm::cos(glm::radians(angle)) * (distance + m_obstacleSettings.m_wallSize.x / 2.0f), 0.0f,
				 -glm::sin(glm::radians(angle)) * (distance + m_obstacleSettings.m_wallSize.x / 2.0f) },
				glm::vec3(0, glm::radians(angle), 0),
				{ m_obstacleSettings.m_wallSize.x, wallY, distance + m_obstacleSettings.m_wallSize.x / 2.0f });
			obstacleGT2.SetValue(
				{ glm::cos(glm::radians(angle + 90.0f)) * (distance + m_obstacleSettings.m_wallSize.x / 2.0f), 0.0f,
				 -glm::sin(glm::radians(angle + 90.0f)) * (distance + m_obstacleSettings.m_wallSize.x / 2.0f) },
				glm::vec3(0, glm::radians(angle + 90.0f), 0),
				{ m_obstacleSettings.m_wallSize.x, wallY, distance + m_obstacleSettings.m_wallSize.x / 2.0f });
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
			if (m_obstacleSettings.m_renderObstacle) {
				auto obstacleMeshRenderer1 = scene->GetOrSetPrivateComponent<MeshRenderer>(wall1).lock();
				obstacleMeshRenderer1->m_material = ProjectManager::CreateTemporaryAsset<Material>();
				obstacleMeshRenderer1->m_material.Get<Material>()->m_materialProperties.m_albedoColor = glm::vec3(0.7f);
				obstacleMeshRenderer1->m_mesh = DefaultResources::Primitives::Cube;
				auto obstacleMeshRenderer2 = scene->GetOrSetPrivateComponent<MeshRenderer>(wall2).lock();
				obstacleMeshRenderer2->m_material = ProjectManager::CreateTemporaryAsset<Material>();
				obstacleMeshRenderer2->m_material.Get<Material>()->m_materialProperties.m_albedoColor = glm::vec3(0.7f);
				obstacleMeshRenderer2->m_mesh = DefaultResources::Primitives::Cube;
			}

			scene->SetParent(wall1, m_obstacle);
			scene->SetParent(wall2, m_obstacle);
		}
		else {
			GlobalTransform obstacleGT;
			float wallYZ = glm::linearRand(glm::min(m_obstacleSettings.m_wallSize.y, m_obstacleSettings.m_wallSize.z),
				glm::max(m_obstacleSettings.m_wallSize.y, m_obstacleSettings.m_wallSize.z));
			obstacleGT.SetValue(
				{ glm::cos(glm::radians(angle)) * (distance + m_obstacleSettings.m_wallSize.x / 2.0f), 0.0f,
				 -glm::sin(glm::radians(angle)) * (distance + m_obstacleSettings.m_wallSize.x / 2.0f) },
				glm::vec3(0, glm::radians(angle), 0),
				{ m_obstacleSettings.m_wallSize.x, wallYZ, wallYZ });

			scene->SetDataComponent<GlobalTransform>(m_obstacle, obstacleGT);
			auto cubeVolume = scene->GetOrSetPrivateComponent<CubeVolume>(m_obstacle).lock();
			cubeVolume->m_minMaxBound.m_min = glm::vec3(-1.0f);
			cubeVolume->m_minMaxBound.m_max = glm::vec3(1.0f);
			cubeVolume->m_asObstacle = true;

			if (m_obstacleSettings.m_renderObstacle) {
				auto obstacleMeshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(m_obstacle).lock();
				obstacleMeshRenderer->m_material = ProjectManager::CreateTemporaryAsset<Material>();
				obstacleMeshRenderer->m_material.Get<Material>()->m_materialProperties.m_albedoColor = glm::vec3(0.7f);
				obstacleMeshRenderer->m_mesh = DefaultResources::Primitives::Cube;
			}
		}
	}

	auto prefabPath = pipeline.m_currentDescriptorPath.m_path;
	prefabPath.replace_extension(".ueprefab");
	auto prefabABPath = ProjectManager::GetProjectPath().parent_path() / prefabPath;
	if (std::filesystem::exists(prefabABPath)) {
		auto prefab = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset(prefabPath));
		m_prefabEntity = prefab->ToEntity(scene);
	}
	if (m_environmentSettings.m_enableGround) scene->SetEnable(m_ground, false);

	if (scene->IsEntityValid(m_volumeEntity.Get())) {
		scene->SetEnable(m_volumeEntity.Get(), true);
	}
}

void TreeDataCapturePipeline::OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) {
	auto scene = pipeline.GetScene();
	auto behaviour = pipeline.GetBehaviour();
	if (m_environmentSettings.m_enableGround) scene->SetEnable(m_ground, true);
	if (scene->IsEntityValid(m_volumeEntity.Get())) {
		scene->SetEnable(m_volumeEntity.Get(), false);
	}
#ifdef RAYTRACERFACILITY
	auto camera = scene->GetOrSetPrivateComponent<RayTracerCamera>(pipeline.GetOwner()).lock();
#endif
	auto internodeLayer = Application::GetLayer<InternodeLayer>();
	auto behaviourType = pipeline.GetBehaviourType();
	std::vector<Entity> rootInternodes;
	for (const auto& tree : pipeline.m_currentGrowingTrees) {
		auto children = scene->GetChildren(tree);
		for (const auto& i : children) {
			if (scene->HasPrivateComponent<Internode>(i)) rootInternodes.emplace_back(i);
		}
	}
	auto treeIOFolder = m_currentExportFolder / "TreeIO";
	auto imagesFolder = m_currentExportFolder / "Image";
	auto objFolder = m_currentExportFolder / "Mesh";
	auto rbvFolder = m_currentExportFolder / "RBV";
	auto maskFolder = m_currentExportFolder / "Mask";
	auto depthFolder = m_currentExportFolder / "Depth";
	auto branchFolder = m_currentExportFolder / "Branch";
	auto graphFolder = m_currentExportFolder / "Graph";
	auto csvFolder = m_currentExportFolder / "CSV";
	auto envGridFolder = m_currentExportFolder / "EnvGrid";
	auto lStringFolder = m_currentExportFolder / "LSystemString";
	auto wallPrefabFolder = m_currentExportFolder / "WallPrefab";
	auto pointCloudFolder = m_currentExportFolder / "PointCloud";
	auto junctionFolder = m_currentExportFolder / "Junction";
	if (m_obstacleSettings.m_enableRandomObstacle && m_exportOptions.m_exportWallPrefab) {
		std::filesystem::create_directories(wallPrefabFolder);
		auto exportPath = wallPrefabFolder /
			(pipeline.m_prefix + ".ueprefab");
		auto wallPrefab = ProjectManager::CreateTemporaryAsset<Prefab>();
		wallPrefab->FromEntity(m_obstacle);
		wallPrefab->Export(exportPath);
	}

	if (m_exportOptions.m_exportOBJ || m_exportOptions.m_exportImage || m_exportOptions.m_exportDepth ||
		m_exportOptions.m_exportMask ||
		m_exportOptions.m_exportBranchCapture || m_exportOptions.m_exportPointCloud) {
		auto settings = m_meshGeneratorSettings;
		if (m_exportOptions.m_exportMask) {
			settings.m_growthDirection = false;
			settings.m_vertexColorOnly = true;
			settings.m_overrideVertexColor = true;
			settings.m_foliageVertexColor = glm::vec3(0, 1, 0);
			settings.m_branchVertexColor = glm::vec3(0.5f, 0.3f, 0.0f);
		}
		else if (m_pointCloudPointSettings.m_junctionIndex) {
			settings.m_growthDirection = true;
			settings.m_vertexColorOnly = true;
		}
		behaviour->GenerateSkinnedMeshes(scene, settings);
		internodeLayer->UpdateInternodeColors();
	}
	std::vector <Bound> plantBounds;
	if (m_exportOptions.m_exportImage || m_exportOptions.m_exportDepth || m_exportOptions.m_exportBranchCapture ||
		m_exportOptions.m_exportPointCloud) {
		for (const auto& tree : pipeline.m_currentGrowingTrees) {
			scene->ForEachChild(tree, [&](Entity child) {
				if (!behaviour->InternodeCheck(scene, child)) return;
			plantBounds.emplace_back(scene->GetOrSetPrivateComponent<Internode>(child).lock()->CalculateChildrenBound());
				});
		}
	}
#pragma region Export
	if (pipeline.m_behaviourType == BehaviourType::GeneralTree && m_exportOptions.m_exportGraph) {
		std::filesystem::create_directories(graphFolder);
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			auto exportPath = graphFolder /
				(pipeline.m_prefix + +"[" + std::to_string(treeIndex) + "].yml");
			ExportGraph(pipeline, behaviour, exportPath, treeIndex);
		}
	}
	if (pipeline.m_behaviourType == BehaviourType::GeneralTree && m_exportOptions.m_exportCSV) {
		std::filesystem::create_directories(csvFolder);
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			auto exportPath = std::filesystem::absolute(csvFolder / (pipeline.m_prefix + +"[" + std::to_string(treeIndex) + "].csv"));
			ExportCSV(pipeline, behaviour, exportPath, treeIndex);
		}
	}
	if (m_obstacleSettings.m_enableRandomObstacle && m_exportOptions.m_exportEnvironmentalGrid) {
		std::filesystem::create_directories(envGridFolder);
		auto exportPath = std::filesystem::absolute(envGridFolder / (pipeline.m_prefix + ".vg"));
		ExportEnvironmentalGrid(pipeline, exportPath);
	}
	if (m_exportOptions.m_exportLString) {
		std::filesystem::create_directories(lStringFolder);

		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			auto lString = ProjectManager::CreateTemporaryAsset<LSystemString>();
			scene->GetOrSetPrivateComponent<Internode>(rootInternodes[treeIndex]).lock()->ExportLString(lString);
			//path here
			lString->Export(
				lStringFolder / (pipeline.m_prefix + "[" + std::to_string(treeIndex) + "].lstring"));
		}
	}

	if (m_exportOptions.m_exportRBV) {
		std::filesystem::create_directories(rbvFolder);
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			auto rbv = scene->GetOrSetPrivateComponent<RadialBoundingVolume>(pipeline.m_currentGrowingTrees[treeIndex]).lock();
			rbv->m_rootInternode = scene->GetOrSetPrivateComponent<Internode>(rootInternodes[treeIndex]).lock();
			rbv->CalculateVolume();
			rbv->GenerateMesh();
			{
				const std::string data = rbv->Save();
				std::ofstream ofs;
				auto rbvPath = rbvFolder /
					(pipeline.m_prefix + ".rbv");
				ofs.open(rbvPath.string().c_str(),
					std::ofstream::out | std::ofstream::trunc);
				ofs.write(data.c_str(), data.length());
				ofs.flush();
				ofs.close();
			}

			auto objPath = rbvFolder /
				(pipeline.m_prefix + "[" + std::to_string(treeIndex) + "]" + "_rbv");
			rbv->ExportAsObj(objPath.string());
		}
	}

	if (m_exportOptions.m_exportTreeIOTrees) {
		std::filesystem::create_directories(treeIOFolder);
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			auto name = "tree" + std::to_string(pipeline.m_descriptorPaths.size() + 1);
			scene->GetOrSetPrivateComponent<Internode>(rootInternodes[treeIndex]).lock()->ExportTreeIOTree(
				treeIOFolder /
				(pipeline.m_prefix + ".tree"));
			//m_treeIOPairs.emplace_back(pipeline.m_prefix, name);
		}
	}
	if ((m_exportOptions.m_exportImage || m_exportOptions.m_exportDepth || m_exportOptions.m_exportBranchCapture ||
		m_exportOptions.m_exportMask) &&
		m_exportOptions.m_exportMatrices) {
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			for (float turnAngle = m_cameraSettings.m_turnAngleStart;
				turnAngle < m_cameraSettings.m_turnAngleEnd; turnAngle += m_cameraSettings.m_turnAngleStep) {
				for (float pitchAngle = m_cameraSettings.m_pitchAngleStart;
					pitchAngle < m_cameraSettings.m_pitchAngleEnd; pitchAngle += m_cameraSettings.m_pitchAngleStep) {
					auto anglePrefix = std::to_string(pitchAngle) + "_" +
						std::to_string(turnAngle);
					auto cameraGlobalTransform = m_cameraSettings.GetTransform(true, plantBounds[treeIndex], turnAngle, pitchAngle);
					m_cameraModels.push_back(cameraGlobalTransform.m_value);
					m_treeModels.push_back(scene->GetDataComponent<GlobalTransform>(pipeline.m_currentGrowingTrees[treeIndex]).m_value);
					m_projections.push_back(Camera::m_cameraInfoBlock.m_projection);
					m_views.push_back(Camera::m_cameraInfoBlock.m_view);
					m_names.push_back(pipeline.m_prefix + "_" + anglePrefix + "[" + std::to_string(treeIndex) + "]");
				}
			}
		}
	}
	if (m_exportOptions.m_exportOBJ) {
		std::filesystem::create_directories(objFolder);
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			Entity foliage, branch;
			scene->ForEachChild(pipeline.m_currentGrowingTrees[treeIndex], [&](Entity child) {
				if (scene->GetEntityName(child) == "FoliageMesh") foliage = child;
				else if (scene->GetEntityName(child) == "BranchMesh") branch = child;
				});
			if (scene->IsEntityValid(foliage) && scene->HasPrivateComponent<SkinnedMeshRenderer>(foliage)) {
				auto smr = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(foliage).lock();
				if (smr->m_skinnedMesh.Get<SkinnedMesh>() &&
					!smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
					auto exportPath = objFolder /
						(pipeline.m_prefix + "[" + std::to_string(treeIndex) + "]" + "_foliage.obj");
					UNIENGINE_LOG(exportPath.string());
					smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
				}
			}
			if (scene->IsEntityValid(branch) && scene->HasPrivateComponent<SkinnedMeshRenderer>(branch)) {
				auto smr = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(branch).lock();
				if (smr->m_skinnedMesh.Get<SkinnedMesh>() &&
					!smr->m_skinnedMesh.Get<SkinnedMesh>()->UnsafeGetSkinnedVertices().empty()) {
					auto exportPath = objFolder /
						(pipeline.m_prefix + "[" + std::to_string(treeIndex) + "]" + "_branch.obj");
					smr->m_skinnedMesh.Get<SkinnedMesh>()->Export(exportPath);
				}
			}
		}
	}

#ifdef RAYTRACERFACILITY
	if (m_exportOptions.m_exportMask) {
		std::filesystem::create_directories(maskFolder);
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			auto cameraEntity = pipeline.GetOwner();
			auto rayTracerCamera = scene->GetOrSetPrivateComponent<RayTracerCamera>(cameraEntity).lock();
			rayTracerCamera->SetOutputType(OutputType::Albedo);
			auto rootChildren = scene->GetChildren(pipeline.m_currentGrowingTrees[treeIndex]);
			for (const auto& child : rootChildren) {
				if (scene->GetEntityName(child) == "FoliageMesh") {
					scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(
						child).lock()->m_material.Get<Material>()->m_vertexColorOnly = true;
				}
				if (scene->GetEntityName(child) == "BranchMesh") {
					scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(
						child).lock()->m_material.Get<Material>()->m_vertexColorOnly = true;
				}
			}
			for (int turnAngle = m_cameraSettings.m_turnAngleStart;
				turnAngle < m_cameraSettings.m_turnAngleEnd; turnAngle += m_cameraSettings.m_turnAngleStep) {
				for (int pitchAngle = m_cameraSettings.m_pitchAngleStart;
					pitchAngle < m_cameraSettings.m_pitchAngleEnd; pitchAngle += m_cameraSettings.m_pitchAngleStep) {
					auto anglePrefix = std::to_string(pitchAngle) + "_" +
						std::to_string(turnAngle);
					auto cameraGlobalTransform = m_cameraSettings.GetTransform(true, plantBounds[treeIndex], turnAngle, pitchAngle);

					scene->SetDataComponent(cameraEntity, cameraGlobalTransform);
					Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
					Application::GetLayer<RayTracerLayer>()->UpdateScene();
					rayTracerCamera->Render(m_cameraSettings.m_rayProperties);
					rayTracerCamera->m_colorTexture->Export(
						maskFolder / (pipeline.m_prefix + "_" + anglePrefix + "[" + std::to_string(treeIndex) + "]" + "_mask.png"));
				}
			}
		}
	}

	if (m_exportOptions.m_exportImage) {
		std::filesystem::create_directories(imagesFolder);
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			auto cameraEntity = pipeline.GetOwner();
			auto rayTracerCamera = scene->GetOrSetPrivateComponent<RayTracerCamera>(cameraEntity).lock();
			rayTracerCamera->SetOutputType(OutputType::Color);
			auto rootChildren = scene->GetChildren(pipeline.m_currentGrowingTrees[treeIndex]);
			for (const auto& child : rootChildren) {
				if (scene->GetEntityName(child) == "FoliageMesh") {
					scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(
						child).lock()->m_material.Get<Material>()->m_vertexColorOnly = false;
				}
				if (scene->GetEntityName(child) == "BranchMesh") {
					scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(
						child).lock()->m_material.Get<Material>()->m_vertexColorOnly = false;
				}
			}

			for (int turnAngle = m_cameraSettings.m_turnAngleStart;
				turnAngle < m_cameraSettings.m_turnAngleEnd; turnAngle += m_cameraSettings.m_turnAngleStep) {
				for (int pitchAngle = m_cameraSettings.m_pitchAngleStart;
					pitchAngle < m_cameraSettings.m_pitchAngleEnd; pitchAngle += m_cameraSettings.m_pitchAngleStep) {
					auto anglePrefix = std::to_string(pitchAngle) + "_" +
						std::to_string(turnAngle);
					auto cameraGlobalTransform = m_cameraSettings.GetTransform(true, plantBounds[treeIndex], turnAngle, pitchAngle);

					scene->SetDataComponent(cameraEntity, cameraGlobalTransform);
					Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
					Application::GetLayer<RayTracerLayer>()->UpdateScene();
					rayTracerCamera->Render(m_cameraSettings.m_rayProperties);
					rayTracerCamera->m_colorTexture->Export(
						imagesFolder / (pipeline.m_prefix + "_" + anglePrefix + "[" + std::to_string(treeIndex) + "]" + "_rgb.png"));
				}
			}
		}
	}


	if (m_exportOptions.m_exportDepth) {
		std::filesystem::create_directories(depthFolder);
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			auto cameraEntity = pipeline.GetOwner();
			auto rayTracerCamera = scene->GetOrSetPrivateComponent<RayTracerCamera>(cameraEntity).lock();
			rayTracerCamera->SetOutputType(OutputType::Depth);
			rayTracerCamera->SetMaxDistance(m_cameraSettings.m_cameraDepthMax);
			for (int turnAngle = m_cameraSettings.m_turnAngleStart;
				turnAngle < m_cameraSettings.m_turnAngleEnd; turnAngle += m_cameraSettings.m_turnAngleStep) {
				for (int pitchAngle = m_cameraSettings.m_pitchAngleStart;
					pitchAngle < m_cameraSettings.m_pitchAngleEnd; pitchAngle += m_cameraSettings.m_pitchAngleStep) {
					auto anglePrefix = std::to_string(pitchAngle) + "_" +
						std::to_string(turnAngle);
					auto cameraGlobalTransform = m_cameraSettings.GetTransform(true, plantBounds[treeIndex], turnAngle, pitchAngle);
					scene->SetDataComponent(cameraEntity, cameraGlobalTransform);
					Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
					Application::GetLayer<RayTracerLayer>()->UpdateScene();
					rayTracerCamera->Render(m_cameraSettings.m_rayProperties);
					rayTracerCamera->m_colorTexture->Export(
						depthFolder / (pipeline.m_prefix + "_" + anglePrefix + "[" + std::to_string(treeIndex) + "]" + "_depth.hdr"));
				}
			}
		}
	}

	if (m_exportOptions.m_exportPointCloud) {
		std::filesystem::create_directories(pointCloudFolder);
		Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
		Application::GetLayer<RayTracerLayer>()->UpdateScene();
		Bound combinedBound;
		combinedBound.m_min = glm::vec3(FLT_MAX);
		combinedBound.m_max = glm::vec3(FLT_MIN);
		for (const auto& plantBound : plantBounds)
		{
			combinedBound.m_min.x = glm::min(combinedBound.m_min.x, plantBound.m_min.x);
			combinedBound.m_min.y = glm::min(combinedBound.m_min.y, plantBound.m_min.y);
			combinedBound.m_min.z = glm::min(combinedBound.m_min.z, plantBound.m_min.z);
			combinedBound.m_max.x = glm::max(combinedBound.m_max.x, plantBound.m_max.x);
			combinedBound.m_max.y = glm::max(combinedBound.m_max.y, plantBound.m_max.y);
			combinedBound.m_max.z = glm::max(combinedBound.m_max.z, plantBound.m_max.z);
		}
		ScanPointCloudLabeled(combinedBound, pipeline,
			pointCloudFolder / (pipeline.m_prefix + ".ply"));

		if (m_pointCloudPointSettings.m_junctionIndex) {
			std::filesystem::create_directories(junctionFolder);
			ExportJunction(pipeline, behaviour, junctionFolder / (pipeline.m_prefix + ".txt"));
		}
	}

	if (m_exportOptions.m_exportBranchCapture) {
		std::filesystem::create_directories(branchFolder);
		for (int treeIndex = 0; treeIndex < rootInternodes.size(); treeIndex++)
		{
			auto cameraEntity = pipeline.GetOwner();
			auto rayTracerCamera = scene->GetOrSetPrivateComponent<RayTracerCamera>(cameraEntity).lock();
			auto rootChildren = scene->GetChildren(pipeline.m_currentGrowingTrees[treeIndex]);
			for (const auto& child : rootChildren) {
				if (scene->GetEntityName(child) == "FoliageMesh") {
					scene->DeleteEntity(child);
				}
			}
			rayTracerCamera->SetOutputType(OutputType::Color);
			for (int turnAngle = m_cameraSettings.m_turnAngleStart;
				turnAngle < m_cameraSettings.m_turnAngleEnd; turnAngle += m_cameraSettings.m_turnAngleStep) {
				for (int pitchAngle = m_cameraSettings.m_pitchAngleStart;
					pitchAngle < m_cameraSettings.m_pitchAngleEnd; pitchAngle += m_cameraSettings.m_pitchAngleStep) {
					auto anglePrefix = std::to_string(pitchAngle) + "_" +
						std::to_string(turnAngle);
					auto cameraGlobalTransform = m_cameraSettings.GetTransform(true, plantBounds[treeIndex], turnAngle, pitchAngle);

					scene->SetDataComponent(cameraEntity, cameraGlobalTransform);
					Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
					Application::GetLayer<RayTracerLayer>()->UpdateScene();

					rayTracerCamera->Render(m_cameraSettings.m_rayProperties);
					rayTracerCamera->m_colorTexture->Export(
						branchFolder / (pipeline.m_prefix + "_" + anglePrefix + "_branch.png"));
				}
			}
		}
	}
#endif
#pragma endregion

	if (m_obstacleSettings.m_enableRandomObstacle)scene->DeleteEntity(m_obstacle);
	if (scene->IsEntityValid(m_prefabEntity)) scene->DeleteEntity(m_prefabEntity);
	pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
}

static const char* DefaultBehaviourTypes[]{ "GeneralTree", "LSystem", "SpaceColonization", "TreeGraph" };

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
	int behaviourType = (int)m_defaultBehaviourType;
	if (ImGui::Combo(
		"Plant behaviour type",
		&behaviourType,
		DefaultBehaviourTypes,
		IM_ARRAYSIZE(DefaultBehaviourTypes))) {
		m_defaultBehaviourType = (BehaviourType)behaviourType;
	}
	ImGui::Text("Current output folder: %s", m_currentExportFolder.string().c_str());
	FileUtils::OpenFolder("Choose output folder...", [&](const std::filesystem::path& path) {
		m_currentExportFolder = std::filesystem::absolute(path);
		}, false);
	if (ImGui::TreeNodeEx("Pipeline Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Multiple trees", &m_enableMultipleTrees);
		if (m_enableMultipleTrees) {
			if (ImGui::TreeNodeEx("Tree Positions", ImGuiTreeNodeFlags_DefaultOpen)) {
				if (ImGui::Button("New position")) {
					m_treePositions.emplace_back();
				}
				for (int i = 0; i < m_treePositions.size(); i++)
				{
					if (ImGui::TreeNodeEx(("No." + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen))
					{
						if (m_treePositions.size() > 1 && ImGui::Button("Remove"))
						{
							m_treePositions.erase(m_treePositions.begin() + i);
							ImGui::TreePop();
							continue;
						}
						ImGui::DragFloat3("Position", &m_treePositions[i].x, 0.01f);
						ImGui::TreePop();
					}
				}
				ImGui::TreePop();
			}
		}
		Editor::DragAndDropButton(m_volumeEntity, "Volume");
		ImGui::Checkbox("Enable ground", &m_environmentSettings.m_enableGround);
		Editor::DragAndDropButton<VoxelGrid>(m_obstacleGrid, "Voxel Grid", true);
		ImGui::Checkbox("Random obstacle", &m_obstacleSettings.m_enableRandomObstacle);
		if (m_obstacleSettings.m_enableRandomObstacle) {
			ImGui::Checkbox("Render obstacle", &m_obstacleSettings.m_renderObstacle);
			ImGui::Checkbox("L-Shaped obstacle", &m_obstacleSettings.m_lShapedWall);
			ImGui::Checkbox("Random rotation obstacle", &m_obstacleSettings.m_randomRotation);
			ImGui::DragFloat2("Obstacle distance (min/max)", &m_obstacleSettings.m_obstacleDistanceRange.x, 0.01f);
			ImGui::DragFloat3("Wall size", &m_obstacleSettings.m_wallSize.x, 0.01f);
		}
		ImGui::Checkbox("Override phyllotaxis", &m_appearanceSettings.m_applyPhyllotaxis);
		if (m_appearanceSettings.m_applyPhyllotaxis) {
			Editor::DragAndDropButton<DefaultInternodeFoliage>(m_appearanceSettings.m_foliagePhyllotaxis, "Phyllotaxis",
				true);
			Editor::DragAndDropButton<Texture2D>(m_appearanceSettings.m_branchTexture, "Branch texture", true);
		}
		if (ImGui::TreeNodeEx("Export settings", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Data:");
			if (m_obstacleSettings.m_enableRandomObstacle) {
				ImGui::Checkbox("Export Voxel Grid", &m_exportOptions.m_exportEnvironmentalGrid);
				ImGui::Checkbox("Export Obstacle as Prefab", &m_exportOptions.m_exportWallPrefab);
			}
			ImGui::Checkbox("Export TreeIO", &m_exportOptions.m_exportTreeIOTrees);
			ImGui::Checkbox("Export OBJ", &m_exportOptions.m_exportOBJ);
			ImGui::Checkbox("Export RBV", &m_exportOptions.m_exportRBV);
			ImGui::Checkbox("Export Graph", &m_exportOptions.m_exportGraph);
			ImGui::Checkbox("Export CSV", &m_exportOptions.m_exportCSV);
			ImGui::Checkbox("Export LSystemString", &m_exportOptions.m_exportLString);
			ImGui::Text("Rendering:");
			ImGui::Checkbox("Export Depth", &m_exportOptions.m_exportDepth);
			ImGui::Checkbox("Export Mask", &m_exportOptions.m_exportMask);
			ImGui::Checkbox("Export Image", &m_exportOptions.m_exportImage);
			ImGui::Checkbox("Export Branch Capture", &m_exportOptions.m_exportBranchCapture);
			ImGui::Checkbox("Export PointCloud", &m_exportOptions.m_exportPointCloud);
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}
	m_meshGeneratorSettings.OnInspect();
	if (m_exportOptions.m_exportDepth || m_exportOptions.m_exportImage || m_exportOptions.m_exportBranchCapture) {
		if (ImGui::TreeNodeEx("Camera settings")) {
			m_cameraSettings.OnInspect();
			ImGui::TreePop();
		}
		ImGui::Checkbox("Export Camera matrices", &m_exportOptions.m_exportMatrices);
		ImGui::DragFloat("Light Size", &m_environmentSettings.m_lightSize, 0.001f);
		ImGui::DragFloat("Ambient light intensity", &m_environmentSettings.m_ambientLightIntensity, 0.01f);
		ImGui::DragFloat("Environment light intensity", &m_environmentSettings.m_envLightIntensity, 0.01f);
		if (m_exportOptions.m_exportBranchCapture) Application::GetLayer<InternodeLayer>()->DrawColorModeSelectionMenu();
	}
	if (m_exportOptions.m_exportPointCloud) {
		if (ImGui::TreeNodeEx("Point cloud settings")) {
			m_pointCloudSettings.OnInspect();
			ImGui::Checkbox("Type", &m_pointCloudPointSettings.m_pointType);

			ImGui::Checkbox("Growth Direction", &m_pointCloudPointSettings.m_growthDirection);
			ImGui::Checkbox("Junction Index", &m_pointCloudPointSettings.m_junctionIndex);
			ImGui::Checkbox("Internode Index", &m_pointCloudPointSettings.m_internodeIndex);
			ImGui::Checkbox("Branch Index", &m_pointCloudPointSettings.m_branchIndex);
			ImGui::DragFloat("Point variance", &m_pointCloudPointSettings.m_variance, 0.1f, 0.0f, 100.0f);
			ImGui::DragFloat("Point ball rand", &m_pointCloudPointSettings.m_ballRandRadius, 0.1f, 0.0f, 100.0f);
			ImGui::DragFloat("Bounding box offset", &m_pointCloudPointSettings.m_boundingBoxOffset, 0.1f, 0.0f, 100.0f);
			ImGui::TreePop();
		}
	}
}

void TreeDataCapturePipeline::SetUpCamera(AutoTreeGenerationPipeline& pipeline) {
	auto scene = pipeline.GetScene();
	auto cameraEntity = pipeline.GetOwner();
#ifdef RAYTRACERFACILITY
	auto camera = scene->GetOrSetPrivateComponent<RayTracerCamera>(cameraEntity).lock();
	camera->SetFov(m_cameraSettings.m_fov);
	camera->m_allowAutoResize = false;
	camera->m_frameSize = m_cameraSettings.m_resolution;
#endif
	if (scene->HasPrivateComponent<PostProcessing>(cameraEntity)) {
		auto postProcessing = scene->GetOrSetPrivateComponent<PostProcessing>(cameraEntity).lock();
		postProcessing->SetEnabled(false);
	}
}

void TreeDataCapturePipeline::OnCreate() {
}

void TreeDataCapturePipeline::ExportGraph(AutoTreeGenerationPipeline& pipeline,
	const std::shared_ptr<IPlantBehaviour>& behaviour,
	const std::filesystem::path& path, int treeIndex) {
	auto scene = pipeline.GetScene();
	try {
		auto directory = path;
		directory.remove_filename();
		std::filesystem::create_directories(directory);
		YAML::Emitter out;

		std::vector<std::vector<std::pair<int, Entity>>> internodes;
		internodes.resize(128);
		scene->ForEachChild(pipeline.m_currentGrowingTrees[treeIndex], [&](Entity root) {
			if (!behaviour->InternodeCheck(scene, root)) return;
		internodes[0].emplace_back(-1, root);
		behaviour->InternodeGraphWalkerRootToEnd(scene, root,
			[&](Entity parent, Entity child) {
				auto childInternodeInfo = scene->GetDataComponent<InternodeInfo>(
					child);
		internodes[childInternodeInfo.m_layer].emplace_back(
			parent.GetIndex(),
			child);
			});
			});
		out << YAML::BeginMap;
		{
			out << YAML::Key << "Layers" << YAML::Value << YAML::BeginSeq;
			{
				int layerIndex = 0;
				for (const auto& layer : internodes) {
					if (layer.empty()) break;
					out << YAML::BeginMap;
					out << YAML::Key << "Layer Index" << YAML::Value << layerIndex;
					out << YAML::Key << "Nodes" << YAML::Value << YAML::BeginSeq;
					for (const auto& instance : layer) {
						ExportGraphNode(pipeline, behaviour, out, instance.first, instance.second);
					}
					out << YAML::EndSeq;
					out << YAML::EndMap;
					layerIndex++;
				}
			}
			out << YAML::EndSeq;
		}
		out << YAML::EndMap;
		std::ofstream fout(path.string());
		fout << out.c_str();
		fout.flush();
		fout.close();
	}
	catch (std::exception e) {
		UNIENGINE_ERROR("Failed to save!");
	}

}

void TreeDataCapturePipeline::ExportGraphNode(AutoTreeGenerationPipeline& pipeline,
	const std::shared_ptr<IPlantBehaviour>& behaviour, YAML::Emitter& out,
	int parentIndex, const Entity& internode) {
	auto scene = pipeline.GetScene();
	out << YAML::BeginMap;
	out << YAML::Key << "Parent Entity Index" << parentIndex;
	out << YAML::Key << "Entity Index" << internode.GetIndex();

	std::vector<int> indices = { -1, -1, -1 };
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
}

void TreeDataCapturePipeline::ExportMatrices(const std::filesystem::path& path) {
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
TreeDataCapturePipeline::ExportCSV(AutoTreeGenerationPipeline& pipeline,
	const std::shared_ptr<IPlantBehaviour>& behaviour,
	const std::filesystem::path& path, int treeIndex) {
	auto scene = pipeline.GetScene();
	std::ofstream ofs;
	ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
	if (ofs.is_open()) {
		std::string output;
		std::vector<std::vector<std::pair<int, Entity>>> internodes;
		internodes.resize(128);
		scene->ForEachChild(pipeline.m_currentGrowingTrees[treeIndex], [&](Entity root) {
			if (!behaviour->InternodeCheck(scene, root)) return;
		internodes[0].emplace_back(-1, root);
		behaviour->InternodeGraphWalkerRootToEnd(scene, root,
			[&](Entity parent, Entity child) {
				auto childInternodeInfo = scene->GetDataComponent<InternodeInfo>(
					child);
		if (childInternodeInfo.m_endNode) return;
		internodes[childInternodeInfo.m_layer].emplace_back(
			parent.GetIndex(),
			child);
			});
			});
		output += "in_id,in_pos_x,in_pos_y,in_pos_z,in_front_x,in_front_y,in_front_z,in_up_x,in_up_y,in_up_z,in_thickness,in_length,in_root_distance,in_chain_distance,in_distance_to_branch_start,in_order,in_quat_x,in_quat_y,in_quat_z,in_quat_w,gravitropism,age,";
		output += "out0_id,out0_pos_x,out0_pos_y,out0_pos_z,out0_front_x,out0_front_y,out0_front_z,out0_up_x,out0_up_y,out0_up_z,out0_thickness,out0_length,out0_root_distance,out0_chain_distance,out0_distance_to_branch_start,out0_order,out0_quat_x,out0_quat_y,out0_quat_z,out0_quat_w,";
		output += "out1_id,out1_pos_x,out1_pos_y,out1_pos_z,out1_front_x,out1_front_y,out1_front_z,out1_up_x,out1_up_y,out1_up_z,out1_thickness,out1_length,out1_root_distance,out1_chain_distance,out1_distance_to_branch_start,out1_order,out1_quat_x,out1_quat_y,out1_quat_z,out1_quat_w,";
		output += "out2_id,out2_pos_x,out2_pos_y,out2_pos_z,out2_front_x,out2_front_y,out2_front_z,out2_up_x,out2_up_y,out2_up_z,out2_thickness,out2_length,out2_root_distance,out2_chain_distance,out2_distance_to_branch_start,out2_order,out2_quat_x,out2_quat_y,out2_quat_z,out2_quat_w\n";
		int layerIndex = 0;
		for (const auto& layer : internodes) {
			if (layer.empty()) break;
			for (const auto& instance : layer) {
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

				row += std::to_string(internodeStatus.m_gravitropism) + ",";
				row += std::to_string(internodeStatus.m_treeAge) + ",";
				for (int i = 0; i < 3; i++) {
					auto child = children[i];
					if (child.GetIndex() == 0 || scene->GetDataComponent<InternodeInfo>(child).m_endNode) {
						row += "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A";
					}
					else {
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
					if (i == 2) {
						row += "\n";
					}
					else row += ",";
				}
				output += row;
			}
			layerIndex++;
		}
		ofs.write(output.c_str(), output.size());
		ofs.flush();
		ofs.close();
	}
	else {
		UNIENGINE_ERROR("Can't open file!");
	}
}

void TreeDataCapturePipeline::OnStart(AutoTreeGenerationPipeline& pipeline) {
	auto scene = pipeline.GetScene();
#ifdef RAYTRACERFACILITY
	auto& environment = Application::GetLayer<RayTracerLayer>()->m_environmentProperties;
	environment.m_environmentalLightingType = EnvironmentalLightingType::SingleLightSource;
	environment.m_sunDirection = glm::quat(glm::radians(glm::vec3(120, 0, 0))) * glm::vec3(0, 0, -1);
	environment.m_lightSize = m_environmentSettings.m_lightSize;
	environment.m_ambientLightIntensity = m_environmentSettings.m_ambientLightIntensity;
	scene->m_environmentSettings.m_ambientLightIntensity = m_environmentSettings.m_envLightIntensity;
	scene->m_environmentSettings.m_backgroundColor = glm::vec3(1, 1, 1);
	scene->m_environmentSettings.m_environmentType = EnvironmentType::Color;
#endif

	m_projections.clear();
	m_views.clear();
	m_names.clear();
	m_cameraModels.clear();
	m_treeModels.clear();

	if (m_environmentSettings.m_enableGround) {
		m_ground = scene->CreateEntity("Ground");
		auto groundMeshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(m_ground).lock();
		groundMeshRenderer->m_material = ProjectManager::CreateTemporaryAsset<Material>();
		groundMeshRenderer->m_mesh = DefaultResources::Primitives::Quad;
		GlobalTransform groundGT;
		groundGT.SetScale({ 1000, 1, 1000 });
		scene->SetDataComponent<GlobalTransform>(m_ground, groundGT);

	}

	if (m_enableMultipleTrees)
	{
		for (const auto& position : m_treePositions)
		{
			Transform t;
			t.SetPosition(position);
			pipeline.m_transforms.emplace_back(t);
		}
	}

}

void TreeDataCapturePipeline::OnEnd(AutoTreeGenerationPipeline& pipeline) {
	auto scene = pipeline.GetScene();
	if ((m_exportOptions.m_exportDepth || m_exportOptions.m_exportImage || m_exportOptions.m_exportBranchCapture) &&
		m_exportOptions.m_exportMatrices)
		ExportMatrices(m_currentExportFolder /
			"matrices.yml");
	if (m_environmentSettings.m_enableGround) scene->DeleteEntity(m_ground);

	if (false) {
		auto treeIOFolder = m_currentExportFolder / "TreeIO";
		std::string data = "";
		for (const auto& pair : m_treeIOPairs) {
			data += pair.first;
			data += ",";
			data += pair.second;
			data += "\n";
		}
		std::ofstream ofs;
		auto path = treeIOFolder /
			("name_pairings.txt");
		ofs.open(path.string().c_str(),
			std::ofstream::out | std::ofstream::trunc);
		ofs.write(data.c_str(), data.length());
		ofs.flush();
		ofs.close();
		m_treeIOPairs.clear();
	}
}

void TreeDataCapturePipeline::DisableAllExport() {
	m_exportOptions.m_exportTreeIOTrees = false;
	m_exportOptions.m_exportOBJ = false;
	m_exportOptions.m_exportCSV = false;
	m_exportOptions.m_exportEnvironmentalGrid = false;
	m_exportOptions.m_exportWallPrefab = false;
	m_exportOptions.m_exportGraph = false;
	m_exportOptions.m_exportImage = false;
	m_exportOptions.m_exportDepth = false;
	m_exportOptions.m_exportMatrices = false;
	m_exportOptions.m_exportBranchCapture = false;
	m_exportOptions.m_exportLString = false;
}

GlobalTransform
TreeDataCapturePipeline::CameraCaptureSettings::GetTransform(bool isCamera, const Bound& bound, float turnAngle,
	float pitchAngle) {
	GlobalTransform cameraGlobalTransform;
	float distance = m_distance;
	glm::vec3 focusPoint = m_focusPoint;
	if (m_autoAdjustFocusPoint) {
		focusPoint = (bound.m_min + bound.m_max) / 2.0f;
		float halfAngle = (m_fov - 35.0f) / 2.0f;
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
	auto height = distance * glm::sin(glm::radians((float)pitchAngle));
	auto groundDistance =
		distance * glm::cos(glm::radians((float)pitchAngle));
	glm::vec3 cameraPosition =
		glm::vec3(glm::sin(glm::radians((float)turnAngle)) * groundDistance,
			height,
			glm::cos(glm::radians((float)turnAngle)) * groundDistance);


	cameraGlobalTransform.SetPosition(cameraPosition + focusPoint);
	cameraGlobalTransform.SetRotation(glm::quatLookAt(glm::normalize(-cameraPosition), glm::vec3(0, 1, 0)));
	return cameraGlobalTransform;
}

void TreeDataCapturePipeline::CameraCaptureSettings::Serialize(const std::string& name, YAML::Emitter& out) {
	out << YAML::Key << name << YAML::Value << YAML::BeginMap;
	out << YAML::Key << "m_autoAdjustFocusPoint" << YAML::Value
		<< m_autoAdjustFocusPoint;
	out << YAML::Key << "m_focusPoint" << YAML::Value << m_focusPoint;
	out << YAML::Key << "m_pitchAngleStart" << YAML::Value << m_pitchAngleStart;
	out << YAML::Key << "m_pitchAngleStep" << YAML::Value << m_pitchAngleStep;
	out << YAML::Key << "m_pitchAngleEnd" << YAML::Value << m_pitchAngleEnd;
	out << YAML::Key << "m_turnAngleStart" << YAML::Value << m_turnAngleStart;
	out << YAML::Key << "m_turnAngleStep" << YAML::Value << m_turnAngleStep;
	out << YAML::Key << "m_turnAngleEnd" << YAML::Value << m_turnAngleEnd;
	out << YAML::Key << "m_distance" << YAML::Value << m_distance;
	out << YAML::Key << "m_fov" << YAML::Value << m_fov;
	out << YAML::Key << "m_resolution" << YAML::Value << m_resolution;
	out << YAML::Key << "m_useClearColor" << YAML::Value << m_useClearColor;
	out << YAML::Key << "m_backgroundColor" << YAML::Value << m_backgroundColor;
	out << YAML::Key << "m_cameraDepthMax" << YAML::Value << m_cameraDepthMax;
	out << YAML::EndMap;
}

void TreeDataCapturePipeline::CameraCaptureSettings::Deserialize(const std::string& name, const YAML::Node& in) {
	if (in[name]) {
		auto& cd = in[name];
		if (cd["m_autoAdjustFocusPoint"]) m_autoAdjustFocusPoint = cd["m_autoAdjustFocusPoint"].as<bool>();
		if (cd["m_focusPoint"]) m_focusPoint = cd["m_focusPoint"].as<glm::vec3>();
		if (cd["m_pitchAngleStart"]) m_pitchAngleStart = cd["m_pitchAngleStart"].as<int>();
		if (cd["m_pitchAngleStep"]) m_pitchAngleStep = cd["m_pitchAngleStep"].as<int>();
		if (cd["m_pitchAngleEnd"]) m_pitchAngleEnd = cd["m_pitchAngleEnd"].as<int>();
		if (cd["m_turnAngleStart"]) m_turnAngleStart = cd["m_turnAngleStart"].as<int>();
		if (cd["m_turnAngleStep"]) m_turnAngleStep = cd["m_turnAngleStep"].as<int>();
		if (cd["m_turnAngleEnd"]) m_turnAngleEnd = cd["m_turnAngleEnd"].as<int>();
		if (cd["m_distance"]) m_distance = cd["m_distance"].as<float>();
		if (cd["m_fov"]) m_fov = cd["m_fov"].as<float>();
		if (cd["m_resolution"]) m_resolution = cd["m_resolution"].as<glm::ivec2>();
		if (cd["m_useClearColor"]) m_useClearColor = cd["m_useClearColor"].as<bool>();
		if (cd["m_backgroundColor"]) m_backgroundColor = cd["m_backgroundColor"].as<glm::vec3>();
		if (cd["m_cameraDepthMax"]) m_cameraDepthMax = cd["m_cameraDepthMax"].as<float>();
	}
}

void TreeDataCapturePipeline::CameraCaptureSettings::OnInspect() {
	ImGui::Checkbox("Auto adjust focus point", &m_autoAdjustFocusPoint);
	if (!m_autoAdjustFocusPoint) {
		ImGui::Text("Position:");
		ImGui::DragFloat3("Focus point", &m_focusPoint.x, 0.1f);
		ImGui::DragFloat("Distance to focus point", &m_distance, 0.1);
	}
	ImGui::Separator();
	ImGui::Text("Rotation:");
	ImGui::DragInt3("Pitch Angle Start/Step/End", &m_pitchAngleStart, 1);
	ImGui::DragInt3("Turn Angle Start/Step/End", &m_turnAngleStart, 1);
	ImGui::Separator();
	ImGui::Text("Camera Settings:");
	ImGui::DragFloat("FOV", &m_fov);
	ImGui::DragInt2("Resolution", &m_resolution.x);
	ImGui::DragFloat("Max Depth", &m_cameraDepthMax);
	ImGui::Checkbox("Use clear color", &m_useClearColor);
	if (m_useClearColor) ImGui::ColorEdit3("Clear Color", &m_backgroundColor.x);

#ifdef RAYTRACERFACILITY
	ImGui::Separator();
	ImGui::Text("Ray tracer Settings");
	ImGui::DragInt("Bounce", &m_rayProperties.m_bounces);
	ImGui::DragInt("Sample", &m_rayProperties.m_samples);
#endif
}

void TreeDataCapturePipeline::CollectAssetRef(std::vector<AssetRef>& list) {
	list.push_back(m_appearanceSettings.m_branchTexture);
	list.push_back(m_appearanceSettings.m_foliagePhyllotaxis);
	list.push_back(m_obstacleGrid);
}

void TreeDataCapturePipeline::Serialize(YAML::Emitter& out) {
	m_obstacleGrid.Save("m_obstacleGrid", out);
	m_cameraSettings.Serialize("m_cameraSettings", out);
	m_pointCloudSettings.Serialize("m_pointCloudSettings", out);
	out << YAML::Key << "m_pointCloudPointSettings.m_pointType" << YAML::Value << m_pointCloudPointSettings.m_pointType;
	out << YAML::Key << "m_pointCloudPointSettings.m_variance" << YAML::Value << m_pointCloudPointSettings.m_variance;
	out << YAML::Key << "m_pointCloudPointSettings.m_ballRandRadius" << YAML::Value
		<< m_pointCloudPointSettings.m_ballRandRadius;
	out << YAML::Key << "m_pointCloudPointSettings.m_junctionIndex" << YAML::Value << m_pointCloudPointSettings.m_junctionIndex;
	out << YAML::Key << "m_pointCloudPointSettings.m_internodeIndex" << YAML::Value << m_pointCloudPointSettings.m_internodeIndex;
	out << YAML::Key << "m_currentExportFolder" << YAML::Value << m_currentExportFolder.string();
	m_meshGeneratorSettings.Save("m_meshGeneratorSettings", out);
	out << YAML::Key << "m_defaultBehaviourType" << YAML::Value << (unsigned)m_defaultBehaviourType;

	out << YAML::Key << "m_obstacleSettings.m_enableRandomObstacle" << YAML::Value
		<< m_obstacleSettings.m_enableRandomObstacle;
	out << YAML::Key << "m_obstacleSettings.m_renderObstacle" << YAML::Value << m_obstacleSettings.m_renderObstacle;
	out << YAML::Key << "m_obstacleSettings.m_lShapedWall" << YAML::Value << m_obstacleSettings.m_lShapedWall;
	out << YAML::Key << "m_obstacleSettings.m_obstacleDistanceRange" << YAML::Value
		<< m_obstacleSettings.m_obstacleDistanceRange;
	out << YAML::Key << "m_obstacleSettings.m_wallSize" << YAML::Value << m_obstacleSettings.m_wallSize;
	out << YAML::Key << "m_obstacleSettings.m_randomRotation" << YAML::Value << m_obstacleSettings.m_randomRotation;

	m_appearanceSettings.m_branchTexture.Save("m_appearanceSettings.m_branchTexture", out);
	m_appearanceSettings.m_foliagePhyllotaxis.Save("m_appearanceSettings.m_foliagePhyllotaxis", out);
	out << YAML::Key << "m_appearanceSettings.m_applyPhyllotaxis" << YAML::Value
		<< m_appearanceSettings.m_applyPhyllotaxis;
	out << YAML::Key << "m_appearanceSettings.m_branchWidth" << YAML::Value << m_appearanceSettings.m_branchWidth;
	out << YAML::Key << "m_appearanceSettings.m_nodeSize" << YAML::Value << m_appearanceSettings.m_nodeSize;

	out << YAML::Key << "m_environmentSettings.m_enableGround" << YAML::Value << m_environmentSettings.m_enableGround;
	out << YAML::Key << "m_environmentSettings.m_lightSize" << YAML::Value << m_environmentSettings.m_lightSize;
	out << YAML::Key << "m_environmentSettings.m_ambientLightIntensity" << YAML::Value
		<< m_environmentSettings.m_ambientLightIntensity;
	out << YAML::Key << "m_environmentSettings.m_envLightIntensity" << YAML::Value
		<< m_environmentSettings.m_envLightIntensity;

	out << YAML::Key << "m_exportOptions.m_exportEnvironmentalGrid" << YAML::Value
		<< m_exportOptions.m_exportEnvironmentalGrid;
	out << YAML::Key << "m_exportOptions.m_exportWallPrefab" << YAML::Value << m_exportOptions.m_exportWallPrefab;
	out << YAML::Key << "m_exportOptions.m_exportTreeIOTrees" << YAML::Value << m_exportOptions.m_exportTreeIOTrees;
	out << YAML::Key << "m_exportOptions.m_exportOBJ" << YAML::Value << m_exportOptions.m_exportOBJ;
	out << YAML::Key << "m_exportOptions.m_exportCSV" << YAML::Value << m_exportOptions.m_exportCSV;
	out << YAML::Key << "m_exportOptions.m_exportGraph" << YAML::Value << m_exportOptions.m_exportGraph;
	out << YAML::Key << "m_exportOptions.m_exportMask" << YAML::Value << m_exportOptions.m_exportMask;
	out << YAML::Key << "m_exportOptions.m_exportRBV" << YAML::Value << m_exportOptions.m_exportRBV;
	out << YAML::Key << "m_exportOptions.m_exportImage" << YAML::Value << m_exportOptions.m_exportImage;
	out << YAML::Key << "m_exportOptions.m_exportDepth" << YAML::Value << m_exportOptions.m_exportDepth;
	out << YAML::Key << "m_exportOptions.m_exportMatrices" << YAML::Value << m_exportOptions.m_exportMatrices;
	out << YAML::Key << "m_exportOptions.m_exportBranchCapture" << YAML::Value << m_exportOptions.m_exportBranchCapture;
	out << YAML::Key << "m_exportOptions.m_exportLString" << YAML::Value << m_exportOptions.m_exportLString;
	out << YAML::Key << "m_exportOptions.m_exportPointCloud" << YAML::Value << m_exportOptions.m_exportPointCloud;

	out << YAML::Key << "m_enableMultipleTrees" << YAML::Value << m_enableMultipleTrees;
	if (!m_treePositions.empty())
	{
		out << YAML::Key << "m_treePositions" << YAML::Value
			<< YAML::Binary((const unsigned char*)m_treePositions.data(), m_treePositions.size() * sizeof(glm::vec3));
	}
}

void TreeDataCapturePipeline::Deserialize(const YAML::Node& in) {
	m_obstacleGrid.Load("m_obstacleGrid", in);
	m_meshGeneratorSettings.Load("m_meshGeneratorSettings", in);
	m_cameraSettings.Deserialize("m_cameraSettings", in);
	m_pointCloudSettings.Deserialize("m_pointCloudSettings", in);
	if (in["m_pointCloudPointSettings.m_pointType"]) m_pointCloudPointSettings.m_pointType = in["m_pointCloudPointSettings.m_pointType"].as<bool>();
	if (in["m_pointCloudPointSettings.m_variance"]) m_pointCloudPointSettings.m_variance = in["m_pointCloudPointSettings.m_variance"].as<float>();
	if (in["m_pointCloudPointSettings.m_ballRandRadius"]) m_pointCloudPointSettings.m_ballRandRadius = in["m_pointCloudPointSettings.m_ballRandRadius"].as<float>();
	if (in["m_pointCloudPointSettings.m_junctionIndex"]) m_pointCloudPointSettings.m_junctionIndex = in["m_pointCloudPointSettings.m_junctionIndex"].as<bool>();
	if (in["m_pointCloudPointSettings.m_internodeIndex"]) m_pointCloudPointSettings.m_internodeIndex = in["m_pointCloudPointSettings.m_internodeIndex"].as<bool>();

	if (in["m_obstacleSettings.m_randomRotation"]) m_obstacleSettings.m_randomRotation = in["m_obstacleSettings.m_randomRotation"].as<bool>();
	if (in["m_obstacleSettings.m_lShapedWall"]) m_obstacleSettings.m_lShapedWall = in["m_obstacleSettings.m_lShapedWall"].as<bool>();
	if (in["m_obstacleSettings.m_enableRandomObstacle"]) m_obstacleSettings.m_enableRandomObstacle = in["m_obstacleSettings.m_enableRandomObstacle"].as<bool>();
	if (in["m_obstacleSettings.m_renderObstacle"]) m_obstacleSettings.m_renderObstacle = in["m_obstacleSettings.m_renderObstacle"].as<bool>();
	if (in["m_obstacleSettings.m_obstacleDistanceRange"]) m_obstacleSettings.m_obstacleDistanceRange = in["m_obstacleSettings.m_obstacleDistanceRange"].as<glm::vec2>();
	if (in["m_obstacleSettings.m_wallSize"]) m_obstacleSettings.m_wallSize = in["m_obstacleSettings.m_wallSize"].as<glm::vec3>();

	if (in["m_defaultBehaviourType"]) m_defaultBehaviourType = (BehaviourType)in["m_defaultBehaviourType"].as<unsigned>();
	if (in["m_currentExportFolder"]) m_currentExportFolder = in["m_currentExportFolder"].as<std::string>();

	m_appearanceSettings.m_branchTexture.Load("m_appearanceSettings.m_branchTexture", in);
	m_appearanceSettings.m_foliagePhyllotaxis.Load("m_appearanceSettings.m_foliagePhyllotaxis", in);
	if (in["m_appearanceSettings.m_applyPhyllotaxis"]) m_appearanceSettings.m_applyPhyllotaxis = in["m_appearanceSettings.m_applyPhyllotaxis"].as<bool>();
	if (in["m_appearanceSettings.m_branchWidth"]) m_appearanceSettings.m_branchWidth = in["m_appearanceSettings.m_branchWidth"].as<float>();
	if (in["m_appearanceSettings.m_nodeSize"]) m_appearanceSettings.m_nodeSize = in["m_appearanceSettings.m_nodeSize"].as<float>();

	if (in["m_environmentSettings.m_lightSize"]) m_environmentSettings.m_lightSize = in["m_environmentSettings.m_lightSize"].as<float>();
	if (in["m_environmentSettings.m_ambientLightIntensity"]) m_environmentSettings.m_ambientLightIntensity = in["m_environmentSettings.m_ambientLightIntensity"].as<float>();
	if (in["m_environmentSettings.m_envLightIntensity"]) m_environmentSettings.m_envLightIntensity = in["m_environmentSettings.m_envLightIntensity"].as<float>();
	if (in["m_environmentSettings.m_enableGround"]) m_environmentSettings.m_enableGround = in["m_environmentSettings.m_enableGround"].as<bool>();

	if (in["m_exportOptions.m_exportTreeIOTrees"]) m_exportOptions.m_exportTreeIOTrees = in["m_exportOptions.m_exportTreeIOTrees"].as<bool>();
	if (in["m_exportOptions.m_exportOBJ"]) m_exportOptions.m_exportOBJ = in["m_exportOptions.m_exportOBJ"].as<bool>();
	if (in["m_exportOptions.m_exportCSV"]) m_exportOptions.m_exportCSV = in["m_exportOptions.m_exportCSV"].as<bool>();
	if (in["m_exportOptions.m_exportRBV"]) m_exportOptions.m_exportRBV = in["m_exportOptions.m_exportRBV"].as<bool>();
	if (in["m_exportOptions.m_exportGraph"]) m_exportOptions.m_exportGraph = in["m_exportOptions.m_exportGraph"].as<bool>();
	if (in["m_exportOptions.m_exportMask"]) m_exportOptions.m_exportMask = in["m_exportOptions.m_exportMask"].as<bool>();
	if (in["m_exportOptions.m_exportImage"]) m_exportOptions.m_exportImage = in["m_exportOptions.m_exportImage"].as<bool>();
	if (in["m_exportOptions.m_exportDepth"]) m_exportOptions.m_exportDepth = in["m_exportOptions.m_exportDepth"].as<bool>();
	if (in["m_exportOptions.m_exportMatrices"]) m_exportOptions.m_exportMatrices = in["m_exportOptions.m_exportMatrices"].as<bool>();
	if (in["m_exportOptions.m_exportBranchCapture"]) m_exportOptions.m_exportBranchCapture = in["m_exportOptions.m_exportBranchCapture"].as<bool>();
	if (in["m_exportOptions.m_exportLString"]) m_exportOptions.m_exportLString = in["m_exportOptions.m_exportLString"].as<bool>();
	if (in["m_exportOptions.m_exportPointCloud"]) m_exportOptions.m_exportPointCloud = in["m_exportOptions.m_exportPointCloud"].as<bool>();
	if (in["m_exportOptions.m_exportWallPrefab"]) m_exportOptions.m_exportWallPrefab = in["m_exportOptions.m_exportWallPrefab"].as<bool>();
	if (in["m_exportOptions.m_exportEnvironmentalGrid"]) m_exportOptions.m_exportEnvironmentalGrid = in["m_exportOptions.m_exportEnvironmentalGrid"].as<bool>();

	if (in["m_enableMultipleTrees"]) m_enableMultipleTrees = in["m_enableMultipleTrees"].as<bool>();

	if (in["m_treePositions"])
	{
		const auto& ds = in["m_treePositions"].as<YAML::Binary>();
		m_treePositions.resize(ds.size() / sizeof(glm::vec3));
		std::memcpy(m_treePositions.data(), ds.data(), ds.size());
	}
}

void TreeDataCapturePipeline::ExportEnvironmentalGrid(AutoTreeGenerationPipeline& pipeline,
	const std::filesystem::path& path) {
	auto grid = m_obstacleGrid.Get<VoxelGrid>();
	if (grid) {
		grid->FillObstacle(pipeline.GetScene());
	}
	grid->Export(path);
}

#ifdef RAYTRACERFACILITY

void TreeDataCapturePipeline::ScanPointCloudLabeled(const Bound& plantBound, AutoTreeGenerationPipeline& pipeline,
	const std::filesystem::path& savePath) {
	std::vector<PointCloudSample> pcSamples;
	int counter = 0;
	for (int turnAngle = m_pointCloudSettings.m_turnAngleStart;
		turnAngle < m_pointCloudSettings.m_turnAngleEnd; turnAngle += m_pointCloudSettings.m_turnAngleStep) {
		for (int pitchAngle = m_pointCloudSettings.m_pitchAngleStart;
			pitchAngle < m_pointCloudSettings.m_pitchAngleEnd; pitchAngle += m_pointCloudSettings.m_pitchAngleStep) {
			pcSamples.resize((counter + 1) * m_pointCloudSettings.m_resolution.x * m_pointCloudSettings.m_resolution.y);
			auto scannerGlobalTransform = m_pointCloudSettings.GetTransform(false, plantBound, turnAngle, pitchAngle);
			auto front = scannerGlobalTransform.GetRotation() * glm::vec3(0, 0, -1);
			auto up = scannerGlobalTransform.GetRotation() * glm::vec3(0, 1, 0);
			auto left = scannerGlobalTransform.GetRotation() * glm::vec3(1, 0, 0);
			auto position = scannerGlobalTransform.GetPosition();
			std::vector<std::shared_future<void>> results;
			Jobs::ParallelFor(
				m_pointCloudSettings.m_resolution.x * m_pointCloudSettings.m_resolution.y,
				[&](unsigned i) {
					unsigned x = i % m_pointCloudSettings.m_resolution.x;
			unsigned y = i / m_pointCloudSettings.m_resolution.x;
			const float xAngle = (x - m_pointCloudSettings.m_resolution.x / 2.0f) /
				(float)m_pointCloudSettings.m_resolution.x * m_pointCloudSettings.m_fov /
				2.0f;
			const float yAngle = (y - m_pointCloudSettings.m_resolution.y / 2.0f) /
				(float)m_pointCloudSettings.m_resolution.y * m_pointCloudSettings.m_fov /
				2.0f;
			auto& sample = pcSamples[
				counter * m_pointCloudSettings.m_resolution.x * m_pointCloudSettings.m_resolution.y +
					i];
			sample.m_direction = glm::normalize(glm::rotate(glm::rotate(front, glm::radians(xAngle), left),
				glm::radians(yAngle), up));
			sample.m_start = position;
				},
				results);
			for (const auto& i : results)
				i.wait();

			counter++;
		}
	}
	CudaModule::SamplePointCloud(
		Application::GetLayer<RayTracerLayer>()->m_environmentProperties,
		pcSamples);
	auto scene = pipeline.GetScene();
	std::vector<glm::vec3> points;
	std::vector<glm::vec3> growthDirection;
	std::vector<int> junctionIndex;
	std::vector<int> internodeIndex;
	std::vector<int> branchIndex;
	std::vector<glm::vec3> ishape;
	std::vector<int> ishapeIndex;
	std::vector<int> pointTypes;
	std::set<Handle> branchMeshRendererHandles, foliageMeshRendererHandles;
	Handle groundMeshRendererHandle;
	if (scene->IsEntityValid(m_ground) && scene->HasPrivateComponent<MeshRenderer>(m_ground)) {
		groundMeshRendererHandle = scene->GetOrSetPrivateComponent<MeshRenderer>(m_ground).lock()->GetHandle();
	}
	for (const auto& tree : pipeline.m_currentGrowingTrees) {
		if (scene->IsEntityValid(tree)) {
			scene->ForEachChild(tree, [&](Entity child) {
				if (scene->GetEntityName(child) == "BranchMesh" && scene->HasPrivateComponent<SkinnedMeshRenderer>(child)) {
					branchMeshRendererHandles.insert(scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(
						child).lock()->GetHandle());
				}
				else if (scene->GetEntityName(child) == "FoliageMesh" &&
					scene->HasPrivateComponent<SkinnedMeshRenderer>(child)) {
					foliageMeshRendererHandles.insert(scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(
						child).lock()->GetHandle());
				}
				});
		}
	}
	for (const auto& sample : pcSamples) {
		if (!sample.m_hit) continue;
		auto& position = sample.m_hitInfo.m_position;
		if (position.x<(plantBound.m_min.x - m_pointCloudPointSettings.m_boundingBoxOffset) ||
			position.z<(plantBound.m_min.z - m_pointCloudPointSettings.m_boundingBoxOffset) || position.x>(plantBound.m_max.x + m_pointCloudPointSettings.m_boundingBoxOffset) ||
			position.z>(plantBound.m_max.z + m_pointCloudPointSettings.m_boundingBoxOffset) || position.y < -0.01f)
			continue;
		auto ballRand = glm::vec3(0.0f);
		if (m_pointCloudPointSettings.m_ballRandRadius > 0.0f) {
			ballRand = glm::ballRand(m_pointCloudPointSettings.m_ballRandRadius);
		}
		points.push_back(
			sample.m_hitInfo.m_position +
			glm::vec3(glm::gaussRand(0.0f, m_pointCloudPointSettings.m_variance),
				glm::gaussRand(0.0f, m_pointCloudPointSettings.m_variance),
				glm::gaussRand(0.0f, m_pointCloudPointSettings.m_variance))
			+ ballRand);
		if(branchMeshRendererHandles.find(sample.m_handle) != branchMeshRendererHandles.end())
		{
			if (m_pointCloudPointSettings.m_pointType) pointTypes.push_back(0);
		}
		else if (foliageMeshRendererHandles.find(sample.m_handle) != foliageMeshRendererHandles.end())
		{
			if (m_pointCloudPointSettings.m_pointType) pointTypes.push_back(1);
		}
		else if(sample.m_handle == groundMeshRendererHandle) {
			if (m_pointCloudPointSettings.m_pointType) pointTypes.push_back(2);
		}else{
			if (m_pointCloudPointSettings.m_pointType) pointTypes.push_back(3);
		}
		if(m_pointCloudPointSettings.m_growthDirection)
		{
			growthDirection.emplace_back(glm::normalize(sample.m_hitInfo.m_color));
		}
		if (m_pointCloudPointSettings.m_junctionIndex) {
			
			junctionIndex.emplace_back((int)(sample.m_hitInfo.m_data.x + 0.1f));
		}
		if(m_pointCloudPointSettings.m_internodeIndex)
		{
			internodeIndex.emplace_back((int)(sample.m_hitInfo.m_data.y + 0.1f));
		}
		if (m_pointCloudPointSettings.m_branchIndex)
		{
			branchIndex.emplace_back((int)(sample.m_hitInfo.m_data.z + 0.1f));
		}
	}
	std::filebuf fb_binary;
	fb_binary.open(savePath.string(), std::ios::out | std::ios::binary);
	std::ostream outstream_binary(&fb_binary);
	if (outstream_binary.fail())
		throw std::runtime_error("failed to open " + savePath.string());
	/*
	std::filebuf fb_ascii;
	fb_ascii.open(filename + "-ascii.ply", std::ios::out);
	std::ostream outstream_ascii(&fb_ascii);
	if (outstream_ascii.fail()) throw std::runtime_error("failed to open " +
	filename);
	*/
	PlyFile cube_file;
	cube_file.add_properties_to_element(
		"vertex", { "x", "y", "z" }, Type::FLOAT32, points.size(),
		reinterpret_cast<uint8_t*>(points.data()), Type::INVALID, 0);

	if (m_pointCloudPointSettings.m_pointType)
		cube_file.add_properties_to_element(
			"pointType", { "value" }, Type::INT32, pointTypes.size(),
			reinterpret_cast<uint8_t*>(pointTypes.data()), Type::INVALID, 0);
	if(m_pointCloudPointSettings.m_growthDirection)
	{
		cube_file.add_properties_to_element(
			"growthDirection", { "gx", "gy", "gz" }, Type::FLOAT32, growthDirection.size(),
			reinterpret_cast<uint8_t*>(growthDirection.data()), Type::INVALID, 0);
	}
	if (m_pointCloudPointSettings.m_junctionIndex) {
		cube_file.add_properties_to_element(
			"junctionIndex", { "ji" }, Type::INT32, junctionIndex.size(),
			reinterpret_cast<uint8_t*>(junctionIndex.data()), Type::INVALID, 0);
	}
	if (m_pointCloudPointSettings.m_internodeIndex)
	{
		cube_file.add_properties_to_element(
			"internodeIndex", { "ii" }, Type::INT32, internodeIndex.size(),
			reinterpret_cast<uint8_t*>(internodeIndex.data()), Type::INVALID, 0);
	}
	if (m_pointCloudPointSettings.m_branchIndex)
	{
		cube_file.add_properties_to_element(
			"branchIndex", { "bi" }, Type::INT32, branchIndex.size(),
			reinterpret_cast<uint8_t*>(branchIndex.data()), Type::INVALID, 0);
	}
	// Write a binary file
	cube_file.write(outstream_binary, true);
}

struct IShapeUnitInfo
{

};

struct JunctionUnitInfo {
	glm::vec3 m_position;
	glm::vec3 m_direction;
	float m_radius;
};
struct Junction {
	int m_junctionIndex;
	int m_twigSize = 0;
	JunctionUnitInfo m_root;
	glm::vec3 m_startPos;
	std::vector<JunctionUnitInfo> m_children;
};
struct IShape {
	int m_iShapeIndex;
	int m_twigSize = 0;
	std::vector<float> m_radius;
	std::vector<glm::vec3> m_positions;
	std::vector<glm::vec3> m_directions;
};
void TreeDataCapturePipeline::ExportJunction(AutoTreeGenerationPipeline& pipeline,
	const std::shared_ptr<IPlantBehaviour>& behaviour,
	const std::filesystem::path& path) {
	auto scene = pipeline.GetScene();
	auto internodeLayer = Application::GetLayer<InternodeLayer>();
	try {
		auto directory = path;
		directory.remove_filename();
		std::filesystem::create_directories(directory);
		YAML::Emitter out;
		out << YAML::BeginMap;

		std::unordered_set<int> junctionRootIndices;
		std::unordered_set<int> junctionChildrenIndices;
		std::unordered_set<int> iShapeIndices;
		std::vector<Junction> junctions;
		std::vector<IShape> ishapes;
		std::vector<Entity> internodeEntities;
		std::vector<Entity> branches;
		std::vector<int> rootIndices;
		scene->GetEntityArray(internodeLayer->m_internodesQuery, internodeEntities);
		for (const auto& internodeEntity : internodeEntities) {
			auto childrenSize = scene->GetChildrenAmount(internodeEntity);
			auto rootInfo = scene->GetDataComponent<InternodeInfo>(internodeEntity);
			if (!scene->HasPrivateComponent<Internode>(scene->GetParent(internodeEntity)))
			{
				rootIndices.emplace_back(internodeEntity.GetIndex());
			}
			if (childrenSize > 1) {
				bool add = true;
				Junction junction;
				junction.m_junctionIndex = internodeEntity.GetIndex();
				auto treeTransform = scene->GetDataComponent<GlobalTransform>(scene->GetRoot(internodeEntity));
				auto rootTransform = scene->GetDataComponent<GlobalTransform>(internodeEntity);
				junction.m_root.m_direction = glm::normalize(rootTransform.GetRotation() * glm::vec3(0, 0, -1));
				junction.m_root.m_position = rootTransform.GetPosition() + junction.m_root.m_direction * rootInfo.m_length;
				junction.m_root.m_radius = rootInfo.m_thickness;
				//junction.m_startPos = rootTransform.GetPosition();
				auto rootInternode = scene->GetOrSetPrivateComponent<Internode>(internodeEntity).lock();
				int rootRingIndex = rootInternode->m_rings.size() * (1.0f - m_meshGeneratorSettings.m_junctionUpperRatio);
				if (rootRingIndex < 0) rootRingIndex = 0;
				if (rootRingIndex >= rootInternode->m_rings.size()) rootRingIndex = rootInternode->m_rings.size() - 1;
				junction.m_startPos = rootInternode->m_rings.at(rootRingIndex).m_endPosition;
				int validChildCount = 0;
				junction.m_twigSize = 0;
				for (const auto& child : scene->GetChildren(internodeEntity)) {
					if (!scene->HasDataComponent<InternodeInfo>(child)) continue;
					auto internode = scene->GetOrSetPrivateComponent<Internode>(child).lock();
					
					auto childTransform = scene->GetDataComponent<GlobalTransform>(child);
					junction.m_children.emplace_back();
					auto& childInfo = junction.m_children.back();
					auto childInternodeInfo = scene->GetDataComponent<InternodeInfo>(child);
					if (childInternodeInfo.m_length > 0.3f) {
						validChildCount++;
					}
					junction.m_twigSize += internode->m_twigs.size();
					childInfo.m_direction = glm::normalize(childTransform.GetRotation() * glm::vec3(0, 0, -1));
					//childInfo.m_position = childTransform.GetPosition() + childInfo.m_direction * childInternodeInfo.m_length;
					auto childInternode = scene->GetOrSetPrivateComponent<Internode>(child).lock();
					int childRingIndex = childInternode->m_rings.size() * m_meshGeneratorSettings.m_junctionLowerRatio / childInternodeInfo.m_length;
					if (childRingIndex < 0) childRingIndex = 0;
					if (childRingIndex >= childInternode->m_rings.size()) childRingIndex = childInternode->m_rings.size() - 1;
					childInfo.m_position = treeTransform.GetPosition() + childInternode->m_rings.at(childRingIndex).m_startPosition;
					childInfo.m_radius = childInternode->m_rings.at(childRingIndex).m_startRadius;

				}
				if(validChildCount != 0) junction.m_twigSize /= validChildCount;
				if (validChildCount > 1) {
					junctions.push_back(junction);
					junctionRootIndices.emplace(junction.m_junctionIndex);
					for (const auto& child : scene->GetChildren(internodeEntity)) {
						if (!scene->HasDataComponent<InternodeInfo>(child)) continue;
						junctionChildrenIndices.emplace(child.GetIndex());
					}
				}
			}

		}
		scene->GetEntityArray(internodeLayer->m_branchesQuery, branches);
		for (const auto& branchEntity : branches)
		{
			auto& iShape = ishapes.emplace_back();
			iShape.m_iShapeIndex = branchEntity.GetIndex();
			iShapeIndices.emplace(iShape.m_iShapeIndex);
			if (!scene->HasPrivateComponent<Branch>(scene->GetParent(branchEntity)))
			{
				rootIndices.emplace_back(branchEntity.GetIndex());
			}

			auto branch = scene->GetOrSetPrivateComponent<Branch>(branchEntity).lock();
			iShape.m_twigSize = 0;
			int validInternodeSize = 0;
			for (const auto& internodeEntity : branch->m_internodeChain)
			{
				auto internode = scene->GetOrSetPrivateComponent<Internode>(internodeEntity).lock();
				
				auto internodeInfo = scene->GetDataComponent<InternodeInfo>(internodeEntity);
				auto treeTransform = scene->GetDataComponent<GlobalTransform>(scene->GetRoot(internodeEntity));
				bool isRoot = false;
				bool isChild = false;
				if (junctionRootIndices.find(internodeEntity.GetIndex()) != junctionRootIndices.end())
				{
					//Root
					//Only output previous rings to the start position to last valid ring.
					isRoot = true;
				}
				else if (junctionChildrenIndices.find(internodeEntity.GetIndex()) != junctionChildrenIndices.end())
				{
					//Children
					isChild = true;
				}
				validInternodeSize++;
				iShape.m_twigSize += internode->m_twigs.size();
				int rootRingIndex = internode->m_rings.size() * (1.0f - m_meshGeneratorSettings.m_junctionUpperRatio);
				if (rootRingIndex < 0) rootRingIndex = 0;
				if (rootRingIndex >= internode->m_rings.size()) rootRingIndex = internode->m_rings.size() - 1;
				int childRingIndex = internode->m_rings.size() * m_meshGeneratorSettings.m_junctionLowerRatio / internodeInfo.m_length;
				if (childRingIndex < 0) childRingIndex = 0;
				if (childRingIndex >= internode->m_rings.size()) childRingIndex = internode->m_rings.size() - 1;
				int startIndex = 0;
				int endIndex = internode->m_rings.size() - 1;
				if(isChild)
				{
					startIndex = childRingIndex;
					iShape.m_radius.emplace_back(internode->m_rings.at(startIndex).m_startRadius);
					iShape.m_positions.emplace_back(treeTransform.GetPosition() + internode->m_rings.at(startIndex).m_startPosition);
					iShape.m_directions.emplace_back(glm::normalize(internode->m_rings.at(startIndex).m_startAxis));
				}
				
				if(isRoot)
				{
					endIndex = rootRingIndex;
					iShape.m_radius.emplace_back(internode->m_rings.at(endIndex).m_endRadius);
					iShape.m_positions.emplace_back(treeTransform.GetPosition() + internode->m_rings.at(endIndex).m_endPosition);
					iShape.m_directions.emplace_back(glm::normalize(internode->m_rings.at(endIndex).m_endAxis));
				}else
				{
					iShape.m_radius.emplace_back(internode->m_rings.back().m_endRadius);
					iShape.m_positions.emplace_back(treeTransform.GetPosition() + internode->m_rings.back().m_endPosition);
					iShape.m_directions.emplace_back(glm::normalize(internode->m_rings.back().m_endAxis));
				}
			}
			if (validInternodeSize != 0) iShape.m_twigSize /= validInternodeSize;
		}
		out << YAML::Key << "Possible Root" << YAML::Value << YAML::BeginSeq;
		for (const auto& i : rootIndices)
		{
			if (junctionRootIndices.find(i) != junctionRootIndices.end() || iShapeIndices.find(i) != iShapeIndices.end()) {
				out << YAML::BeginMap;
				out << YAML::Key << "I" << YAML::Value << i;
				out << YAML::EndMap;
			}
		}
		out << YAML::EndSeq;

		out << YAML::Key << "Junctions" << YAML::Value << YAML::BeginSeq;
		for (const auto& junction : junctions) {
			out << YAML::BeginMap;
			out << YAML::Key << "I" << YAML::Value << junction.m_junctionIndex;
			out << YAML::Key << "TS" << YAML::Value << junction.m_twigSize;
			out << YAML::Key << "RD" << YAML::Value << junction.m_root.m_direction;
			out << YAML::Key << "RP" << YAML::Value << junction.m_root.m_position;
			out << YAML::Key << "RR" << YAML::Value << junction.m_root.m_radius;
			out << YAML::Key << "SP" << YAML::Value << junction.m_startPos;
			out << YAML::Key << "C" << YAML::Value << YAML::BeginSeq;
			for (const auto& child : junction.m_children) {
				out << YAML::BeginMap;
				out << YAML::Key << "D" << YAML::Value << child.m_direction;
				out << YAML::Key << "P" << YAML::Value << child.m_position;
				out << YAML::Key << "R" << YAML::Value << child.m_radius;
				out << YAML::EndMap;
			}
			out << YAML::EndSeq;
			out << YAML::EndMap;
		}

		out << YAML::EndSeq;

		out << YAML::Key << "Branches" << YAML::Value << YAML::BeginSeq;
		for (const auto& ishape : ishapes)
		{
			out << YAML::BeginMap;
			out << YAML::Key << "I" << YAML::Value << ishape.m_iShapeIndex;
			out << YAML::Key << "TS" << YAML::Value << ishape.m_twigSize;
			out << YAML::Key << "SN" << YAML::Value << YAML::BeginSeq;
			for (int i = 0; i < ishape.m_radius.size(); i++)
			{
				out << YAML::BeginMap;
				out << YAML::Key << "D" << YAML::Value << ishape.m_directions[i];
				out << YAML::Key << "P" << YAML::Value << ishape.m_positions[i];
				out << YAML::Key << "R" << YAML::Value << ishape.m_radius[i];
				out << YAML::EndMap;
			}
			out << YAML::EndSeq;
			out << YAML::EndMap;
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

#endif