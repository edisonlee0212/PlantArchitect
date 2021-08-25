#include <CubeVolume.hpp>
#include <Curve.hpp>
#include <Joint.hpp>
#include <PhysicsManager.hpp>
#include <ProjectManager.hpp>
#include <RadialBoundingVolume.hpp>
#include <RayTracedRenderer.hpp>
#include <RigidBody.hpp>
#include <SkinnedMeshRenderer.hpp>
#include <TreeLeaves.hpp>
#include <TreeSystem.hpp>
#include <Volume.hpp>

using namespace UniEngine;
using namespace PlantArchitect;

void TreeSystem::ExportChains(int parentOrder, Entity internode,
                              rapidxml::xml_node<> *chains,
                              rapidxml::xml_document<> *doc) {
    auto order = internode.GetDataComponent<InternodeInfo>().m_order;
    if (order != parentOrder) {
        WriteChain(order, internode, chains, doc);
    }
    internode.ForEachChild([order, &chains, doc](Entity child) {
        ExportChains(order, child, chains, doc);
    });
}

void TreeSystem::WriteChain(int order, Entity internode,
                            rapidxml::xml_node<> *chains,
                            rapidxml::xml_document<> *doc) {
    Entity walker = internode;
    rapidxml::xml_node<> *chain =
            doc->allocate_node(rapidxml::node_element, "Chain", "Node");
    chain->append_attribute(doc->allocate_attribute(
            "gravelius", doc->allocate_string(std::to_string(order + 1).c_str())));
    chains->append_node(chain);
    std::vector<rapidxml::xml_node<> *> nodes;
    while (walker.GetChildrenAmount() != 0) {
        auto *node = doc->allocate_node(rapidxml::node_element, "Node");
        node->append_attribute(doc->allocate_attribute(
                "id", doc->allocate_string(std::to_string(walker.GetIndex()).c_str())));
        nodes.push_back(node);
        // chain->append_node(node);
        Entity temp;
        walker.ForEachChild([&temp, order](Entity child) {
            if (child.GetDataComponent<InternodeInfo>().m_order == order) {
                temp = child;
            }
        });
        walker = temp;
    }
    if (nodes.empty())
        return;
    for (int i = nodes.size() - 1; i >= 0; i--) {
        chain->append_node(nodes[i]);
    }
    if (internode.GetParent().HasDataComponent<InternodeInfo>()) {
        auto *node = doc->allocate_node(rapidxml::node_element, "Node");
        node->append_attribute(doc->allocate_attribute(
                "id", doc->allocate_string(
                        std::to_string(internode.GetParent().GetIndex()).c_str())));
        chain->append_node(node);
    }
}

Entity TreeSystem::GetRootInternode(const Entity &tree) {
    auto retVal = Entity();
    if (!tree.HasDataComponent<PlantInfo>() ||
        tree.GetDataComponent<PlantInfo>().m_plantType != PlantType::GeneralTree)
        return retVal;
    tree.ForEachChild([&](Entity child) {
        if (child.HasDataComponent<InternodeInfo>())
            retVal = child;
    });
    return retVal;
}

Entity TreeSystem::GetLeaves(const Entity &tree) {
    auto retVal = Entity();
    if (!tree.HasDataComponent<PlantInfo>() ||
        tree.GetDataComponent<PlantInfo>().m_plantType != PlantType::GeneralTree)
        return retVal;
    tree.ForEachChild([&](Entity child) {
        if (child.HasDataComponent<TreeLeavesTag>())
            retVal = child;
    });
    if (!retVal.IsValid()) {
        const auto leaves =
                EntityManager::CreateEntity(m_leavesArchetype, "Leaves");
        leaves.SetParent(tree);
        leaves.GetOrSetPrivateComponent<TreeLeaves>();
        auto meshRenderer = leaves.GetOrSetPrivateComponent<MeshRenderer>().lock();
        auto skinnedMeshRenderer =
                leaves.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
        auto rayTracerRenderer =
                leaves.GetOrSetPrivateComponent<RayTracerFacility::RayTracedRenderer>()
                        .lock();
        auto leafMat = AssetManager::LoadMaterial(
                DefaultResources::GLPrograms::StandardProgram);
        meshRenderer->m_material = leafMat;
        leafMat->m_name = "Leaves mat";
        leafMat->m_roughness = 1.0f;
        leafMat->m_cullingMode = MaterialCullingMode::Off;
        leafMat->m_metallic = 0.0f;
        leafMat->m_albedoColor = glm::vec3(0.0f, 1.0f, 0.0f);
        meshRenderer->m_mesh = AssetManager::CreateAsset<Mesh>();
        rayTracerRenderer->SyncWithMeshRenderer();
        meshRenderer->SetEnabled(true);

        skinnedMeshRenderer->m_skinnedMesh =
                AssetManager::CreateAsset<SkinnedMesh>();
        auto animation =
                tree.GetOrSetPrivateComponent<Animator>().lock()->GetAnimation();
        skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>()->m_animation =
                animation;
        skinnedMeshRenderer->AttachAnimator(
                tree.GetOrSetPrivateComponent<Animator>().lock());
        auto skinnedMeshMat = AssetManager::LoadMaterial(
                DefaultResources::GLPrograms::StandardSkinnedProgram);
        skinnedMeshRenderer->m_material = skinnedMeshMat;
        skinnedMeshMat->m_name = "Leaves mat";
        skinnedMeshMat->m_roughness = 1.0f;
        skinnedMeshMat->m_cullingMode = MaterialCullingMode::Off;
        skinnedMeshMat->m_metallic = 0.0f;
        skinnedMeshMat->m_albedoColor = glm::vec3(0.0f, 1.0f, 0.0f);

        skinnedMeshRenderer->SetEnabled(true);
    }
    return retVal;
}

Entity TreeSystem::GetRbv(const Entity &tree) {
    auto retVal = Entity();
    if (!tree.HasDataComponent<PlantInfo>() ||
        tree.GetDataComponent<PlantInfo>().m_plantType != PlantType::GeneralTree)
        return retVal;
    tree.ForEachChild([&](Entity child) {
        if (child.HasDataComponent<RbvTag>())
            retVal = child;
    });
    if (!retVal.IsValid()) {
        const auto rbv = EntityManager::CreateEntity(m_rbvArchetype, "RBV");
        rbv.SetParent(tree, false);
        rbv.GetOrSetPrivateComponent<RadialBoundingVolume>();
    }
    return retVal;
}

void TreeSystem::UpdateBranchCylinder(const bool &displayThickness,
                                      const float &width) {
    EntityManager::ForEach<GlobalTransform, BranchCylinder, BranchCylinderWidth,
            InternodeGrowth, InternodeInfo>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
            [displayThickness,
                    width](int i, Entity entity, GlobalTransform &ltw, BranchCylinder &c,
                           BranchCylinderWidth &branchCylinderWidth,
                           InternodeGrowth &internodeGrowth, InternodeInfo &internodeInfo) {
                glm::vec3 scale;
                glm::quat rotation;
                glm::vec3 translation;
                glm::vec3 skew;
                glm::vec4 perspective;
                glm::decompose(ltw.m_value, scale, rotation, translation, skew,
                               perspective);
                const glm::vec3 parentTranslation =
                        entity.GetParent()
                                .GetDataComponent<GlobalTransform>()
                                .GetPosition();
                const auto direction = glm::normalize(parentTranslation - translation);
                rotation = glm::quatLookAt(
                        direction, glm::vec3(direction.y, direction.z, direction.x));
                rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
                const glm::mat4 rotationTransform = glm::mat4_cast(rotation);

                if (entity.GetParent().HasPrivateComponent<TreeData>()) {
                    c.m_value = glm::translate((translation + parentTranslation) / 2.0f) *
                                rotationTransform * glm::scale(glm::vec3(0.0f));
                    branchCylinderWidth.m_value = 0;
                } else {
                    branchCylinderWidth.m_value =
                            displayThickness ? internodeGrowth.m_thickness : width;
                    c.m_value =
                            glm::translate((translation + parentTranslation) / 2.0f) *
                            rotationTransform *
                            glm::scale(glm::vec3(
                                    branchCylinderWidth.m_value,
                                    glm::distance(translation, parentTranslation) / 2.0f,
                                    displayThickness ? internodeGrowth.m_thickness : width));
                }
            },
            false);
}

void TreeSystem::UpdateBranchPointer(const float &length, const float &width) {
    switch (m_pointerRenderType) {
        case PointerRenderType::Illumination:
            EntityManager::ForEach<GlobalTransform, BranchPointer, Illumination>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [length, width](int i, Entity entity, GlobalTransform &ltw,
                                    BranchPointer &c, Illumination &internodeIllumination) {
                        const glm::vec3 start = ltw.GetPosition();
                        const glm::vec3 direction =
                                glm::normalize(internodeIllumination.m_accumulatedDirection);
                        const glm::vec3 end =
                                start +
                                direction * length *
                                glm::length(internodeIllumination.m_accumulatedDirection);
                        const glm::quat rotation =
                                glm::quatLookAt(direction, glm::vec3(0, 0, 1));
                        const glm::mat4 rotationTransform = glm::mat4_cast(
                                rotation * glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f)));
                        c.m_value = glm::translate((start + end) / 2.0f) * rotationTransform *
                                    glm::scale(glm::vec3(
                                            width, glm::distance(start, end) / 2.0f, width));
                    },
                    false);
            break;
        case PointerRenderType::Bending:
            EntityManager::ForEach<GlobalTransform, BranchPointer, InternodeGrowth>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [length, width](int i, Entity entity, GlobalTransform &ltw,
                                    BranchPointer &c, InternodeGrowth &internodeGrowth) {
                        const glm::vec3 start = ltw.GetPosition();
                        const auto target = internodeGrowth.m_childMeanPosition -
                                            internodeGrowth.m_desiredGlobalPosition;
                        const glm::vec3 direction = glm::normalize(target);
                        const glm::vec3 end = start + direction * length *
                                                      internodeGrowth.m_MassOfChildren *
                                                      glm::length(target);
                        const glm::quat rotation =
                                glm::quatLookAt(direction, glm::vec3(0, 0, 1));
                        const glm::mat4 rotationTransform = glm::mat4_cast(
                                rotation * glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f)));
                        c.m_value = glm::translate((start + end) / 2.0f) * rotationTransform *
                                    glm::scale(glm::vec3(
                                            width, glm::distance(start, end) / 2.0f, width));
                    },
                    false);
            break;
    }
}

void TreeSystem::UpdateBranchColors() {
    auto globalTime = m_plantSystem.Get<PlantSystem>()->m_globalTime;

    auto focusingInternode = Entity();
    auto selectedEntity = Entity();
    if (m_currentFocusingInternode.Get().IsValid()) {
        focusingInternode = m_currentFocusingInternode.Get();
    }
    if (EditorManager::GetSelectedEntity().IsValid()) {
        selectedEntity = EditorManager::GetSelectedEntity();
    }
#pragma region Process internode color
    switch (m_branchRenderType) {
        case BranchRenderType::Illumination: {
            EntityManager::ForEach<BranchColor, Illumination, InternodeInfo>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        Illumination &illumination, InternodeInfo &internodeInfo) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        auto value = illumination.m_currentIntensity;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);
        }
            break;
        case BranchRenderType::Inhibitor: {
            EntityManager::ForEach<BranchColor, InternodeGrowth>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeGrowth &internodeGrowth) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        auto value = internodeGrowth.m_inhibitor;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);
        }
            break;
        case BranchRenderType::Sagging: {
            EntityManager::ForEach<BranchColor, InternodeGrowth>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeGrowth &internodeGrowth) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        auto value = internodeGrowth.m_sagging;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);
        }
            break;
        case BranchRenderType::InhibitorTransmitFactor: {
            EntityManager::ForEach<BranchColor, InternodeGrowth>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeGrowth &internodeGrowth) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        auto value = internodeGrowth.m_inhibitorTransmitFactor;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);

        }
            break;
        case BranchRenderType::ResourceToGrow: {
            EntityManager::ForEach<BranchColor, InternodeGrowth>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeGrowth &internodeGrowth) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        auto internodeData =
                                entity.GetOrSetPrivateComponent<InternodeData>().lock();
                        float totalResource = 0;
                        for (const auto &bud : internodeData->m_buds)
                            totalResource += bud.m_currentResource.m_nutrient;
                        float value = totalResource;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);

        }
            break;
        case BranchRenderType::Order: {
            EntityManager::ForEach<BranchColor, InternodeInfo>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeInfo &internodeInfo) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        float value = internodeInfo.m_order;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);

        }
            break;
        case BranchRenderType::MaxChildOrder: {
            EntityManager::ForEach<BranchColor, InternodeStatistics>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        float value = internodeStatistics.m_maxChildOrder;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);

        }
            break;
        case BranchRenderType::Level: {
            EntityManager::ForEach<BranchColor, InternodeInfo>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeInfo &internodeInfo) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        float value = internodeInfo.m_level;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);

        }
            break;
        case BranchRenderType::MaxChildLevel: {
            EntityManager::ForEach<BranchColor, InternodeStatistics>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        float value = internodeStatistics.m_maxChildLevel;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);

        }
            break;
        case BranchRenderType::IsMaxChild: {
            EntityManager::ForEach<BranchColor, InternodeStatistics>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        auto value = internodeStatistics.m_isMaxChild ? 1.0f : 0.2f;
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);
        }
            break;
        case BranchRenderType::ChildrenEndNodeAmount: {
            EntityManager::ForEach<BranchColor, InternodeStatistics>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        float value = internodeStatistics.m_childrenEndNodeAmount;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);

        }
            break;
        case BranchRenderType::IsEndNode: {
            EntityManager::ForEach<BranchColor, InternodeStatistics>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        auto value = internodeStatistics.m_isEndNode ? 1.0f : 0.2f;
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);
        }
            break;
        case BranchRenderType::DistanceToBranchEnd: {
            EntityManager::ForEach<BranchColor, InternodeStatistics>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        float value = internodeStatistics.m_distanceToBranchEnd;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);
        }
            break;
        case BranchRenderType::DistanceToBranchStart: {
            EntityManager::ForEach<BranchColor, InternodeStatistics>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        float value = internodeStatistics.m_distanceToBranchStart;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);
        }
            break;
        case BranchRenderType::TotalLength: {
            EntityManager::ForEach<BranchColor, InternodeStatistics>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        float value = internodeStatistics.m_totalLength;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);
        }
            break;
        case BranchRenderType::LongestDistanceToAnyEndNode: {
            EntityManager::ForEach<BranchColor, InternodeStatistics>(
                    JobManager::PrimaryWorkers(),
                    m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                    [=](int i, Entity entity, BranchColor &internodeRenderColor,
                        InternodeStatistics &internodeStatistics) {
                        if (focusingInternode == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
                            return;
                        }
                        if (selectedEntity == entity) {
                            internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
                            return;
                        }
                        float value = internodeStatistics.m_longestDistanceToAnyEndNode;
                        if (m_enableBranchDataCompress)
                            value = glm::pow(value, m_branchCompressFactor);
                        ColorSet(internodeRenderColor.m_value, value);
                    },
                    false);
        }
            break;
    }
#pragma endregion
}

void TreeSystem::ColorSet(glm::vec4 &target, const float &value) {
    if (m_useColorMap) {
        int compareResult = -1;
        for (int i = 0; i < m_colorMapValues.size(); i++) {
            if (value > m_colorMapValues[i])
                compareResult = i;
        }
        glm::vec3 color;
        if (compareResult == -1) {
            color = m_colorMapColors[0];
        } else if (compareResult == m_colorMapValues.size() - 1) {
            color = m_colorMapColors.back();
        } else {
            const auto value1 = m_colorMapValues[compareResult];
            const auto value2 = m_colorMapValues[compareResult + 1];
            const auto color1 = m_colorMapColors[compareResult];
            const auto color2 = m_colorMapColors[compareResult + 1];
            const auto left = value - value1;
            const auto right = value2 - value1;
            color = color1 * left / right + color2 * (1.0f - left / right);
        }
        if (m_useTransparency)
            target = glm::vec4(color.x, color.y, color.z, m_transparency);
        else
            target = glm::vec4(color.x, color.y, color.z, 1.0f);
    } else {
        if (m_useTransparency)
            target = glm::vec4(value, value, value, m_transparency);
        else
            target = glm::vec4(value, value, value, 1.0f);
    }
}

void TreeSystem::Update() {
    if (m_rightMouseButtonHold &&
        !InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT,
                                        WindowManager::GetWindow())) {
        m_rightMouseButtonHold = false;
        m_startMouse = false;
    }
    m_internodeDebuggingCamera->ResizeResolution(
            m_internodeDebuggingCameraResolutionX,
            m_internodeDebuggingCameraResolutionY);
    m_internodeDebuggingCamera->Clear();

#pragma region Internode debug camera
    Camera::m_cameraInfoBlock.UpdateMatrices(
            EditorManager::GetInstance().m_sceneCamera,
            EditorManager::GetInstance().m_sceneCameraPosition,
            EditorManager::GetInstance().m_sceneCameraRotation);
    Camera::m_cameraInfoBlock.UploadMatrices(
            EditorManager::GetInstance().m_sceneCamera);
#pragma endregion

    bool needUpdate = false;
    if (m_plantSystem.Get<PlantSystem>()->m_globalTime != m_previousGlobalTime) {
        m_previousGlobalTime = m_plantSystem.Get<PlantSystem>()->m_globalTime;
        if (m_displayTime != m_previousGlobalTime) {
            m_displayTime = m_previousGlobalTime;
            needUpdate = true;
        }
    }
#pragma region Rendering
    if (m_drawBranches) {
        if (m_alwaysUpdate || m_updateBranch || needUpdate) {
            m_updateBranch = false;
            UpdateBranchColors();
            UpdateBranchCylinder(m_displayThickness, m_connectionWidth);
        }
        if (m_internodeDebuggingCamera->IsEnabled())
            RenderBranchCylinders(m_displayTime);
    }
    if (m_drawPointers) {
        if (m_alwaysUpdate || m_updatePointer || needUpdate) {
            m_updatePointer = false;
            UpdateBranchPointer(m_pointerLength, m_pointerWidth);
        }
        if (m_internodeDebuggingCamera->IsEnabled())
            RenderBranchPointers(m_displayTime);
    }
#pragma endregion
}

Entity TreeSystem::CreateTree(const Transform &transform) {
    if (Application::IsPlaying()) {
        UNIENGINE_ERROR("Pause game to create tree!");
        return Entity();
    }
    const auto plant = m_plantSystem.Get<PlantSystem>()->CreatePlant(
            PlantType::GeneralTree, transform);
    const auto rootInternode = plant.GetChildren()[0];
    auto rigidBody = rootInternode.GetOrSetPrivateComponent<RigidBody>().lock();
    rigidBody->SetKinematic(true);
    // The rigidbody can only apply mesh bound after it's attached to an entity
    // with mesh renderer.
    rigidBody->SetEnabled(true);
    rigidBody->SetEnableGravity(false);
    plant.SetParent(m_plantSystem.Get<PlantSystem>()->m_anchor.Get());

    auto animator = plant.GetOrSetPrivateComponent<Animator>().lock();
    animator->Setup(AssetManager::CreateAsset<Animation>());

    GetLeaves(plant);
    GetRbv(plant);
    auto treeData = plant.GetOrSetPrivateComponent<TreeData>().lock();
    treeData->m_parameters = TreeParameters();

    auto meshRenderer = plant.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer->m_mesh = AssetManager::CreateAsset<Mesh>();
    auto mat =
            AssetManager::LoadMaterial(DefaultResources::GLPrograms::StandardProgram);
    meshRenderer->m_material = mat;
    mat->m_albedoColor = glm::vec3(0.7f, 0.3f, 0.0f);
    mat->m_roughness = 1.0f;
    mat->m_metallic = 0.0f;
    mat->SetTexture(TextureType::Normal,
                    m_defaultBranchNormalTexture.Get<Texture2D>());
    mat->SetTexture(TextureType::Albedo,
                    m_defaultBranchAlbedoTexture.Get<Texture2D>());
    meshRenderer->SetEnabled(true);

    auto skinnedMeshRenderer =
            plant.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
    skinnedMeshRenderer->m_skinnedMesh = AssetManager::CreateAsset<SkinnedMesh>();
    auto animation =
            plant.GetOrSetPrivateComponent<Animator>().lock()->GetAnimation();
    skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>()->m_animation =
            animation;

    auto skinnedMat = AssetManager::LoadMaterial(
            DefaultResources::GLPrograms::StandardSkinnedProgram);
    skinnedMeshRenderer->m_material = skinnedMat;
    skinnedMat->m_albedoColor = glm::vec3(0.7f, 0.3f, 0.0f);
    skinnedMat->m_roughness = 1.0f;
    skinnedMat->m_metallic = 0.0f;
    skinnedMat->SetTexture(TextureType::Normal,
                           m_defaultBranchNormalTexture.Get<Texture2D>());
    skinnedMat->SetTexture(TextureType::Albedo,
                           m_defaultBranchAlbedoTexture.Get<Texture2D>());
    skinnedMeshRenderer->AttachAnimator(
            plant.GetOrSetPrivateComponent<Animator>().lock());

    auto rtt =
            plant.GetOrSetPrivateComponent<RayTracerFacility::RayTracedRenderer>()
                    .lock();
    rtt->m_albedoTexture = m_defaultRayTracingBranchAlbedoTexture;
    rtt->m_normalTexture = m_defaultRayTracingBranchNormalTexture;
    rtt->m_mesh = plant.GetOrSetPrivateComponent<MeshRenderer>().lock()->m_mesh;

    return plant;
}

const char *BranchRenderTypes[]{"Illumination",
                                "Sagging",
                                "Inhibitor",
                                "InhibitorTransmitFactor",
                                "ResourceToGrow",
                                "Order",
                                "MaxChildOrder",
                                "Level",
                                "MaxChildLevel",
                                "IsMaxChild",
                                "ChildrenEndNodeAmount",
                                "IsEndNode",
                                "DistanceToBranchEnd",
                                "DistanceToBranchStart",
                                "TotalLength",
                                "LongestDistanceToAnyEndNode"

};

const char *PointerRenderTypes[]{"Illumination", "Bending"};

void TreeSystem::OnInspect() {
    if (!m_ready) {
        ImGui::Text("System not ready!");
        return;
    }

    ImGui::Text("Physics");
    ImGui::DragFloat("Internode Density", &m_density, 0.1f, 0.01f, 1000.0f);
    ImGui::DragFloat2("RigidBody Damping", &m_linearDamping, 0.1f, 0.01f,
                      1000.0f);
    ImGui::DragFloat2("Drive Stiffness", &m_jointDriveStiffnessFactor, 0.1f,
                      0.01f, 1000000.0f);
    ImGui::DragFloat2("Drive Damping", &m_jointDriveDampingFactor, 0.1f, 0.01f,
                      1000000.0f);
    ImGui::Checkbox("Use acceleration", &m_enableAccelerationForDrive);

    int pi = m_positionSolverIteration;
    int vi = m_velocitySolverIteration;
    if (ImGui::DragInt("Velocity solver iteration", &vi, 1, 1, 100)) {
        m_velocitySolverIteration = vi;
    }
    if (ImGui::DragInt("Position solver iteration", &pi, 1, 1, 100)) {
        m_positionSolverIteration = pi;
    }
    ImGui::Separator();
    ImGui::Text("Foliage");
    ImGui::DragInt("Leaf amount", &m_leafAmount, 0, 0, 50);
    ImGui::DragFloat("Generation radius", &m_radius, 0.01, 0.01, 10);
    ImGui::DragFloat("Generation distance", &m_distanceToEndNode, 0.01, 0.01, 20);

    ImGui::DragFloat2("Leaf size", &m_leafSize.x, 0.01, 0.01, 10);
    ImGui::Separator();
    ImGui::Text("Crown shyness");
    ImGui::DragFloat("Crown shyness D", &m_crownShynessDiameter, 0.01f, 0.0f,
                     2.0f);
    if (m_crownShynessDiameter > m_voxelSpaceModule.GetDiameter())
        Debug::Error("Diameter too large!");

    ImGui::Separator();
    ImGui::Text("Metadata");
    if (ImGui::Button("Update metadata")) {
        m_plantSystem.Get<PlantSystem>()->Refresh();
        UpdateTreesMetaData();
    }
    ImGui::Separator();

    if (ImGui::Button("Create...")) {
        ImGui::OpenPopup("New tree wizard");
        Application::SetPlaying(false);
    }

    ImGui::DragFloat("Mesh resolution", &m_meshResolution, 0.001f, 0, 1);
    ImGui::DragFloat("Mesh subdivision", &m_meshSubdivision, 0.001f, 0, 1);
    if (ImGui::Button("Generate mesh")) {
        m_plantSystem.Get<PlantSystem>()->Refresh();
        UpdateTreesMetaData();
        GenerateMeshForTree();
    }
    if (ImGui::Button("Generate skinned mesh")) {
        GenerateSkinnedMeshForTree();
    }
    FileUtils::SaveFile("Save scene as XML", "Scene XML", {".xml"},
                        [this](const std::filesystem::path &path) {
                            SerializeScene(path.string());
                        });
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("New tree wizard", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        static std::vector<TreeParameters> newTreeParameters;
        static std::vector<glm::vec3> newTreePositions;
        static std::vector<glm::vec3> newTreeRotations;
        static int newTreeAmount = 1;
        static int currentFocusedNewTreeIndex = 0;
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
        ImGui::BeginChild("ChildL", ImVec2(300, 400), true,
                          ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Settings")) {
                static float distance = 10;
                static float variance = 4;
                static float yAxisVar = 180.0f;
                static float xzAxisVar = 0.0f;
                static int expand = 1;
                if (ImGui::BeginMenu("Create forest...")) {
                    ImGui::DragFloat("Avg. Y axis rotation", &yAxisVar, 0.01f, 0.0f,
                                     180.0f);
                    ImGui::DragFloat("Avg. XZ axis rotation", &xzAxisVar, 0.01f, 0.0f,
                                     90.0f);
                    ImGui::DragFloat("Avg. Distance", &distance, 0.01f);
                    ImGui::DragFloat("Position variance", &variance, 0.01f);
                    ImGui::DragInt("Expand", &expand, 1, 0, 3);
                    if (ImGui::Button("Apply")) {
                        newTreeAmount = (2 * expand + 1) * (2 * expand + 1);
                        newTreePositions.resize(newTreeAmount);
                        newTreeRotations.resize(newTreeAmount);
                        const auto currentSize = newTreeParameters.size();
                        newTreeParameters.resize(newTreeAmount);
                        for (auto i = currentSize; i < newTreeAmount; i++) {
                            newTreeParameters[i] = newTreeParameters[0];
                        }
                        int index = 0;
                        for (int i = -expand; i <= expand; i++) {
                            for (int j = -expand; j <= expand; j++) {
                                glm::vec3 value = glm::vec3(i * distance, 0, j * distance);
                                value.x += glm::linearRand(-variance, variance);
                                value.z += glm::linearRand(-variance, variance);
                                newTreePositions[index] = value;
                                value = glm::vec3(glm::linearRand(-xzAxisVar, xzAxisVar),
                                                  glm::linearRand(-yAxisVar, yAxisVar),
                                                  glm::linearRand(-xzAxisVar, xzAxisVar));
                                newTreeRotations[index] = value;
                                index++;
                            }
                        }
                    }
                    ImGui::EndMenu();
                }
                ImGui::InputInt("New Tree Amount", &newTreeAmount);
                if (newTreeAmount < 1)
                    newTreeAmount = 1;
                FileUtils::OpenFile("Import parameters for all", "Tree Params", {".treeparam"},
                                    [](const std::filesystem::path &path) {
                                        newTreeParameters[0].Deserialize(path.string());
                                        for (int i = 1; i < newTreeParameters.size(); i++)
                                            newTreeParameters[i] = newTreeParameters[0];
                                    });
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
        ImGui::Columns(1);
        if (newTreePositions.size() < newTreeAmount) {
            if (newTreeParameters.empty()) {
                newTreeParameters.resize(1);
                newTreeParameters[0].Deserialize(
                        std::filesystem::path(PLANT_ARCHITECT_RESOURCE_FOLDER) /
                        "Parameters/default.treeparam");
            }
            const auto currentSize = newTreePositions.size();
            newTreeParameters.resize(newTreeAmount);
            for (auto i = currentSize; i < newTreeAmount; i++) {
                newTreeParameters[i] = newTreeParameters[0];
            }
            newTreePositions.resize(newTreeAmount);
            newTreeRotations.resize(newTreeAmount);
        }
        for (auto i = 0; i < newTreeAmount; i++) {
            std::string title = "New Tree No.";
            title += std::to_string(i);
            const bool opened = ImGui::TreeNodeEx(
                    title.c_str(), ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                   ImGuiTreeNodeFlags_OpenOnArrow |
                                   ImGuiTreeNodeFlags_NoAutoOpenOnLog |
                                   (currentFocusedNewTreeIndex == i
                                    ? ImGuiTreeNodeFlags_Framed
                                    : ImGuiTreeNodeFlags_FramePadding));
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                currentFocusedNewTreeIndex = i;
            }
            if (opened) {
                ImGui::TreePush();
                ImGui::InputFloat3(("Position##" + std::to_string(i)).c_str(),
                                   &newTreePositions[i].x);
                ImGui::TreePop();
            }
        }

        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::SameLine();
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
        ImGui::BeginChild("ChildR", ImVec2(400, 400), true,
                          ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Parameters")) {
                FileUtils::OpenFile(
                        "Import parameters", "Tree Params", {".treeparam"},
                        [](const std::filesystem::path &path) {
                            newTreeParameters[currentFocusedNewTreeIndex].Deserialize(
                                    path.string());
                        });

                FileUtils::SaveFile(
                        "Export parameters", "Tree Params", {".treeparam"},
                        [](const std::filesystem::path &path) {
                            newTreeParameters[currentFocusedNewTreeIndex].Serialize(
                                    path.string());
                        });
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
        ImGui::Columns(1);
        ImGui::PushItemWidth(200);
        newTreeParameters[currentFocusedNewTreeIndex].OnGui();
        ImGui::PopItemWidth();
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::Separator();
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            // Create tree here.
            for (auto i = 0; i < newTreeAmount; i++) {
                Transform treeTransform;
                treeTransform.SetPosition(newTreePositions[i]);
                treeTransform.SetEulerRotation(glm::radians(newTreeRotations[i]));
                Entity tree = CreateTree(treeTransform);
                tree.GetOrSetPrivateComponent<TreeData>().lock()->m_parameters =
                        newTreeParameters[i];
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
#pragma endregion
}

void TreeSystem::RenderBranchCylinders(const float &displayTime) {
    std::vector<BranchCylinder> branchCylinders;
    m_plantSystem.Get<PlantSystem>()
            ->m_internodeQuery.ToComponentDataArray<BranchCylinder, InternodeInfo>(
                    branchCylinders, [displayTime](const InternodeInfo &internodeInfo) {
                        return internodeInfo.m_startGlobalTime <= displayTime;
                    });
    std::vector<BranchColor> branchColors;
    m_plantSystem.Get<PlantSystem>()
            ->m_internodeQuery.ToComponentDataArray<BranchColor, InternodeInfo>(
                    branchColors, [displayTime](const InternodeInfo &internodeInfo) {
                        return internodeInfo.m_startGlobalTime <= displayTime;
                    });
    std::vector<Entity> rootInternodes;
    m_plantSystem.Get<PlantSystem>()
            ->m_internodeQuery.ToEntityArray<InternodeGrowth>(
                    rootInternodes,
                    [displayTime](const Entity &entity,
                                  const InternodeGrowth &internodeGrowth) {
                        return internodeGrowth.m_distanceToRoot == 0;
                    });

    for (const auto &i : rootInternodes) {
        auto gt = i.GetDataComponent<GlobalTransform>();
        float thickness = 0.1f;
        if (i.GetChildrenAmount() > 0)
            thickness = glm::max(
                    0.05f,
                    i.GetChildren()[0].GetDataComponent<InternodeGrowth>().m_thickness);
        RenderManager::DrawGizmoMesh(
                DefaultResources::Primitives::Sphere, m_internodeDebuggingCamera,
                EditorManager::GetInstance().m_sceneCameraPosition,
                EditorManager::GetInstance().m_sceneCameraRotation,
                EditorManager::GetSelectedEntity() == i ? glm::vec4(1)
                                                        : m_currentFocusingInternode.Get() == i ? glm::vec4(0, 0, 1, 1)
                                                                                                : glm::vec4(1, 0, 1, 1),
                gt.m_value, thickness * 2.0f);
    }

    if (!branchCylinders.empty())
        RenderManager::DrawGizmoMeshInstancedColored(
                DefaultResources::Primitives::Cylinder, m_internodeDebuggingCamera,
                EditorManager::GetInstance().m_sceneCameraPosition,
                EditorManager::GetInstance().m_sceneCameraRotation,
                *reinterpret_cast<std::vector<glm::vec4> *>(&branchColors),
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchCylinders),
                glm::mat4(1.0f), 1.0f);
}

void TreeSystem::RenderBranchPointers(const float &displayTime) {
    std::vector<BranchPointer> branchPointers;
    m_plantSystem.Get<PlantSystem>()
            ->m_internodeQuery.ToComponentDataArray<BranchPointer, InternodeInfo>(
                    branchPointers, [displayTime](const InternodeInfo &internodeInfo) {
                        return internodeInfo.m_startGlobalTime <= displayTime;
                    });
    if (!branchPointers.empty())
        RenderManager::DrawGizmoMeshInstanced(
                DefaultResources::Primitives::Cylinder, m_internodeDebuggingCamera,
                EditorManager::GetInstance().m_sceneCameraPosition,
                EditorManager::GetInstance().m_sceneCameraRotation, m_pointerColor,
                *reinterpret_cast<std::vector<glm::mat4> *>(&branchPointers),
                glm::mat4(1.0f), 1.0f);
}

void TreeSystem::TreeNodeWalker(std::vector<Entity> &boundEntities,
                                std::vector<int> &parentIndices,
                                const int &parentIndex, const Entity &node) {
    boundEntities.push_back(node);
    parentIndices.push_back(parentIndex);
    const size_t currentIndex = boundEntities.size() - 1;
    auto info = node.GetDataComponent<InternodeInfo>();
    info.m_index = currentIndex;
    node.SetDataComponent(info);
    node.ForEachChild([&](Entity child) {
        TreeNodeWalker(boundEntities, parentIndices, currentIndex, child);
    });
}

void TreeSystem::TreeMeshGenerator(std::vector<Entity> &internodes,
                                   std::vector<int> &parentIndices,
                                   std::vector<Vertex> &vertices,
                                   std::vector<unsigned> &indices) {
    int parentStep = -1;
    for (int internodeIndex = 1; internodeIndex < internodes.size();
         internodeIndex++) {
        auto &internode = internodes[internodeIndex];
        glm::vec3 newNormalDir = internodes[parentIndices[internodeIndex]]
                .GetOrSetPrivateComponent<InternodeData>()
                .lock()
                ->m_normalDir;
        const glm::vec3 front =
                internode.GetDataComponent<InternodeGrowth>().m_desiredGlobalRotation *
                glm::vec3(0.0f, 0.0f, -1.0f);
        newNormalDir = glm::cross(glm::cross(front, newNormalDir), front);
        auto list = internode.GetOrSetPrivateComponent<InternodeData>().lock();
        if (list->m_rings.empty()) {
            continue;
        }
        auto step = list->m_step;
        // For stitching
        const int pStep = parentStep > 0 ? parentStep : step;
        parentStep = step;
        list->m_normalDir = newNormalDir;
        float angleStep = 360.0f / static_cast<float>(pStep);
        int vertexIndex = vertices.size();
        Vertex archetype;
        float textureXStep = 1.0f / pStep * 4.0f;
        for (int i = 0; i < pStep; i++) {
            archetype.m_position =
                    list->m_rings.at(0).GetPoint(newNormalDir, angleStep * i, true);
            const float x =
                    i < pStep / 2 ? i * textureXStep : (pStep - i) * textureXStep;
            archetype.m_texCoords = glm::vec2(x, 0.0f);
            vertices.push_back(archetype);
        }
        std::vector<float> angles;
        angles.resize(step);
        std::vector<float> pAngles;
        pAngles.resize(pStep);

        for (auto i = 0; i < pStep; i++) {
            pAngles[i] = angleStep * i;
        }
        angleStep = 360.0f / static_cast<float>(step);
        for (auto i = 0; i < step; i++) {
            angles[i] = angleStep * i;
        }

        std::vector<unsigned> pTarget;
        std::vector<unsigned> target;
        pTarget.resize(pStep);
        target.resize(step);
        for (int i = 0; i < pStep; i++) {
            // First we allocate nearest vertices for parent.
            auto minAngleDiff = 360.0f;
            for (auto j = 0; j < step; j++) {
                const float diff = glm::abs(pAngles[i] - angles[j]);
                if (diff < minAngleDiff) {
                    minAngleDiff = diff;
                    pTarget[i] = j;
                }
            }
        }
        for (int i = 0; i < step; i++) {
            // Second we allocate nearest vertices for child
            float minAngleDiff = 360.0f;
            for (int j = 0; j < pStep; j++) {
                const float diff = glm::abs(angles[i] - pAngles[j]);
                if (diff < minAngleDiff) {
                    minAngleDiff = diff;
                    target[i] = j;
                }
            }
        }
        for (int i = 0; i < pStep; i++) {
            if (pTarget[i] == pTarget[i == pStep - 1 ? 0 : i + 1]) {
                indices.push_back(vertexIndex + i);
                indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
                indices.push_back(vertexIndex + pStep + pTarget[i]);
            } else {
                indices.push_back(vertexIndex + i);
                indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
                indices.push_back(vertexIndex + pStep + pTarget[i]);

                indices.push_back(vertexIndex + pStep +
                                  pTarget[i == pStep - 1 ? 0 : i + 1]);
                indices.push_back(vertexIndex + pStep + pTarget[i]);
                indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
            }
        }

        vertexIndex += pStep;
        textureXStep = 1.0f / step * 4.0f;
        const int ringSize = list->m_rings.size();
        for (auto ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            for (auto i = 0; i < step; i++) {
                archetype.m_position = list->m_rings.at(ringIndex).GetPoint(
                        newNormalDir, angleStep * i, false);
                const auto x =
                        i < (step / 2) ? i * textureXStep : (step - i) * textureXStep;
                const auto y = ringIndex % 2 == 0 ? 1.0f : 0.0f;
                archetype.m_texCoords = glm::vec2(x, y);
                vertices.push_back(archetype);
            }
            if (ringIndex != 0) {
                for (int i = 0; i < step - 1; i++) {
                    // Down triangle
                    indices.push_back(vertexIndex + (ringIndex - 1) * step + i);
                    indices.push_back(vertexIndex + (ringIndex - 1) * step + i + 1);
                    indices.push_back(vertexIndex + (ringIndex) * step + i);
                    // Up triangle
                    indices.push_back(vertexIndex + (ringIndex) * step + i + 1);
                    indices.push_back(vertexIndex + (ringIndex) * step + i);
                    indices.push_back(vertexIndex + (ringIndex - 1) * step + i + 1);
                }
                // Down triangle
                indices.push_back(vertexIndex + (ringIndex - 1) * step + step - 1);
                indices.push_back(vertexIndex + (ringIndex - 1) * step);
                indices.push_back(vertexIndex + (ringIndex) * step + step - 1);
                // Up triangle
                indices.push_back(vertexIndex + (ringIndex) * step);
                indices.push_back(vertexIndex + (ringIndex) * step + step - 1);
                indices.push_back(vertexIndex + (ringIndex - 1) * step);
            }
        }
    }
}

void TreeSystem::TreeSkinnedMeshGenerator(std::vector<Entity> &internodes,
                                          std::vector<int> &parentIndices,
                                          std::vector<SkinnedVertex> &vertices,
                                          std::vector<unsigned> &indices) {
    int parentStep = -1;
    for (int internodeIndex = 1; internodeIndex < internodes.size();
         internodeIndex++) {
        auto &internode = internodes[internodeIndex];
        glm::vec3 newNormalDir = internodes[parentIndices[internodeIndex]]
                .GetOrSetPrivateComponent<InternodeData>()
                .lock()
                ->m_normalDir;
        const glm::vec3 front =
                internode.GetDataComponent<InternodeGrowth>().m_desiredGlobalRotation *
                glm::vec3(0.0f, 0.0f, -1.0f);
        newNormalDir = glm::cross(glm::cross(front, newNormalDir), front);
        auto list = internode.GetOrSetPrivateComponent<InternodeData>().lock();
        if (list->m_rings.empty()) {
            continue;
        }
        auto step = list->m_step;
        // For stitching
        const int pStep = parentStep > 0 ? parentStep : step;
        parentStep = step;
        list->m_normalDir = newNormalDir;
        float angleStep = 360.0f / static_cast<float>(pStep);
        int vertexIndex = vertices.size();
        SkinnedVertex archetype;
        float textureXStep = 1.0f / pStep * 4.0f;

        const auto startPosition = list->m_rings.at(0).m_startPosition;
        const auto endPosition = list->m_rings.back().m_endPosition;
        for (int i = 0; i < pStep; i++) {
            archetype.m_position =
                    list->m_rings.at(0).GetPoint(newNormalDir, angleStep * i, true);

            float distanceToStart = 0;
            float distanceToEnd = 1;
            archetype.m_bondId =
                    glm::ivec4(internodeIndex, parentIndices[internodeIndex], -1, -1);
            archetype.m_bondId2 = glm::ivec4(-1, -1, -1, -1);
            archetype.m_weight = glm::vec4(
                    distanceToStart / (distanceToStart + distanceToEnd),
                    distanceToEnd / (distanceToStart + distanceToEnd), 0.0f, 0.0f);
            archetype.m_weight2 = glm::vec4(0.0f);

            const float x =
                    i < pStep / 2 ? i * textureXStep : (pStep - i) * textureXStep;
            archetype.m_texCoords = glm::vec2(x, 0.0f);
            vertices.push_back(archetype);
        }
        std::vector<float> angles;
        angles.resize(step);
        std::vector<float> pAngles;
        pAngles.resize(pStep);

        for (auto i = 0; i < pStep; i++) {
            pAngles[i] = angleStep * i;
        }
        angleStep = 360.0f / static_cast<float>(step);
        for (auto i = 0; i < step; i++) {
            angles[i] = angleStep * i;
        }

        std::vector<unsigned> pTarget;
        std::vector<unsigned> target;
        pTarget.resize(pStep);
        target.resize(step);
        for (int i = 0; i < pStep; i++) {
            // First we allocate nearest vertices for parent.
            auto minAngleDiff = 360.0f;
            for (auto j = 0; j < step; j++) {
                const float diff = glm::abs(pAngles[i] - angles[j]);
                if (diff < minAngleDiff) {
                    minAngleDiff = diff;
                    pTarget[i] = j;
                }
            }
        }
        for (int i = 0; i < step; i++) {
            // Second we allocate nearest vertices for child
            float minAngleDiff = 360.0f;
            for (int j = 0; j < pStep; j++) {
                const float diff = glm::abs(angles[i] - pAngles[j]);
                if (diff < minAngleDiff) {
                    minAngleDiff = diff;
                    target[i] = j;
                }
            }
        }
        for (int i = 0; i < pStep; i++) {
            if (pTarget[i] == pTarget[i == pStep - 1 ? 0 : i + 1]) {
                indices.push_back(vertexIndex + i);
                indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
                indices.push_back(vertexIndex + pStep + pTarget[i]);
            } else {
                indices.push_back(vertexIndex + i);
                indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
                indices.push_back(vertexIndex + pStep + pTarget[i]);

                indices.push_back(vertexIndex + pStep +
                                  pTarget[i == pStep - 1 ? 0 : i + 1]);
                indices.push_back(vertexIndex + pStep + pTarget[i]);
                indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
            }
        }

        vertexIndex += pStep;
        textureXStep = 1.0f / step * 4.0f;
        const int ringSize = list->m_rings.size();
        for (auto ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            for (auto i = 0; i < step; i++) {
                archetype.m_position = list->m_rings.at(ringIndex).GetPoint(
                        newNormalDir, angleStep * i, false);

                float distanceToStart = glm::distance(
                        list->m_rings.at(ringIndex).m_endPosition, startPosition);
                float distanceToEnd = glm::distance(
                        list->m_rings.at(ringIndex).m_endPosition, endPosition);
                archetype.m_bondId =
                        glm::ivec4(internodeIndex, parentIndices[internodeIndex], -1, -1);
                archetype.m_bondId2 = glm::ivec4(-1, -1, -1, -1);
                archetype.m_weight = glm::vec4(
                        distanceToStart / (distanceToStart + distanceToEnd),
                        distanceToEnd / (distanceToStart + distanceToEnd), 0.0f, 0.0f);
                archetype.m_weight2 = glm::vec4(0.0f);

                const auto x =
                        i < (step / 2) ? i * textureXStep : (step - i) * textureXStep;
                const auto y = ringIndex % 2 == 0 ? 1.0f : 0.0f;
                archetype.m_texCoords = glm::vec2(x, y);
                vertices.push_back(archetype);
            }
            if (ringIndex != 0) {
                for (int i = 0; i < step - 1; i++) {
                    // Down triangle
                    indices.push_back(vertexIndex + (ringIndex - 1) * step + i);
                    indices.push_back(vertexIndex + (ringIndex - 1) * step + i + 1);
                    indices.push_back(vertexIndex + (ringIndex) * step + i);
                    // Up triangle
                    indices.push_back(vertexIndex + (ringIndex) * step + i + 1);
                    indices.push_back(vertexIndex + (ringIndex) * step + i);
                    indices.push_back(vertexIndex + (ringIndex - 1) * step + i + 1);
                }
                // Down triangle
                indices.push_back(vertexIndex + (ringIndex - 1) * step + step - 1);
                indices.push_back(vertexIndex + (ringIndex - 1) * step);
                indices.push_back(vertexIndex + (ringIndex) * step + step - 1);
                // Up triangle
                indices.push_back(vertexIndex + (ringIndex) * step);
                indices.push_back(vertexIndex + (ringIndex) * step + step - 1);
                indices.push_back(vertexIndex + (ringIndex - 1) * step);
            }
        }
    }
}

void TreeSystem::GenerateMeshForTree() {
    if (m_meshResolution <= 0.0f) {
        Debug::Error("TreeSystem: Resolution must be larger than 0!");
        return;
    }
    int plantSize = m_plantSystem.Get<PlantSystem>()->m_plants.size();
    std::vector<std::vector<Entity>> boundEntitiesLists;
    std::vector<std::vector<unsigned>> boneIndicesLists;
    std::vector<std::vector<int>> parentIndicesLists;
    boundEntitiesLists.resize(plantSize);
    boneIndicesLists.resize(plantSize);
    parentIndicesLists.resize(plantSize);
    EntityManager::ForEach<PlantInfo, GlobalTransform>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_plantQuery,
            [&](int i, Entity tree, PlantInfo &plantInfo,
                GlobalTransform &globalTransform) {
                if (plantInfo.m_plantType != PlantType::GeneralTree)
                    return;
                const Entity rootInternode = GetRootInternode(tree);
                if (rootInternode.IsValid()) {
                    for (int plantIndex = 0;
                         plantIndex < m_plantSystem.Get<PlantSystem>()->m_plants.size();
                         plantIndex++) {
                        if (m_plantSystem.Get<PlantSystem>()->m_plants[plantIndex] ==
                            tree) {
                            const auto plantGlobalTransform =
                                    tree.GetDataComponent<GlobalTransform>();
                            TreeNodeWalker(boundEntitiesLists[plantIndex],
                                           parentIndicesLists[plantIndex], -1, rootInternode);
                        }
                    }
                }
                Entity leaves = GetLeaves(tree);
                if (leaves.IsValid()) {
                    leaves.GetOrSetPrivateComponent<TreeLeaves>()
                            .lock()
                            ->m_transforms.clear();
                    leaves.GetOrSetPrivateComponent<TreeLeaves>()
                            .lock()
                            ->m_targetBoneIndices.clear();
                }
            },
            false);
#pragma region Prepare rings for branch mesh.
    EntityManager::ForEach<GlobalTransform, Transform, InternodeGrowth,
            InternodeInfo>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
            [this](int i, Entity internode, GlobalTransform &globalTransform,
                   Transform &transform, InternodeGrowth &internodeGrowth,
                   InternodeInfo &internodeInfo) {
                if (internodeInfo.m_plantType != PlantType::GeneralTree)
                    return;
                auto internodeData =
                        internode.GetOrSetPrivateComponent<InternodeData>().lock();
                auto plant = internodeData->m_plant.Get();
                auto thickestChild = internodeData->m_thickestChild.Get();
                const Entity parent = internode.GetParent();
                if (parent == plant)
                    return;
                bool isRootInternode = false;
                if (parent.GetParent() == plant)
                    isRootInternode = true;
                internodeData->m_rings.clear();
                glm::mat4 treeTransform =
                        plant.GetDataComponent<GlobalTransform>().m_value;
                GlobalTransform parentGlobalTransform;
                parentGlobalTransform.m_value =
                        glm::inverse(treeTransform) *
                        parent.GetDataComponent<GlobalTransform>().m_value;
                float parentThickness =
                        isRootInternode
                        ? internodeGrowth.m_thickness * 1.25f
                        : parent.GetDataComponent<InternodeGrowth>().m_thickness;
                glm::vec3 parentScale;
                glm::quat parentRotation;
                glm::vec3 parentTranslation;
                parentGlobalTransform.Decompose(parentTranslation, parentRotation,
                                                parentScale);

                glm::vec3 scale;
                glm::quat rotation;
                glm::vec3 translation;
                GlobalTransform copyGT;
                copyGT.m_value = glm::inverse(treeTransform) * globalTransform.m_value;
                copyGT.Decompose(translation, rotation, scale);

                glm::vec3 parentDir = isRootInternode
                                      ? glm::vec3(0, 1, 0)
                                      : parentRotation * glm::vec3(0, 0, -1);
                glm::vec3 dir = rotation * glm::vec3(0, 0, -1);
                glm::quat mainChildRotation = rotation;
                if (!thickestChild.IsNull()) {
                    GlobalTransform thickestChildTransform;
                    thickestChildTransform.m_value =
                            glm::inverse(treeTransform) *
                            thickestChild.GetDataComponent<GlobalTransform>().m_value;
                    mainChildRotation = thickestChildTransform.GetRotation();
                }
                glm::vec3 mainChildDir = mainChildRotation * glm::vec3(0, 0, -1);
                GlobalTransform parentThickestChildGlobalTransform;
                auto parentInternodeData =
                        parent.GetOrSetPrivateComponent<InternodeData>().lock();
                auto parentThickestChild = parentInternodeData->m_thickestChild.Get();
                parentThickestChildGlobalTransform.m_value =
                        glm::inverse(treeTransform) *
                        parentThickestChild.GetDataComponent<GlobalTransform>().m_value;
                glm::vec3 parentMainChildDir =
                        parentThickestChildGlobalTransform.GetRotation() *
                        glm::vec3(0, 0, -1);
                glm::vec3 fromDir = isRootInternode
                                    ? parentDir
                                    : (parentDir + parentMainChildDir) / 2.0f;
                dir = (dir + mainChildDir) / 2.0f;
#pragma region Subdivision internode here.
                auto distance = glm::distance(parentTranslation, translation);

                int step = parentThickness / m_meshResolution;
                if (step < 4)
                    step = 4;
                if (step % 2 != 0)
                    step++;
                internodeData->m_step = step;
                int amount = static_cast<int>(0.5f + distance * m_meshSubdivision);
                if (amount % 2 != 0)
                    amount++;
                BezierCurve curve = BezierCurve(
                        parentTranslation, parentTranslation + distance / 3.0f * fromDir,
                        translation - distance / 3.0f * dir, translation);
                float posStep = 1.0f / static_cast<float>(amount);
                glm::vec3 dirStep = (dir - fromDir) / static_cast<float>(amount);
                float radiusStep = (internodeGrowth.m_thickness - parentThickness) /
                                   static_cast<float>(amount);

                for (int i = 1; i < amount; i++) {
                    float startThickness = static_cast<float>(i - 1) * radiusStep;
                    float endThickness = static_cast<float>(i) * radiusStep;
                    internodeData->m_rings.emplace_back(
                            curve.GetPoint(posStep * (i - 1)), curve.GetPoint(posStep * i),
                            fromDir + static_cast<float>(i - 1) * dirStep,
                            fromDir + static_cast<float>(i) * dirStep,
                            parentThickness + startThickness, parentThickness + endThickness);
                }
                if (amount > 1)
                    internodeData->m_rings.emplace_back(
                            curve.GetPoint(1.0f - posStep), translation, dir - dirStep, dir,
                            internodeGrowth.m_thickness - radiusStep,
                            internodeGrowth.m_thickness);
                else
                    internodeData->m_rings.emplace_back(parentTranslation, translation,
                                                        fromDir, dir, parentThickness,
                                                        internodeGrowth.m_thickness);
#pragma endregion
            }

    );
#pragma endregion
#pragma region Prepare leaf transforms.
    std::mutex mutex;
    EntityManager::ForEach<GlobalTransform, InternodeInfo, InternodeStatistics,
            Illumination>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
            [&](int index, Entity internode, GlobalTransform &globalTransform,
                InternodeInfo &internodeInfo,
                InternodeStatistics &internodeStatistics,
                Illumination &internodeIllumination) {
                auto internodeData =
                        internode.GetOrSetPrivateComponent<InternodeData>().lock();
                auto plant = internodeData->m_plant.Get();
                // auto thickestChild = internodeData->m_thickestChild.Get();
                if (!plant.IsEnabled())
                    return;
                if (internodeInfo.m_plantType != PlantType::GeneralTree)
                    return;
                if (internodeStatistics.m_longestDistanceToAnyEndNode >
                    m_distanceToEndNode)
                    return;
                auto treeLeaves =
                        GetLeaves(plant).GetOrSetPrivateComponent<TreeLeaves>().lock();

                internodeData->m_leavesTransforms.clear();
                const glm::quat rotation = globalTransform.GetRotation();
                const glm::vec3 left = rotation * glm::vec3(1, 0, 0);
                const glm::vec3 right = rotation * glm::vec3(-1, 0, 0);
                const glm::vec3 up = rotation * glm::vec3(0, 1, 0);
                std::lock_guard lock(mutex);
                auto inversePlantGlobalTransform =
                        glm::inverse(plant.GetDataComponent<GlobalTransform>().m_value);
                for (int i = 0; i < m_leafAmount; i++) {
                    const auto transform =
                            inversePlantGlobalTransform * globalTransform.m_value *
                            (glm::translate(
                                    glm::linearRand(glm::vec3(-m_radius), glm::vec3(m_radius))) *
                             glm::mat4_cast(glm::quat(glm::radians(
                                     glm::linearRand(glm::vec3(0.0f), glm::vec3(360.0f))))) *
                             glm::scale(glm::vec3(m_leafSize.x, 1.0f, m_leafSize.y)));
                    internodeData->m_leavesTransforms.push_back(transform);
                    treeLeaves->m_transforms.push_back(transform);
                    treeLeaves->m_targetBoneIndices.push_back(internodeInfo.m_index);
                }
                /*
                internodeData->m_leavesTransforms.push_back(globalTransform.m_value
           *
                        (
                                glm::translate(left * 0.1f) *
           glm::mat4_cast(glm::quatLookAt(-left, up)) *
           glm::scale(glm::vec3(0.1f))
                                )
                );
                internodeData->m_leavesTransforms.push_back(globalTransform.m_value
           *
                        (
                                glm::translate(right * 0.1f) *
           glm::mat4_cast(glm::quatLookAt(-right, up)) *
           glm::scale(glm::vec3(0.1f))
                                )
                );
                */
            });
#pragma endregion
    for (int plantIndex = 0;
         plantIndex < m_plantSystem.Get<PlantSystem>()->m_plants.size();
         plantIndex++) {
        const auto &plant = m_plantSystem.Get<PlantSystem>()->m_plants[plantIndex];
        if (plant.GetDataComponent<PlantInfo>().m_plantType !=
            PlantType::GeneralTree)
            continue;
        if (!plant.HasPrivateComponent<TreeData>())
            continue;

        if (plant.HasPrivateComponent<SkinnedMeshRenderer>()) {
            plant.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock()->SetEnabled(
                    false);
        }
        auto treeData = plant.GetOrSetPrivateComponent<TreeData>().lock();
        if (Entity rootInternode = GetRootInternode(plant);
                !rootInternode.IsNull()) {
            const auto plantGlobalTransform =
                    plant.GetDataComponent<GlobalTransform>();
#pragma region Branch mesh
            if (rootInternode.GetChildrenAmount() != 0) {
                std::vector<unsigned> indices;
                std::vector<Vertex> vertices;
                TreeMeshGenerator(boundEntitiesLists[plantIndex],
                                  parentIndicesLists[plantIndex], vertices, indices);
                treeData->m_branchMesh.Get<Mesh>()->SetVertices(17, vertices, indices);
                treeData->m_meshGenerated = true;
                if (plant.HasPrivateComponent<MeshRenderer>()) {
                    plant.GetOrSetPrivateComponent<MeshRenderer>().lock()->m_mesh =
                            treeData->m_branchMesh;
                }
                if (plant.HasPrivateComponent<RayTracerFacility::RayTracedRenderer>()) {
                    plant.GetOrSetPrivateComponent<RayTracerFacility::RayTracedRenderer>()
                            .lock()
                            ->m_mesh = treeData->m_branchMesh;
                }
            }
#pragma endregion
            Entity leaves = GetLeaves(plant);
            if (leaves.IsValid())
                leaves.GetOrSetPrivateComponent<TreeLeaves>().lock()->FormMesh();
        }
    }
}

void TreeSystem::GenerateSkinnedMeshForTree() {
    if (m_meshResolution <= 0.0f) {
        Debug::Error("TreeSystem: Resolution must be larger than 0!");
        return;
    }
    int plantSize = m_plantSystem.Get<PlantSystem>()->m_plants.size();
    std::vector<std::vector<Entity>> boundEntitiesLists;
    std::vector<std::vector<unsigned>> boneIndicesLists;
    std::vector<std::vector<int>> parentIndicesLists;
    boundEntitiesLists.resize(plantSize);
    boneIndicesLists.resize(plantSize);
    parentIndicesLists.resize(plantSize);
    EntityManager::ForEach<PlantInfo, GlobalTransform>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_plantQuery,
            [&](int i, Entity tree, PlantInfo &plantInfo,
                GlobalTransform &globalTransform) {
                if (plantInfo.m_plantType != PlantType::GeneralTree)
                    return;
                const Entity rootInternode = GetRootInternode(tree);
                if (rootInternode.IsValid()) {
                    for (int plantIndex = 0;
                         plantIndex < m_plantSystem.Get<PlantSystem>()->m_plants.size();
                         plantIndex++) {
                        if (m_plantSystem.Get<PlantSystem>()->m_plants[plantIndex] ==
                            tree) {
                            const auto plantGlobalTransform =
                                    tree.GetDataComponent<GlobalTransform>();
                            TreeNodeWalker(boundEntitiesLists[plantIndex],
                                           parentIndicesLists[plantIndex], -1, rootInternode);
                        }
                    }
                }
            },
            false);

    for (int plantIndex = 0;
         plantIndex < m_plantSystem.Get<PlantSystem>()->m_plants.size();
         plantIndex++) {
        const auto &plant = m_plantSystem.Get<PlantSystem>()->m_plants[plantIndex];
        if (plant.GetDataComponent<PlantInfo>().m_plantType !=
            PlantType::GeneralTree)
            continue;
        if (!plant.HasPrivateComponent<Animator>() ||
            !plant.HasPrivateComponent<SkinnedMeshRenderer>())
            continue;
        auto animator = plant.GetOrSetPrivateComponent<Animator>().lock();
        auto skinnedMeshRenderer =
                plant.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
        skinnedMeshRenderer->SetEnabled(true);
        if (plant.HasPrivateComponent<MeshRenderer>()) {
            plant.GetOrSetPrivateComponent<MeshRenderer>().lock()->SetEnabled(false);
        }
        auto treeData = plant.GetOrSetPrivateComponent<TreeData>().lock();
        if (Entity rootInternode = GetRootInternode(plant);
                !rootInternode.IsNull()) {
            const auto plantGlobalTransform =
                    plant.GetDataComponent<GlobalTransform>();
#pragma region Branch mesh
#pragma region Animator
            std::vector<glm::mat4> offsetMatrices;
            std::vector<std::string> names;
            offsetMatrices.resize(boundEntitiesLists[plantIndex].size());
            names.resize(boundEntitiesLists[plantIndex].size());
            boneIndicesLists[plantIndex].resize(
                    boundEntitiesLists[plantIndex].size());
            for (int i = 0; i < boundEntitiesLists[plantIndex].size(); i++) {
                names[i] = boundEntitiesLists[plantIndex][i].GetName();
                offsetMatrices[i] =
                        glm::inverse(glm::inverse(plantGlobalTransform.m_value) *
                                     boundEntitiesLists[plantIndex][i]
                                             .GetDataComponent<GlobalTransform>()
                                             .m_value);
                boneIndicesLists[plantIndex][i] = i;
            }
            animator->Setup(boundEntitiesLists[plantIndex], names, offsetMatrices);
#pragma endregion
            if (rootInternode.GetChildrenAmount() != 0) {
                std::vector<unsigned> skinnedIndices;
                std::vector<SkinnedVertex> skinnedVertices;
                TreeSkinnedMeshGenerator(boundEntitiesLists[plantIndex],
                                         parentIndicesLists[plantIndex],
                                         skinnedVertices, skinnedIndices);
                treeData->m_skinnedBranchMesh.Get<SkinnedMesh>()->SetVertices(
                        17, skinnedVertices, skinnedIndices);
                treeData->m_skinnedBranchMesh.Get<SkinnedMesh>()
                        ->m_boneAnimatorIndices = boneIndicesLists[plantIndex];
                skinnedMeshRenderer->m_skinnedMesh.Set<SkinnedMesh>(
                        treeData->m_skinnedBranchMesh.Get<SkinnedMesh>());
            }
#pragma endregion
            Entity leaves = GetLeaves(plant);
            if (leaves.IsValid())
                leaves.GetOrSetPrivateComponent<TreeLeaves>().lock()->FormSkinnedMesh(
                        boneIndicesLists[plantIndex]);
        }
    }
}

void TreeSystem::FormCandidates(std::vector<InternodeCandidate> &candidates) {
    const float globalTime = m_plantSystem.Get<PlantSystem>()->m_globalTime;
    std::mutex mutex;
    EntityManager::ForEach<GlobalTransform, Transform, InternodeInfo,
            InternodeGrowth, InternodeStatistics, Illumination>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
            [&, globalTime](int index, Entity internode,
                            GlobalTransform &globalTransform, Transform &transform,
                            InternodeInfo &internodeInfo,
                            InternodeGrowth &internodeGrowth,
                            InternodeStatistics &internodeStatistics,
                            Illumination &internodeIllumination) {
                if (internodeInfo.m_plantType != PlantType::GeneralTree)
                    return;
                auto internodeData =
                        internode.GetOrSetPrivateComponent<InternodeData>().lock();
                auto plant = internodeData->m_plant.Get();
                // auto thickestChild = internodeData->m_thickestChild.Get();

                auto treeData = plant.GetOrSetPrivateComponent<TreeData>().lock();
                auto plantInfo = plant.GetDataComponent<PlantInfo>();
                if (!plant.IsEnabled())
                    return;
#pragma region Go through each bud
                for (int i = 0; i < internodeData->m_buds.size(); i++) {
                    auto &bud = internodeData->m_buds[i];
                    if (!bud.m_active || !bud.m_enoughForGrowth)
                        continue;
                    bud.m_active = false;
                    bud.m_deathGlobalTime = globalTime;
                    const bool &isApical = bud.m_isApical;
#pragma region Form candidate
                    glm::quat prevGlobalRotation = globalTransform.GetRotation();
                    auto candidate = InternodeCandidate();
                    candidate.m_parent = internode;
                    candidate.m_info.m_startGlobalTime = globalTime;
                    candidate.m_plant = plant;
                    candidate.m_info.m_startAge = plantInfo.m_age;
                    candidate.m_info.m_order = internodeInfo.m_order + (isApical ? 0 : 1);
                    candidate.m_info.m_level = internodeInfo.m_level + (isApical ? 0 : 1);

                    candidate.m_growth.m_distanceToRoot =
                            internodeGrowth.m_distanceToRoot + 1;
                    candidate.m_growth.m_inhibitorTransmitFactor = GetGrowthParameter(
                            GrowthParameterType::InhibitorTransmitFactor, treeData,
                            internodeInfo, internodeGrowth, internodeStatistics);
                    glm::quat desiredRotation =
                            glm::radians(glm::vec3(0.0f, 0.0f, 0.0f)); // Apply main angle
                    glm::vec3 up = glm::vec3(0, 1, 0);
                    up = glm::rotate(up, glm::radians(bud.m_mainAngle),
                                     glm::vec3(0, 0, -1));
                    if (!bud.m_isApical) {
                        desiredRotation = glm::rotate(
                                desiredRotation,
                                GetGrowthParameter(GrowthParameterType::BranchingAngle,
                                                   treeData, internodeInfo, internodeGrowth,
                                                   internodeStatistics),
                                desiredRotation * up); // Apply branching angle
                    }
                    desiredRotation = glm::rotate(
                            desiredRotation,
                            GetGrowthParameter(GrowthParameterType::RollAngle, treeData,
                                               internodeInfo, internodeGrowth,
                                               internodeStatistics),
                            desiredRotation * glm::vec3(0, 0, -1)); // Apply roll angle
                    desiredRotation = glm::rotate(
                            desiredRotation,
                            GetGrowthParameter(GrowthParameterType::ApicalAngle, treeData,
                                               internodeInfo, internodeGrowth,
                                               internodeStatistics),
                            desiredRotation * glm::vec3(0, 1, 0)); // Apply apical angle
#pragma region Apply tropisms
                    glm::quat globalDesiredRotation =
                            prevGlobalRotation * desiredRotation;
                    glm::vec3 desiredFront =
                            globalDesiredRotation * glm::vec3(0.0f, 0.0f, -1.0f);
                    glm::vec3 desiredUp =
                            globalDesiredRotation * glm::vec3(0.0f, 1.0f, 0.0f);
                    m_plantSystem.Get<PlantSystem>()->ApplyTropism(
                            -treeData->m_gravityDirection,
                            GetGrowthParameter(GrowthParameterType::Gravitropism, treeData,
                                               internodeInfo, internodeGrowth,
                                               internodeStatistics),
                            desiredFront, desiredUp);
                    if (internodeIllumination.m_accumulatedDirection != glm::vec3(0.0f))
                        m_plantSystem.Get<PlantSystem>()->ApplyTropism(
                                glm::normalize(internodeIllumination.m_accumulatedDirection),
                                GetGrowthParameter(GrowthParameterType::Phototropism, treeData,
                                                   internodeInfo, internodeGrowth,
                                                   internodeStatistics),
                                desiredFront, desiredUp);
                    globalDesiredRotation = glm::quatLookAt(desiredFront, desiredUp);
                    desiredRotation =
                            glm::inverse(prevGlobalRotation) * globalDesiredRotation;
#pragma endregion

#pragma region Calculate transform
                    glm::quat globalRotation = globalTransform.GetRotation() *
                                               candidate.m_growth.m_desiredLocalRotation;
                    glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
                    glm::vec3 positionDelta =
                            front * treeData->m_parameters.m_internodeLengthBase;
                    glm::vec3 newInternodePosition =
                            globalTransform.GetPosition() + positionDelta;
                    candidate.m_globalTransform.m_value =
                            glm::translate(newInternodePosition) *
                            glm::mat4_cast(globalRotation) * glm::scale(glm::vec3(1.0f));
#pragma endregion

                    candidate.m_growth.m_desiredLocalRotation = desiredRotation;

                    candidate.m_statistics.m_isEndNode = true;
                    candidate.m_buds = std::vector<Bud>();

                    Bud apicalBud;
                    float totalResourceWeight = 0.0f;
                    apicalBud.m_isApical = true;
                    apicalBud.m_resourceWeight = GetGrowthParameter(
                            GrowthParameterType::ResourceWeightApical, treeData,
                            internodeInfo, internodeGrowth, internodeStatistics);
                    totalResourceWeight += apicalBud.m_resourceWeight;
                    apicalBud.m_avoidanceAngle = GetGrowthParameter(
                            GrowthParameterType::AvoidanceAngle, treeData, internodeInfo,
                            internodeGrowth, internodeStatistics);
                    candidate.m_buds.push_back(apicalBud);
                    const auto budAmount = GetGrowthParameter(
                            GrowthParameterType::LateralBudPerNode, treeData, internodeInfo,
                            internodeGrowth, internodeStatistics);
                    for (int budIndex = 0; budIndex < budAmount; budIndex++) {
                        Bud lateralBud;
                        lateralBud.m_isApical = false;
                        lateralBud.m_resourceWeight = GetGrowthParameter(
                                GrowthParameterType::ResourceWeightLateral, treeData,
                                internodeInfo, internodeGrowth, internodeStatistics);
                        totalResourceWeight += lateralBud.m_resourceWeight;
                        lateralBud.m_mainAngle =
                                360.0f * (glm::gaussRand(1.0f, 0.5f) + budIndex) / budAmount;
                        lateralBud.m_avoidanceAngle = GetGrowthParameter(
                                GrowthParameterType::AvoidanceAngle, treeData, internodeInfo,
                                internodeGrowth, internodeStatistics);
                        candidate.m_buds.push_back(lateralBud);
                    }
#pragma region Calculate resource weight for new buds and transport resource   \
    from current bud to new buds.
                    ResourceParcel &currentBudResourceParcel = bud.m_currentResource;
                    auto consumer = ResourceParcel(-1.0f, -1.0f);
                    consumer.m_globalTime = globalTime;
                    currentBudResourceParcel += consumer;
                    bud.m_resourceLog.push_back(consumer);
                    for (auto &newBud : candidate.m_buds) {
                        newBud.m_resourceWeight /= totalResourceWeight;
                        auto resourceParcel = ResourceParcel(
                                currentBudResourceParcel.m_nutrient * newBud.m_resourceWeight,
                                currentBudResourceParcel.m_carbon * newBud.m_resourceWeight);
                        resourceParcel.m_globalTime = globalTime;
                        newBud.m_currentResource += resourceParcel;
                        newBud.m_resourceLog.push_back(resourceParcel);
                    }
                    auto resourceParcel =
                            ResourceParcel(-currentBudResourceParcel.m_nutrient,
                                           -currentBudResourceParcel.m_carbon);
                    resourceParcel.m_globalTime = globalTime;
                    bud.m_currentResource += resourceParcel;
                    bud.m_resourceLog.push_back(resourceParcel);
#pragma endregion
                    std::lock_guard lock(mutex);
                    candidates.push_back(std::move(candidate));
#pragma endregion
                }
#pragma endregion
            },
            false);
}

float TreeSystem::GetGrowthParameter(const GrowthParameterType &type,
                                     const std::shared_ptr<TreeData> &treeData,
                                     InternodeInfo &internodeInfo,
                                     InternodeGrowth &internodeGrowth,
                                     InternodeStatistics &internodeStatistics) {
    float value = 0;
    switch (type) {
        case GrowthParameterType::InhibitorTransmitFactor:
            value = treeData->m_parameters.m_inhibitorDistanceFactor;
            break;
        case GrowthParameterType::Gravitropism:
            value = treeData->m_parameters.m_gravitropism;
            break;
        case GrowthParameterType::Phototropism:
            value = treeData->m_parameters.m_phototropism;
            break;
        case GrowthParameterType::BranchingAngle:
            value = glm::radians(
                    glm::gaussRand(treeData->m_parameters.m_branchingAngleMean,
                                   treeData->m_parameters.m_branchingAngleVariance));
            break;
        case GrowthParameterType::ApicalAngle:
            value = glm::radians(
                    glm::gaussRand(treeData->m_parameters.m_apicalAngleMean,
                                   treeData->m_parameters.m_apicalAngleVariance));
            break;
        case GrowthParameterType::RollAngle:
            value = glm::radians(
                    glm::gaussRand(treeData->m_parameters.m_rollAngleMean,
                                   treeData->m_parameters.m_rollAngleVariance));
            break;
        case GrowthParameterType::LateralBudPerNode:
            value = treeData->m_parameters.m_lateralBudPerNode;
            break;
        case GrowthParameterType::ResourceWeightApical:
            value = treeData->m_parameters.m_resourceWeightApical *
                    (1.0f + glm::gaussRand(
                            0.0f, treeData->m_parameters.m_resourceWeightVariance));
            break;
        case GrowthParameterType::ResourceWeightLateral:
            value = 1.0f + glm::gaussRand(
                    0.0f, treeData->m_parameters.m_resourceWeightVariance);
            break;
        case GrowthParameterType::AvoidanceAngle:
            value = treeData->m_parameters.m_avoidanceAngle;
            break;
    }
    return value;
}

void TreeSystem::PruneTrees(
        std::vector<std::pair<GlobalTransform, Volume *>> &obstacles) {

    m_voxelSpaceModule.Clear();
    EntityManager::ForEach<GlobalTransform>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
            [&](int index, Entity internode, GlobalTransform &globalTransform) {
                auto internodeData =
                        internode.GetOrSetPrivateComponent<InternodeData>().lock();
                auto plant = internodeData->m_plant.Get();
                // auto thickestChild = internodeData->m_thickestChild.Get();
                m_voxelSpaceModule.Push(globalTransform.GetPosition(), plant,
                                        internode);
            });

    std::vector<float> distanceLimits;
    std::vector<float> randomCutOffs;
    std::vector<float> randomCutOffAgeFactors;
    std::vector<float> randomCutOffMaxes;
    std::vector<float> avoidanceAngles;
    std::vector<float> internodeLengths;
    std::vector<RadialBoundingVolume *> rbvs;
    distanceLimits.resize(m_plantSystem.Get<PlantSystem>()->m_plants.size());
    randomCutOffs.resize(m_plantSystem.Get<PlantSystem>()->m_plants.size());
    randomCutOffAgeFactors.resize(
            m_plantSystem.Get<PlantSystem>()->m_plants.size());
    randomCutOffMaxes.resize(m_plantSystem.Get<PlantSystem>()->m_plants.size());
    avoidanceAngles.resize(m_plantSystem.Get<PlantSystem>()->m_plants.size());
    internodeLengths.resize(m_plantSystem.Get<PlantSystem>()->m_plants.size());
    rbvs.resize(m_plantSystem.Get<PlantSystem>()->m_plants.size());
    for (int i = 0; i < m_plantSystem.Get<PlantSystem>()->m_plants.size(); i++) {
        if (m_plantSystem.Get<PlantSystem>()
                ->m_plants[i]
                .HasPrivateComponent<TreeData>()) {
            auto treeData = m_plantSystem.Get<PlantSystem>()
                    ->m_plants[i]
                    .GetOrSetPrivateComponent<TreeData>()
                    .lock();
            distanceLimits[i] = 0;
            m_plantSystem.Get<PlantSystem>()->m_plants[i].ForEachChild(
                    [&](Entity child) {
                        if (child.HasDataComponent<InternodeInfo>())
                            distanceLimits[i] = child.GetDataComponent<InternodeStatistics>()
                                    .m_longestDistanceToAnyEndNode;
                    });
            distanceLimits[i] *= treeData->m_parameters.m_lowBranchCutOff;
            randomCutOffs[i] = treeData->m_parameters.m_randomCutOff;
            randomCutOffAgeFactors[i] =
                    treeData->m_parameters.m_randomCutOffAgeFactor;
            randomCutOffMaxes[i] = treeData->m_parameters.m_randomCutOffMax;
            avoidanceAngles[i] = treeData->m_parameters.m_avoidanceAngle;
            internodeLengths[i] = treeData->m_parameters.m_internodeLengthBase;
            rbvs[i] = GetRbv(m_plantSystem.Get<PlantSystem>()->m_plants[i])
                    .GetOrSetPrivateComponent<RadialBoundingVolume>()
                    .lock()
                    .get();
        }
    }
    std::vector<Entity> cutOff;
    std::mutex mutex;
    EntityManager::ForEach<GlobalTransform, InternodeInfo, Illumination,
            InternodeStatistics, InternodeGrowth>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
            [&](int index, Entity internode, GlobalTransform &globalTransform,
                InternodeInfo &internodeInfo, Illumination &illumination,
                InternodeStatistics &internodeStatistics,
                InternodeGrowth &internodeGrowth) {
                if (internodeInfo.m_plantType != PlantType::GeneralTree)
                    return;
                auto internodeData =
                        internode.GetOrSetPrivateComponent<InternodeData>().lock();
                auto plant = internodeData->m_plant.Get();
                // auto thickestChild = internodeData->m_thickestChild.Get();
                int targetIndex = 0;
                const auto position = globalTransform.GetPosition();
                for (auto &obstacle : obstacles) {
                    if (obstacle.second->InVolume(obstacle.first, position)) {
                        std::lock_guard lock(mutex);
                        cutOff.push_back(internode);
                        return;
                    }
                }
                for (int i = 0; i < m_plantSystem.Get<PlantSystem>()->m_plants.size();
                     i++) {
                    if (m_plantSystem.Get<PlantSystem>()->m_plants[i] == plant) {
                        targetIndex = i;
                        break;
                    }
                }
                if (internodeInfo.m_order != 1 &&
                    internodeGrowth.m_distanceToRoot < distanceLimits[targetIndex]) {
                    std::lock_guard lock(mutex);
                    cutOff.push_back(internode);
                    return;
                }

                if (!rbvs[targetIndex]->InVolume(position)) {
                    std::lock_guard lock(mutex);
                    cutOff.push_back(internode);
                    return;
                }
                // Below are pruning process which only for end nodes.
                if (!internodeStatistics.m_isEndNode)
                    return;
                const float angle = avoidanceAngles[targetIndex];

                /*
        if(m_voxelSpaceModule.HasNeighbor(globalTransform.GetPosition(),
        internode, internode.GetParent(), angle)){ std::lock_guard lock(mutex);
          cutOff.push_back(internode);
          return;
        }
        */

                const glm::vec3 direction =
                        glm::normalize(globalTransform.GetRotation() * glm::vec3(0, 0, -1));

                if (angle > 0 && m_voxelSpaceModule.HasObstacleConeSameOwner(
                        angle,
                        globalTransform.GetPosition() -
                        direction * internodeLengths[targetIndex],
                        direction, plant, internode, internode.GetParent(),
                        internodeLengths[targetIndex])) {
                    std::lock_guard lock(mutex);
                    cutOff.push_back(internode);
                    return;
                }
                /*
        if (m_voxelSpaceModule.HasNeighborFromDifferentOwner(
                globalTransform.GetPosition(), internodeInfo.m_plant,
                m_crownShynessDiameter)) {
          std::lock_guard lock(mutex);
          cutOff.push_back(internode);
          return;
        }
        */
                const float randomCutOffProb =
                        glm::min(m_plantSystem.Get<PlantSystem>()->m_deltaTime *
                                 randomCutOffMaxes[targetIndex],
                                 m_plantSystem.Get<PlantSystem>()->m_deltaTime *
                                 randomCutOffs[targetIndex] +
                                 (m_plantSystem.Get<PlantSystem>()->m_globalTime -
                                  internodeInfo.m_startGlobalTime) *
                                 randomCutOffAgeFactors[targetIndex]);
                if (glm::linearRand(0.0f, 1.0f) < randomCutOffProb) {
                    std::lock_guard lock(mutex);
                    cutOff.push_back(internode);
                    return;
                }
            },
            false);
    for (const auto &i : cutOff)
        EntityManager::DeleteEntity(i);
}

void TreeSystem::UpdateTreesMetaData() {
    EntityManager::ForEach<PlantInfo, GlobalTransform>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_plantQuery,
            [this](int i, Entity tree, PlantInfo &plantInfo,
                   GlobalTransform &globalTransform) {
                if (plantInfo.m_plantType != PlantType::GeneralTree)
                    return;
                const Entity rootInternode = GetRootInternode(tree);
                if (rootInternode.IsValid()) {
                    auto rootInternodeGrowth =
                            rootInternode.GetDataComponent<InternodeGrowth>();
                    rootInternodeGrowth.m_desiredGlobalPosition = glm::vec3(0.0f);
                    rootInternodeGrowth.m_desiredGlobalRotation =
                            globalTransform.GetRotation() *
                            rootInternodeGrowth.m_desiredLocalRotation;
                    rootInternode.SetDataComponent(rootInternodeGrowth);
                    auto treeData = tree.GetOrSetPrivateComponent<TreeData>().lock();
                    UpdateDistances(rootInternode, treeData);
                }
            },
            false);

    for (int plantIndex = 0;
         plantIndex < m_plantSystem.Get<PlantSystem>()->m_plants.size();
         plantIndex++) {
        const auto &plant = m_plantSystem.Get<PlantSystem>()->m_plants[plantIndex];
        if (plant.GetDataComponent<PlantInfo>().m_plantType !=
            PlantType::GeneralTree)
            continue;
        if (!plant.HasPrivateComponent<TreeData>())
            continue;
        if (Entity rootInternode = GetRootInternode(plant);
                !rootInternode.IsNull()) {
            auto treeData = plant.GetOrSetPrivateComponent<TreeData>().lock();
            UpdateLevels(rootInternode, treeData);
        }
    }
}

void TreeSystem::UpdateDistances(const Entity &internode,
                                 const std::shared_ptr<TreeData> &treeData) {
    Entity currentInternode = internode;
    auto currentInternodeInfo =
            currentInternode.GetDataComponent<InternodeInfo>();
    auto currentInternodeGrowth =
            currentInternode.GetDataComponent<InternodeGrowth>();
    auto currentInternodeStatistics =
            currentInternode.GetDataComponent<InternodeStatistics>();
    auto currentInternodeData =
            currentInternode.GetOrSetPrivateComponent<InternodeData>().lock();
#pragma region Single child chain from root to branch
    while (currentInternode.GetChildrenAmount() == 1) {
#pragma region Retrive child status
        Entity child = currentInternode.GetChildren()[0];
        auto childInternodeGrowth = child.GetDataComponent<InternodeGrowth>();
        auto childInternodeStatistics =
                child.GetDataComponent<InternodeStatistics>();
        auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
#pragma endregion
#pragma region Update child status
        childInternodeGrowth.m_inhibitor = 0;
        childInternodeStatistics.m_distanceToBranchStart =
                currentInternodeStatistics.m_distanceToBranchStart + 1;
        childInternodeGrowth.m_desiredGlobalRotation =
                currentInternodeGrowth.m_desiredGlobalRotation *
                childInternodeGrowth.m_desiredLocalRotation;
        childInternodeGrowth.m_desiredGlobalPosition =
                currentInternodeGrowth.m_desiredGlobalPosition +
                treeData->m_parameters.m_internodeLengthBase *
                (currentInternodeGrowth.m_desiredGlobalRotation *
                 glm::vec3(0, 0, -1));
#pragma endregion
#pragma region Apply child status
        child.SetDataComponent(childInternodeStatistics);
        child.SetDataComponent(childInternodeGrowth);
        child.SetDataComponent(childInternodeInfo);
#pragma endregion
#pragma region Retarget current internode
        currentInternode = child;
        currentInternodeInfo = childInternodeInfo;
        currentInternodeGrowth = childInternodeGrowth;
        currentInternodeStatistics = childInternodeStatistics;
        currentInternodeData =
                currentInternode.GetOrSetPrivateComponent<InternodeData>().lock();
#pragma endregion
    }
#pragma region Reset current status
    currentInternodeGrowth.m_inhibitor = 0;
    currentInternodeStatistics.m_totalLength = 0;
    currentInternodeStatistics.m_longestDistanceToAnyEndNode = 0;
    currentInternodeStatistics.m_maxChildOrder = 0;
    currentInternodeStatistics.m_isEndNode = false;
    currentInternodeStatistics.m_childrenEndNodeAmount = 0;
    currentInternodeGrowth.m_thickness = 0;
    currentInternodeGrowth.m_childrenTotalTorque = glm::vec3(0.0f);
    currentInternodeGrowth.m_MassOfChildren = 0.0f;
#pragma endregion
#pragma endregion
    if (currentInternode.GetChildrenAmount() != 0) {
        float maxThickness = 0;
        currentInternode.ForEachChild([&](Entity child) {
#pragma region From root to end
#pragma region Retrive child status
            auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
            auto childInternodeGrowth = child.GetDataComponent<InternodeGrowth>();
            auto childInternodeStatistics =
                    child.GetDataComponent<InternodeStatistics>();
#pragma endregion
#pragma region Update child status
            childInternodeStatistics.m_distanceToBranchStart =
                    (currentInternodeInfo.m_order == childInternodeInfo.m_order
                     ? currentInternodeStatistics.m_distanceToBranchStart
                     : 0) +
                    1;
            childInternodeGrowth.m_desiredGlobalRotation =
                    currentInternodeGrowth.m_desiredGlobalRotation *
                    childInternodeGrowth.m_desiredLocalRotation;
            childInternodeGrowth.m_desiredGlobalPosition =
                    currentInternodeGrowth.m_desiredGlobalPosition +
                    treeData->m_parameters.m_internodeLengthBase *
                    (currentInternodeGrowth.m_desiredGlobalRotation *
                     glm::vec3(0, 0, -1));

#pragma endregion
#pragma region Apply child status
            child.SetDataComponent(childInternodeStatistics);
            child.SetDataComponent(childInternodeGrowth);
            child.SetDataComponent(childInternodeInfo);
#pragma endregion
#pragma endregion
            UpdateDistances(child, treeData);
#pragma region From end to root
#pragma region Retrive child status
            childInternodeInfo = child.GetDataComponent<InternodeInfo>();
            childInternodeGrowth = child.GetDataComponent<InternodeGrowth>();
            childInternodeStatistics = child.GetDataComponent<InternodeStatistics>();
#pragma endregion
#pragma region Update self status
            auto childInternodeData =
                    child.GetOrSetPrivateComponent<InternodeData>().lock();
            for (const auto &bud : childInternodeData->m_buds)
                if (bud.m_active && bud.m_isApical)
                    currentInternodeGrowth.m_inhibitor +=
                            treeData->m_parameters.m_inhibitorBase;
            currentInternodeGrowth.m_inhibitor +=
                    childInternodeGrowth.m_inhibitor *
                    childInternodeGrowth.m_inhibitorTransmitFactor;
            currentInternodeStatistics.m_childrenEndNodeAmount +=
                    childInternodeStatistics.m_childrenEndNodeAmount;
            if (currentInternodeStatistics.m_maxChildOrder <
                childInternodeStatistics.m_maxChildOrder)
                currentInternodeStatistics.m_maxChildOrder =
                        childInternodeStatistics.m_maxChildOrder;
            currentInternodeStatistics.m_totalLength +=
                    childInternodeStatistics.m_totalLength + 1;
            const int tempDistanceToEndNode =
                    childInternodeStatistics.m_longestDistanceToAnyEndNode + 1;
            if (currentInternodeStatistics.m_longestDistanceToAnyEndNode <
                tempDistanceToEndNode)
                currentInternodeStatistics.m_longestDistanceToAnyEndNode =
                        tempDistanceToEndNode;
            currentInternodeGrowth.m_thickness +=
                    glm::pow(childInternodeGrowth.m_thickness, 2.0f);
            currentInternodeGrowth.m_MassOfChildren +=
                    childInternodeGrowth.m_MassOfChildren +
                    childInternodeGrowth.m_thickness;
            currentInternodeGrowth.m_childrenTotalTorque +=
                    childInternodeGrowth.m_childrenTotalTorque +
                    childInternodeGrowth.m_desiredGlobalPosition *
                    childInternodeGrowth.m_thickness;

            if (childInternodeGrowth.m_thickness > maxThickness) {
                maxThickness = childInternodeGrowth.m_thickness;
                currentInternodeData->m_thickestChild = child;
            }
#pragma endregion
#pragma endregion
        });
        currentInternodeGrowth.m_thickness =
                glm::pow(currentInternodeGrowth.m_thickness, 0.5f) *
                treeData->m_parameters.m_thicknessControlFactor;
        currentInternodeGrowth.m_childMeanPosition =
                currentInternodeGrowth.m_childrenTotalTorque /
                currentInternodeGrowth.m_MassOfChildren;
        const float strength =
                currentInternodeGrowth.m_MassOfChildren *
                glm::distance(
                        glm::vec2(currentInternodeGrowth.m_childMeanPosition.x,
                                  currentInternodeGrowth.m_childMeanPosition.z),
                        glm::vec2(currentInternodeGrowth.m_desiredGlobalPosition.x,
                                  currentInternodeGrowth.m_desiredGlobalPosition.z));
        currentInternodeGrowth.m_sagging = glm::min(
                treeData->m_parameters.m_gravityBendingMax,
                strength * treeData->m_parameters.m_gravityBendingFactor /
                glm::pow(currentInternodeGrowth.m_thickness /
                         treeData->m_parameters.m_endNodeThickness,
                         treeData->m_parameters.m_gravityBendingThicknessFactor));
    } else {
#pragma region Update self status(end node)
        currentInternodeStatistics.m_childrenEndNodeAmount = 1;
        currentInternodeStatistics.m_isEndNode = true;
        currentInternodeStatistics.m_maxChildOrder = currentInternodeInfo.m_order;
        currentInternodeStatistics.m_totalLength = 0;
        currentInternodeStatistics.m_longestDistanceToAnyEndNode = 0;
        currentInternodeGrowth.m_thickness =
                treeData->m_parameters.m_endNodeThickness;
        currentInternodeData->m_thickestChild = Entity();
#pragma endregion
    }
#pragma region From end to root
    currentInternodeStatistics.m_distanceToBranchEnd = 0;
    while (currentInternode != internode) {
#pragma region Apply current status
        currentInternode.SetDataComponent(currentInternodeInfo);
        currentInternode.SetDataComponent(currentInternodeGrowth);
        currentInternode.SetDataComponent(currentInternodeStatistics);
#pragma endregion
#pragma region Retarget to parent
        auto childInternodeData =
                currentInternode.GetOrSetPrivateComponent<InternodeData>().lock();
        Entity child = currentInternode;

        currentInternode = currentInternode.GetParent();
        auto childInternodeInfo = currentInternodeInfo;
        auto childInternodeGrowth = currentInternodeGrowth;
        auto childInternodeStatistics = currentInternodeStatistics;
        currentInternodeInfo = currentInternode.GetDataComponent<InternodeInfo>();
        currentInternodeGrowth =
                currentInternode.GetDataComponent<InternodeGrowth>();
        currentInternodeStatistics =
                currentInternode.GetDataComponent<InternodeStatistics>();
        currentInternodeData =
                currentInternode.GetOrSetPrivateComponent<InternodeData>().lock();
#pragma endregion
#pragma region Reset current status
        currentInternodeGrowth.m_inhibitor = 0;
        currentInternodeStatistics.m_totalLength = 0;
        currentInternodeStatistics.m_longestDistanceToAnyEndNode = 0;
        currentInternodeStatistics.m_maxChildOrder = 0;
        currentInternodeStatistics.m_isEndNode = false;
        currentInternodeStatistics.m_childrenEndNodeAmount = 0;
        currentInternodeGrowth.m_thickness = 0;
        currentInternodeGrowth.m_childrenTotalTorque = glm::vec3(0.0f);
        currentInternodeGrowth.m_MassOfChildren = 0.0f;
#pragma endregion
#pragma region Update self status
        for (const auto &bud : childInternodeData->m_buds)
            if (bud.m_active && bud.m_isApical)
                currentInternodeGrowth.m_inhibitor +=
                        treeData->m_parameters.m_inhibitorBase;
        currentInternodeGrowth.m_inhibitor +=
                childInternodeGrowth.m_inhibitor *
                childInternodeGrowth.m_inhibitorTransmitFactor;
        currentInternodeStatistics.m_childrenEndNodeAmount =
                childInternodeStatistics.m_childrenEndNodeAmount;
        currentInternodeGrowth.m_thickness = childInternodeGrowth.m_thickness;
        currentInternodeStatistics.m_maxChildOrder =
                childInternodeStatistics.m_maxChildOrder;
        currentInternodeStatistics.m_isEndNode = false;
        currentInternodeStatistics.m_distanceToBranchEnd =
                childInternodeStatistics.m_distanceToBranchEnd + 1;
        currentInternodeStatistics.m_totalLength =
                childInternodeStatistics.m_totalLength + 1;
        currentInternodeStatistics.m_longestDistanceToAnyEndNode =
                childInternodeStatistics.m_longestDistanceToAnyEndNode + 1;

        currentInternodeGrowth.m_MassOfChildren +=
                childInternodeGrowth.m_MassOfChildren +
                childInternodeGrowth.m_thickness;
        currentInternodeGrowth.m_childrenTotalTorque +=
                childInternodeGrowth.m_childrenTotalTorque +
                childInternodeGrowth.m_desiredGlobalPosition *
                childInternodeGrowth.m_thickness;
        currentInternodeGrowth.m_childMeanPosition =
                currentInternodeGrowth.m_childrenTotalTorque /
                currentInternodeGrowth.m_MassOfChildren;
        const float strength =
                currentInternodeGrowth.m_MassOfChildren *
                glm::distance(
                        glm::vec2(currentInternodeGrowth.m_childMeanPosition.x,
                                  currentInternodeGrowth.m_childMeanPosition.z),
                        glm::vec2(currentInternodeGrowth.m_desiredGlobalPosition.x,
                                  currentInternodeGrowth.m_desiredGlobalPosition.z));
        currentInternodeGrowth.m_sagging =
                strength * treeData->m_parameters.m_gravityBendingFactor /
                glm::pow(currentInternodeGrowth.m_thickness /
                         treeData->m_parameters.m_endNodeThickness,
                         treeData->m_parameters.m_gravityBendingThicknessFactor);
        currentInternodeData->m_thickestChild = child;
#pragma endregion
    }
#pragma endregion
#pragma region Apply self status
    currentInternode.SetDataComponent(currentInternodeInfo);
    currentInternode.SetDataComponent(currentInternodeGrowth);
    currentInternode.SetDataComponent(currentInternodeStatistics);
#pragma endregion
}

void TreeSystem::UpdateLevels(const Entity &internode,
                              const std::shared_ptr<TreeData> &treeData) {
    auto currentInternode = internode;
    auto currentInternodeInfo = internode.GetDataComponent<InternodeInfo>();
    auto currentInternodeGlobalTransform =
            internode.GetDataComponent<GlobalTransform>();
#pragma region Single child chain from root to branch
    while (currentInternode.GetChildrenAmount() == 1) {
#pragma region Retrive child status
        Entity child = currentInternode.GetChildren()[0];
        auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
        auto childInternodeGrowth = child.GetDataComponent<InternodeGrowth>();
#pragma endregion
#pragma region Update child status
        childInternodeInfo.m_level = currentInternodeInfo.m_level;
#pragma region Gravity bending.
        GlobalTransform childInternodeGlobalTransform;
        glm::quat globalRotation = currentInternodeGlobalTransform.GetRotation() *
                                   childInternodeGrowth.m_desiredLocalRotation;
        glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
        glm::vec3 up = globalRotation * glm::vec3(0, 1, 0);
        m_plantSystem.Get<PlantSystem>()->ApplyTropism(
                glm::vec3(0, -1, 0), childInternodeGrowth.m_sagging, front, up);
        globalRotation = glm::quatLookAt(front, up);
        const glm::vec3 globalPosition =
                currentInternodeGlobalTransform.GetPosition() +
                front * treeData->m_parameters.m_internodeLengthBase;
        childInternodeGlobalTransform.SetValue(globalPosition, globalRotation,
                                               glm::vec3(1.0f));
#pragma endregion
#pragma endregion
#pragma region Apply child status
        child.SetDataComponent(childInternodeInfo);
        child.SetDataComponent(childInternodeGlobalTransform);
        child.SetDataComponent(childInternodeGrowth);
        auto rigidBody = child.GetOrSetPrivateComponent<RigidBody>().lock();
        rigidBody->SetDensityAndMassCenter(m_density *
                                           childInternodeGrowth.m_thickness *
                                           childInternodeGrowth.m_thickness);
        rigidBody->SetLinearDamping(m_linearDamping);
        rigidBody->SetAngularDamping(m_angularDamping);
        rigidBody->SetSolverIterations(m_positionSolverIteration,
                                       m_velocitySolverIteration);
        auto joint = child.GetOrSetPrivateComponent<Joint>().lock();
        joint->Link(currentInternode);
        joint->SetType(JointType::D6);
        joint->SetMotion(MotionAxis::SwingY, MotionType::Free);
        joint->SetMotion(MotionAxis::SwingZ, MotionType::Free);
        joint->SetDrive(DriveType::Swing,
                        glm::pow(childInternodeGrowth.m_thickness /
                                 treeData->m_parameters.m_endNodeThickness,
                                 m_jointDriveStiffnessThicknessFactor) *
                        m_jointDriveStiffnessFactor,
                        glm::pow(childInternodeGrowth.m_thickness /
                                 treeData->m_parameters.m_endNodeThickness,
                                 m_jointDriveDampingThicknessFactor) *
                        m_jointDriveDampingFactor,
                        m_enableAccelerationForDrive);
#pragma endregion
#pragma region Retarget current internode
        currentInternode = child;
        currentInternodeInfo = childInternodeInfo;
        currentInternodeGlobalTransform = childInternodeGlobalTransform;
#pragma endregion
    }
    auto currentInternodeStatistics =
            currentInternode.GetDataComponent<InternodeStatistics>();
#pragma endregion
    if (currentInternode.GetChildrenAmount() != 0) {
#pragma region Select max child
        float maxChildLength = 0;
        int minChildOrder = 9999;
        Entity maxChild = Entity();
        currentInternode.ForEachChild([&](Entity child) {
            const auto childInternodeStatistics =
                    child.GetDataComponent<InternodeStatistics>();
            const auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
            if (maxChildLength <= childInternodeStatistics.m_totalLength &&
                childInternodeInfo.m_order < minChildOrder) {
                minChildOrder = childInternodeInfo.m_order;
                maxChildLength = childInternodeStatistics.m_totalLength;
                maxChild = child;
            }
        });
#pragma endregion
#pragma region Apply level
        currentInternode.ForEachChild([&](Entity child) {
            auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
            auto childInternodeStatistics =
                    child.GetDataComponent<InternodeStatistics>();
            auto childInternodeGrowth = child.GetDataComponent<InternodeGrowth>();

            if (child == maxChild) {
                childInternodeStatistics.m_isMaxChild = true;
                childInternodeInfo.m_level = currentInternodeInfo.m_level;
            } else {
                childInternodeStatistics.m_isMaxChild = false;
                childInternodeInfo.m_level = currentInternodeInfo.m_level + 1;
            }
#pragma region Gravity bending.
            GlobalTransform childInternodeGlobalTransform;
            glm::quat globalRotation = currentInternodeGlobalTransform.GetRotation() *
                                       childInternodeGrowth.m_desiredLocalRotation;
            glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
            glm::vec3 up = globalRotation * glm::vec3(0, 1, 0);
            m_plantSystem.Get<PlantSystem>()->ApplyTropism(
                    childInternodeGrowth.m_desiredGlobalPosition -
                    childInternodeGrowth.m_childMeanPosition,
                    childInternodeGrowth.m_sagging, front, up);
            globalRotation = glm::quatLookAt(front, up);
            const glm::vec3 globalPosition =
                    currentInternodeGlobalTransform.GetPosition() +
                    front * treeData->m_parameters.m_internodeLengthBase;
            childInternodeGlobalTransform.SetValue(globalPosition, globalRotation,
                                                   glm::vec3(1.0f));
#pragma endregion
#pragma region Apply child status
            child.SetDataComponent(childInternodeGlobalTransform);
            child.SetDataComponent(childInternodeStatistics);
            child.SetDataComponent(childInternodeInfo);
            child.SetDataComponent(childInternodeGrowth);
            auto rigidBody = child.GetOrSetPrivateComponent<RigidBody>().lock();
            rigidBody->SetDensityAndMassCenter(m_density *
                                               childInternodeGrowth.m_thickness *
                                               childInternodeGrowth.m_thickness);
            rigidBody->SetLinearDamping(m_linearDamping);
            rigidBody->SetAngularDamping(m_angularDamping);
            rigidBody->SetSolverIterations(m_positionSolverIteration,
                                           m_velocitySolverIteration);
            auto joint = child.GetOrSetPrivateComponent<Joint>().lock();
            joint->Link(currentInternode);
            joint->SetType(JointType::D6);
            joint->SetMotion(MotionAxis::SwingY, MotionType::Free);
            joint->SetMotion(MotionAxis::SwingZ, MotionType::Free);
            joint->SetDrive(DriveType::Swing,
                            glm::pow(childInternodeGrowth.m_thickness /
                                     treeData->m_parameters.m_endNodeThickness,
                                     m_jointDriveStiffnessThicknessFactor) *
                            m_jointDriveStiffnessFactor,
                            glm::pow(childInternodeGrowth.m_thickness /
                                     treeData->m_parameters.m_endNodeThickness,
                                     m_jointDriveDampingThicknessFactor) *
                            m_jointDriveDampingFactor,
                            m_enableAccelerationForDrive);

#pragma endregion
            UpdateLevels(child, treeData);
            childInternodeStatistics = child.GetDataComponent<InternodeStatistics>();
            if (childInternodeStatistics.m_maxChildLevel >
                currentInternodeStatistics.m_maxChildLevel)
                currentInternodeStatistics.m_maxChildLevel =
                        childInternodeStatistics.m_maxChildLevel;
        });
#pragma endregion
    } else {
        currentInternodeStatistics.m_maxChildLevel = currentInternodeInfo.m_level;
    }
    while (currentInternode != internode) {
#pragma region Apply current status
        currentInternode.SetDataComponent(currentInternodeStatistics);
#pragma endregion
#pragma region Retarget to parent
        currentInternode = currentInternode.GetParent();
        const auto childInternodeStatistics = currentInternodeStatistics;
        currentInternodeStatistics =
                currentInternode.GetDataComponent<InternodeStatistics>();
#pragma endregion
#pragma region Update self status
        currentInternodeStatistics.m_maxChildLevel =
                childInternodeStatistics.m_maxChildLevel;
#pragma endregion
    }
#pragma region Apply self status
    currentInternode.SetDataComponent(currentInternodeStatistics);
#pragma endregion
}

void TreeSystem::ResetTimeForTree(const float &value) {
    if (value < 0 || value >= m_plantSystem.Get<PlantSystem>()->m_globalTime)
        return;
    m_plantSystem.Get<PlantSystem>()->m_globalTime = value;
    std::vector<Entity> trees;
    m_plantSystem.Get<PlantSystem>()->m_plantQuery.ToEntityArray(trees);
    for (const auto &tree : trees) {
        auto plantInfo = tree.GetDataComponent<PlantInfo>();
        if (plantInfo.m_startTime > value) {
            EntityManager::DeleteEntity(tree);
            continue;
        }
        plantInfo.m_age = value - plantInfo.m_startTime;
        tree.SetDataComponent(plantInfo);
        Entity rootInternode = GetRootInternode(tree);
        if (rootInternode.IsValid()) {
            ResetTimeForTree(rootInternode, value);
        }
    }
    EntityManager::ForEach<InternodeInfo>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
            [value](int i, Entity internode, InternodeInfo &internodeInfo) {
                auto childInternodeData =
                        internode.GetOrSetPrivateComponent<InternodeData>().lock();
                for (auto &bud : childInternodeData->m_buds) {
                    if (!bud.m_active && bud.m_deathGlobalTime > value) {
                        bud.m_active = true;
                        bud.m_deathGlobalTime = -1;
                    }
                    bud.m_enoughForGrowth = false;
                    bud.m_currentResource = ResourceParcel();
                    for (auto it = bud.m_resourceLog.begin();
                         it != bud.m_resourceLog.end(); ++it) {
                        if (it->m_globalTime > value) {
                            bud.m_resourceLog.erase(it, bud.m_resourceLog.end());
                            break;
                        }
                    }
                    for (const auto &parcel : bud.m_resourceLog)
                        bud.m_currentResource += parcel;
                    if (bud.m_currentResource.IsEnough())
                        bud.m_enoughForGrowth = true;
                }
            },
            false);
    UpdateTreesMetaData();
}

void TreeSystem::ResetTimeForTree(const Entity &internode,
                                  const float &globalTime) {
    Entity currentInternode = internode;
    while (currentInternode.GetChildrenAmount() == 1) {
        Entity child = currentInternode.GetChildren()[0];
        const auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
        if (childInternodeInfo.m_startGlobalTime > globalTime) {
            EntityManager::DeleteEntity(child);
            return;
        }
        currentInternode = child;
    }

    if (currentInternode.GetChildrenAmount() != 0) {
        std::vector<Entity> childrenToDelete;
        currentInternode.ForEachChild([this, globalTime,
                                              &childrenToDelete](Entity child) {
            const auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
            if (childInternodeInfo.m_startGlobalTime > globalTime) {
                childrenToDelete.push_back(child);
                return;
            }
            ResetTimeForTree(child, globalTime);
        });
        for (const auto &child : childrenToDelete)
            EntityManager::DeleteEntity(child);
    }
}

void TreeSystem::DistributeResourcesForTree(
        std::vector<ResourceParcel> &totalNutrients) {
    auto &plants = m_plantSystem.Get<PlantSystem>()->m_plants;
    std::vector<float> divisors;
    std::vector<float> apicalControlLevelFactors;
    std::vector<float> resourceAllocationDistFactors;
    std::vector<float> apicalIlluminationRequirements;
    std::vector<float> lateralIlluminationRequirements;
    std::vector<float> requirementMaximums;
    std::vector<glm::vec3> treePositions;
    std::vector<float> heightResourceBase;
    std::vector<float> heightResourceFactor;
    std::vector<float> heightResourceFactorMin;
    divisors.resize(plants.size());
    apicalControlLevelFactors.resize(plants.size());
    resourceAllocationDistFactors.resize(plants.size());
    apicalIlluminationRequirements.resize(plants.size());
    lateralIlluminationRequirements.resize(plants.size());
    requirementMaximums.resize(plants.size());
    treePositions.resize(plants.size());
    heightResourceBase.resize(plants.size());
    heightResourceFactor.resize(plants.size());
    heightResourceFactorMin.resize(plants.size());
    for (int i = 0; i < plants.size(); i++) {
        if (plants[i].HasPrivateComponent<TreeData>()) {
            auto treeData = plants[i].GetOrSetPrivateComponent<TreeData>().lock();
            divisors[i] = 0;
            apicalControlLevelFactors[i] =
                    treeData->m_parameters.m_apicalControlLevelFactor;
            apicalIlluminationRequirements[i] =
                    treeData->m_parameters.m_apicalIlluminationRequirement;
            lateralIlluminationRequirements[i] =
                    treeData->m_parameters.m_lateralIlluminationRequirement;
            requirementMaximums[i] = 0;
            treePositions[i] =
                    plants[i].GetDataComponent<GlobalTransform>().GetPosition();
            heightResourceBase[i] =
                    treeData->m_parameters.m_heightResourceHeightDecreaseBase;
            heightResourceFactor[i] =
                    treeData->m_parameters.m_heightResourceHeightDecreaseFactor;
            heightResourceFactorMin[i] =
                    treeData->m_parameters.m_heightResourceHeightDecreaseMin;
        }
    }
    std::mutex maximumLock;
    EntityManager::ForEach<GlobalTransform, InternodeInfo, InternodeGrowth,
            InternodeStatistics>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
            [&](int index, Entity internode, GlobalTransform &globalTransform,
                InternodeInfo &internodeInfo, InternodeGrowth &internodeGrowth,
                InternodeStatistics &internodeStatistics) {
                if (internodeInfo.m_plantType != PlantType::GeneralTree)
                    return;
                auto internodeData =
                        internode.GetOrSetPrivateComponent<InternodeData>().lock();
                auto plant = internodeData->m_plant.Get();
                // auto thickestChild = internodeData->m_thickestChild.Get();
                for (int i = 0; i < plants.size(); i++) {
                    if (plants[i] == plant) {
                        const float internodeRequirement =
                                glm::max(heightResourceFactorMin[i],
                                         1.0f - glm::pow(globalTransform.GetPosition().y -
                                                         treePositions[i].y,
                                                         heightResourceFactor[i]) *
                                                heightResourceBase[i]) *
                                glm::max(0.0f, 1.0f - internodeGrowth.m_inhibitor) *
                                glm::pow(apicalControlLevelFactors[i],
                                         static_cast<float>(internodeInfo.m_level));
                        auto internodeData =
                                internode.GetOrSetPrivateComponent<InternodeData>().lock();
                        float budsRequirement = 0;
                        for (const auto &bud : internodeData->m_buds) {
                            if (bud.m_active && !bud.m_enoughForGrowth) {
                                budsRequirement += bud.m_resourceWeight;
                                std::lock_guard<std::mutex> lock(maximumLock);
                                const float budRequirement =
                                        internodeRequirement * bud.m_resourceWeight;
                                if (budRequirement > requirementMaximums[i]) {
                                    requirementMaximums[i] = budRequirement;
                                }
                            }
                        }
                        divisors[i] += budsRequirement * internodeRequirement;
                        break;
                    }
                }
            },
            false);
    const auto globalTime = m_plantSystem.Get<PlantSystem>()->m_globalTime;
    EntityManager::ForEach<GlobalTransform, Illumination, InternodeInfo,
            InternodeGrowth, InternodeStatistics>(
            JobManager::PrimaryWorkers(),
            m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
            [&](int index, Entity internode, GlobalTransform &globalTransform,
                Illumination &illumination, InternodeInfo &internodeInfo,
                InternodeGrowth &internodeGrowth,
                InternodeStatistics &internodeStatistics) {
                if (internodeInfo.m_plantType != PlantType::GeneralTree)
                    return;
                auto internodeData =
                        internode.GetOrSetPrivateComponent<InternodeData>().lock();
                auto plant = internodeData->m_plant.Get();
                // auto thickestChild = internodeData->m_thickestChild.Get();
                for (int i = 0; i < plants.size(); i++) {
                    if (plants[i] == plant) {
                        auto internodeData =
                                internode.GetOrSetPrivateComponent<InternodeData>().lock();
                        float budsRequirement = 0;
                        for (const auto &bud : internodeData->m_buds) {
                            if (bud.m_active && !bud.m_enoughForGrowth)
                                budsRequirement += bud.m_resourceWeight;
                        }
                        const float internodeRequirement =
                                glm::max(heightResourceFactorMin[i],
                                         1.0f - glm::pow(globalTransform.GetPosition().y -
                                                         treePositions[i].y,
                                                         heightResourceFactor[i]) *
                                                heightResourceBase[i]) *
                                glm::max(0.0f, 1.0f - internodeGrowth.m_inhibitor) *
                                glm::pow(apicalControlLevelFactors[i],
                                         static_cast<float>(internodeInfo.m_level));
                        const float internodeNutrient =
                                glm::min(totalNutrients[i].m_nutrient /
                                         (divisors[i] / requirementMaximums[i]),
                                         m_plantSystem.Get<PlantSystem>()->m_deltaTime) *
                                budsRequirement * internodeRequirement / requirementMaximums[i];
                        const float internodeCarbon =
                                illumination.m_currentIntensity *
                                m_plantSystem.Get<PlantSystem>()->m_deltaTime;
                        for (auto &bud : internodeData->m_buds) {
                            if (bud.m_active && !bud.m_enoughForGrowth) {
                                ResourceParcel resourceParcel = ResourceParcel(
                                        internodeNutrient / budsRequirement * bud.m_resourceWeight,
                                        internodeCarbon /
                                        (bud.m_isApical ? apicalIlluminationRequirements[i]
                                                        : lateralIlluminationRequirements[i]));
                                resourceParcel.m_globalTime = globalTime;
                                bud.m_currentResource += resourceParcel;
                                bud.m_resourceLog.push_back(resourceParcel);
                                if (bud.m_currentResource.IsEnough())
                                    bud.m_enoughForGrowth = true;
                            }
                        }
                        break;
                    }
                }
            },
            false);
}

void TreeSystem::Start() {
    if (!m_plantSystem.Get<PlantSystem>())
        m_plantSystem = EntityManager::GetSystem<PlantSystem>();
    m_voxelSpaceModule.Reset();

    m_colorMapSegmentAmount = 3;
    m_colorMapValues.resize(m_colorMapSegmentAmount);
    m_colorMapColors.resize(m_colorMapSegmentAmount);
    for (int i = 0; i < m_colorMapSegmentAmount; i++) {
        m_colorMapValues[i] = i;
        m_colorMapColors[i] = glm::linearRand(glm::vec3(0.0f), glm::vec3(1.0f));
    }

    m_leavesArchetype =
            EntityManager::CreateEntityArchetype("Tree Leaves", TreeLeavesTag());
    m_rbvArchetype = EntityManager::CreateEntityArchetype("RBV", RbvTag());
#pragma region Materials
    for (int i = 0; i < 64; i++) {
        m_randomColors.emplace_back(glm::linearRand(0.0f, 1.0f),
                                    glm::linearRand(0.0f, 1.0f),
                                    glm::linearRand(0.0f, 1.0f));
    }
    if (!m_defaultRayTracingBranchAlbedoTexture.Get<Texture2D>())
        m_defaultRayTracingBranchAlbedoTexture = AssetManager::Import<Texture2D>(
                std::filesystem::path(PLANT_ARCHITECT_RESOURCE_FOLDER) /
                "Textures/BarkMaterial/Bark_Pine_baseColor.jpg");
    if (!m_defaultRayTracingBranchNormalTexture.Get<Texture2D>())
        m_defaultRayTracingBranchNormalTexture = AssetManager::Import<Texture2D>(
                std::filesystem::path(PLANT_ARCHITECT_RESOURCE_FOLDER) /
                "Textures/BarkMaterial/Bark_Pine_normal.jpg");
    if (!m_defaultBranchAlbedoTexture.Get<Texture2D>())
        m_defaultBranchAlbedoTexture = AssetManager::Import<Texture2D>(
                std::filesystem::path(PLANT_ARCHITECT_RESOURCE_FOLDER) /
                "Textures/BarkMaterial/Bark_Pine_baseColor.jpg");
    if (!m_defaultBranchNormalTexture.Get<Texture2D>())
        m_defaultBranchNormalTexture = AssetManager::Import<Texture2D>(
                std::filesystem::path(PLANT_ARCHITECT_RESOURCE_FOLDER) /
                "Textures/BarkMaterial/Bark_Pine_normal.jpg");
#pragma endregion
#pragma region General tree growth
    m_plantSystem.Get<PlantSystem>()->m_plantMeshGenerators.insert_or_assign(
            PlantType::GeneralTree, [&]() { GenerateMeshForTree(); });

    m_plantSystem.Get<PlantSystem>()
            ->m_plantSkinnedMeshGenerators.insert_or_assign(
                    PlantType::GeneralTree, [&]() { GenerateSkinnedMeshForTree(); });

    m_plantSystem.Get<PlantSystem>()->m_plantResourceAllocators.insert_or_assign(
            PlantType::GeneralTree, [&](std::vector<ResourceParcel> &resources) {
                DistributeResourcesForTree(resources);
            });

    m_plantSystem.Get<PlantSystem>()->m_plantGrowthModels.insert_or_assign(
            PlantType::GeneralTree, [&](std::vector<InternodeCandidate> &candidates) {
                FormCandidates(candidates);
            });

    m_plantSystem.Get<PlantSystem>()->m_plantInternodePruners.insert_or_assign(
            PlantType::GeneralTree,
            [&](std::vector<std::pair<GlobalTransform, Volume *>> &obstacles) {
                PruneTrees(obstacles);
            });

    m_plantSystem.Get<PlantSystem>()
            ->m_plantInternodePostProcessors.insert_or_assign(
                    PlantType::GeneralTree,
                    [&](const Entity &newInternode, const InternodeCandidate &candidate) {
                        InternodePostProcessor(newInternode, candidate);
                    });

    m_plantSystem.Get<PlantSystem>()->m_plantMetaDataCalculators.insert_or_assign(
            PlantType::GeneralTree, [&]() { UpdateTreesMetaData(); });

    m_plantSystem.Get<PlantSystem>()->m_deleteAllPlants.insert_or_assign(
            PlantType::GeneralTree, [this]() { DeleteAllPlantsHelper(); });
#pragma endregion

    m_ready = true;
}

void TreeSystem::SerializeScene(const std::string &filename) {
    std::ofstream ofs;
    ofs.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (!ofs.is_open()) {
        Debug::Error("Can't open file!");
        return;
    }
    rapidxml::xml_document<> doc;
    auto *type = doc.allocate_node(rapidxml::node_doctype, 0, "Scene");
    doc.append_node(type);
    auto *scene = doc.allocate_node(rapidxml::node_element, "Scene", "Tree");
    doc.append_node(scene);
    std::vector<Entity> trees;
    m_plantSystem.Get<PlantSystem>()->m_plantQuery.ToEntityArray(trees);
    for (const auto &plant : trees) {
        Serialize(plant, doc, scene);
    }
    ofs << doc;
    ofs.flush();
    ofs.close();
}

void TreeSystem::Serialize(const Entity &treeEntity,
                           rapidxml::xml_document<> &doc,
                           rapidxml::xml_node<> *sceneNode) {
    if (treeEntity.GetDataComponent<PlantInfo>().m_plantType !=
        PlantType::GeneralTree)
        return;
    auto *tree = doc.allocate_node(rapidxml::node_element, "Tree", "Textures");
    sceneNode->append_node(tree);

    auto *textures =
            doc.allocate_node(rapidxml::node_element, "Textures", "Texture");
    tree->append_node(textures);
    auto *barkTex = doc.allocate_node(rapidxml::node_element, "Texture", "");
    barkTex->append_attribute(doc.allocate_attribute("name", "Bark"));
    barkTex->append_attribute(
            doc.allocate_attribute("path", "Data/Textures/Bark/UlmusLaevis.jpg"));
    auto *leafTex = doc.allocate_node(rapidxml::node_element, "Texture", "");
    leafTex->append_attribute(doc.allocate_attribute("name", "Leaf"));
    leafTex->append_attribute(
            doc.allocate_attribute("path", "Data/Textures/Leaf/UlmusLaevis"));

    textures->append_node(leafTex);
    textures->append_node(barkTex);

    auto *nodes = doc.allocate_node(rapidxml::node_element, "Nodes", "Node");
    tree->append_node(nodes);

    // std::vector<InternodeInfo> internodeInfos;
    std::vector<Entity> internodes;
    auto plantSystem = EntityManager::GetSystem<PlantSystem>();
    plantSystem->m_internodeQuery.ToEntityArray<InternodeInfo>(
            internodes, [treeEntity](const Entity &internode,
                                     const InternodeInfo &internodeInfo) {
                auto internodeData =
                        internode.GetOrSetPrivateComponent<InternodeData>().lock();
                auto plant = internodeData->m_plant.Get();
                // auto thickestChild = internodeData->m_thickestChild.Get();
                return treeEntity == plant;
            });

    Entity rootInternode;
    unsigned rootNodeIndex = 0;
    treeEntity.ForEachChild([&rootNodeIndex, &rootInternode](Entity child) {
        if (child.HasDataComponent<InternodeInfo>()) {
            rootNodeIndex = child.GetIndex() - 1;
            rootInternode = child;
        }
    });
    rootNodeIndex = 0;
    for (const auto &i : internodes) {
        auto internodeGrowth = i.GetDataComponent<InternodeGrowth>();
        auto internodeInfo = i.GetDataComponent<InternodeInfo>();
        auto internodeStatistics = i.GetDataComponent<InternodeStatistics>();
        auto *node = doc.allocate_node(rapidxml::node_element, "Node", "Position");
        node->append_attribute(doc.allocate_attribute(
                "id", doc.allocate_string(
                        std::to_string(i.GetIndex() - rootNodeIndex).c_str())));
        node->append_attribute(doc.allocate_attribute(
                "additional", doc.allocate_string(std::to_string(0).c_str())));
        nodes->append_node(node);
        auto globalTransform = i.GetDataComponent<GlobalTransform>().m_value;
        auto *position = doc.allocate_node(rapidxml::node_element, "Position");
        position->append_attribute(doc.allocate_attribute(
                "x",
                doc.allocate_string(std::to_string(globalTransform[3].x).c_str())));
        position->append_attribute(doc.allocate_attribute(
                "y",
                doc.allocate_string(std::to_string(globalTransform[3].y).c_str())));
        position->append_attribute(doc.allocate_attribute(
                "z",
                doc.allocate_string(std::to_string(globalTransform[3].z).c_str())));
        node->append_node(position);

        auto *distRoot = doc.allocate_node(rapidxml::node_element, "DistToRoot");
        distRoot->append_attribute(doc.allocate_attribute(
                "value",
                doc.allocate_string(
                        std::to_string(internodeGrowth.m_distanceToRoot).c_str())));
        node->append_node(distRoot);

        auto *thickness = doc.allocate_node(rapidxml::node_element, "Thickness");
        thickness->append_attribute(doc.allocate_attribute(
                "value", doc.allocate_string(
                        std::to_string(internodeGrowth.m_thickness).c_str())));
        node->append_node(thickness);

        auto *maxLength =
                doc.allocate_node(rapidxml::node_element, "MaxBranchLength");
        maxLength->append_attribute(doc.allocate_attribute(
                "value",
                doc.allocate_string(
                        std::to_string(internodeStatistics.m_longestDistanceToAnyEndNode)
                                .c_str())));
        node->append_node(maxLength);

        unsigned parentIndex = i.GetParent().GetIndex();
        float thicknessVal = 0;
        if (parentIndex != treeEntity.GetIndex()) {
            thicknessVal =
                    i.GetParent().GetDataComponent<InternodeGrowth>().m_thickness;
        } else {
            auto *root = doc.allocate_node(rapidxml::node_element, "Root");
            root->append_attribute(doc.allocate_attribute("value", "true"));
            node->append_node(root);
        }
        auto *parent = doc.allocate_node(rapidxml::node_element, "Parent");
        parent->append_attribute(doc.allocate_attribute(
                "id", doc.allocate_string(
                        std::to_string(parentIndex - rootNodeIndex).c_str())));
        parent->append_attribute(doc.allocate_attribute(
                "thickness",
                doc.allocate_string(std::to_string(thicknessVal).c_str())));
        node->append_node(parent);
        if (i.GetChildrenAmount() != 0) {
            auto *children =
                    doc.allocate_node(rapidxml::node_element, "Children", "Child");
            node->append_node(children);

            i.ForEachChild([children, &doc, rootNodeIndex](Entity child) {
                auto *childNode = doc.allocate_node(rapidxml::node_element, "Child");
                childNode->append_attribute(doc.allocate_attribute(
                        "id",
                        doc.allocate_string(
                                std::to_string(child.GetIndex() - rootNodeIndex).c_str())));
                children->append_node(childNode);
            });
        }
    }

    auto *chains = doc.allocate_node(rapidxml::node_element, "Chains", "Chain");
    tree->append_node(chains);

    ExportChains(-1, rootInternode, chains, &doc);

    auto *leaves = doc.allocate_node(rapidxml::node_element, "Leaves", "Leaf");
    tree->append_node(leaves);
    int counter = 0;
    for (const auto &i : internodes) {
        glm::vec3 nodePos = i.GetDataComponent<GlobalTransform>().m_value[3];
        auto internodeData = i.GetOrSetPrivateComponent<InternodeData>().lock();
        for (const auto &leafTransform : internodeData->m_leavesTransforms) {
            auto *leaf = doc.allocate_node(rapidxml::node_element, "Leaf");
            leaf->append_attribute(doc.allocate_attribute(
                    "id", doc.allocate_string(std::to_string(counter).c_str())));
            counter++;
            leaves->append_node(leaf);

            auto *nodeAtt = doc.allocate_node(rapidxml::node_element, "Node");
            nodeAtt->append_attribute(doc.allocate_attribute(
                    "id", doc.allocate_string(std::to_string(i.GetIndex()).c_str())));
            leaf->append_node(nodeAtt);

            auto *posAtt = doc.allocate_node(rapidxml::node_element, "Center");
            posAtt->append_attribute(doc.allocate_attribute(
                    "x", doc.allocate_string(std::to_string(nodePos.x).c_str())));
            posAtt->append_attribute(doc.allocate_attribute(
                    "y", doc.allocate_string(std::to_string(nodePos.y).c_str())));
            posAtt->append_attribute(doc.allocate_attribute(
                    "z", doc.allocate_string(std::to_string(nodePos.z).c_str())));
            leaf->append_node(posAtt);

            Transform transform;
            transform.m_value = leafTransform;
            auto rotation = transform.GetRotation();

            auto *frontAtt = doc.allocate_node(rapidxml::node_element, "Forward");
            frontAtt->append_attribute(doc.allocate_attribute(
                    "x",
                    doc.allocate_string(
                            std::to_string((rotation * glm::vec3(0, 0, -1)).x).c_str())));
            frontAtt->append_attribute(doc.allocate_attribute(
                    "y",
                    doc.allocate_string(
                            std::to_string((rotation * glm::vec3(0, 0, -1)).y).c_str())));
            frontAtt->append_attribute(doc.allocate_attribute(
                    "z",
                    doc.allocate_string(
                            std::to_string((rotation * glm::vec3(0, 0, -1)).z).c_str())));
            leaf->append_node(frontAtt);

            auto *leftAtt = doc.allocate_node(rapidxml::node_element, "Left");
            leftAtt->append_attribute(doc.allocate_attribute(
                    "x", doc.allocate_string(
                            std::to_string((rotation * glm::vec3(1, 0, 0)).x).c_str())));
            leftAtt->append_attribute(doc.allocate_attribute(
                    "y", doc.allocate_string(
                            std::to_string((rotation * glm::vec3(1, 0, 0)).y).c_str())));
            leftAtt->append_attribute(doc.allocate_attribute(
                    "z", doc.allocate_string(
                            std::to_string((rotation * glm::vec3(1, 0, 0)).z).c_str())));
            leaf->append_node(leftAtt);

            auto *centerAtt = doc.allocate_node(rapidxml::node_element, "Position");
            centerAtt->append_attribute(doc.allocate_attribute(
                    "x",
                    doc.allocate_string(std::to_string(leafTransform[3].x).c_str())));
            centerAtt->append_attribute(doc.allocate_attribute(
                    "y",
                    doc.allocate_string(std::to_string(leafTransform[3].y).c_str())));
            centerAtt->append_attribute(doc.allocate_attribute(
                    "z",
                    doc.allocate_string(std::to_string(leafTransform[3].z).c_str())));
            leaf->append_node(centerAtt);

            auto *sizeAtt = doc.allocate_node(rapidxml::node_element, "Size");
            sizeAtt->append_attribute(doc.allocate_attribute(
                    "x", doc.allocate_string(std::to_string(0.105).c_str())));
            sizeAtt->append_attribute(doc.allocate_attribute(
                    "y", doc.allocate_string(std::to_string(0.1155).c_str())));
            leaf->append_node(sizeAtt);

            auto *distAtt = doc.allocate_node(rapidxml::node_element, "Dist");
            distAtt->append_attribute(doc.allocate_attribute(
                    "value", doc.allocate_string(
                            std::to_string(
                                    glm::distance(nodePos, glm::vec3(leafTransform[3])))
                                    .c_str())));
            leaf->append_node(distAtt);
        }
    }
}

void TreeSystem::InternodePostProcessor(const Entity &newInternode,
                                        const InternodeCandidate &candidate) {
    auto rigidBody = newInternode.GetOrSetPrivateComponent<RigidBody>().lock();
    rigidBody->SetDensityAndMassCenter(m_density *
                                       candidate.m_growth.m_thickness *
                                       candidate.m_growth.m_thickness);
    rigidBody->SetLinearDamping(m_linearDamping);
    rigidBody->SetAngularDamping(m_angularDamping);
    rigidBody->SetSolverIterations(m_positionSolverIteration,
                                   m_velocitySolverIteration);
    rigidBody->SetEnableGravity(false);
    // The rigidbody can only apply mesh bound after it's attached to an
    // entity with mesh renderer.
    auto joint = newInternode.GetOrSetPrivateComponent<Joint>().lock();
    joint->Link(candidate.m_parent);
    joint->SetType(JointType::D6);
    joint->SetMotion(MotionAxis::SwingY, MotionType::Free);
    joint->SetMotion(MotionAxis::SwingZ, MotionType::Free);
    joint->SetDrive(DriveType::Swing,
                    glm::pow(1.0f, m_jointDriveStiffnessThicknessFactor) *
                    m_jointDriveStiffnessFactor,
                    glm::pow(1.0f, m_jointDriveDampingThicknessFactor) *
                    m_jointDriveDampingFactor,
                    m_enableAccelerationForDrive);
}

void TreeSystem::DeleteAllPlantsHelper() { GenerateMeshForTree(); }

void TreeSystem::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_density" << YAML::Value << m_density;
    out << YAML::Key << "m_linearDamping" << YAML::Value << m_linearDamping;
    out << YAML::Key << "m_angularDamping" << YAML::Value << m_angularDamping;
    out << YAML::Key << "m_positionSolverIteration" << YAML::Value
        << m_positionSolverIteration;
    out << YAML::Key << "m_velocitySolverIteration" << YAML::Value
        << m_velocitySolverIteration;
    out << YAML::Key << "m_jointDriveStiffnessFactor" << YAML::Value
        << m_jointDriveStiffnessFactor;
    out << YAML::Key << "m_jointDriveStiffnessThicknessFactor" << YAML::Value
        << m_jointDriveStiffnessThicknessFactor;
    out << YAML::Key << "m_jointDriveDampingFactor" << YAML::Value
        << m_jointDriveDampingFactor;
    out << YAML::Key << "m_jointDriveDampingThicknessFactor" << YAML::Value
        << m_jointDriveDampingThicknessFactor;
    out << YAML::Key << "m_enableAccelerationForDrive" << YAML::Value
        << m_enableAccelerationForDrive;

    out << YAML::Key << "m_displayTime" << YAML::Value << m_displayTime;
    out << YAML::Key << "m_previousGlobalTime" << YAML::Value
        << m_previousGlobalTime;
    out << YAML::Key << "m_connectionWidth" << YAML::Value << m_connectionWidth;
    out << YAML::Key << "m_displayThickness" << YAML::Value << m_displayThickness;

    out << YAML::Key << "m_crownShynessDiameter" << YAML::Value
        << m_crownShynessDiameter;

    out << YAML::Key << "m_leafAmount" << YAML::Value << m_leafAmount;
    out << YAML::Key << "m_radius" << YAML::Value << m_radius;
    out << YAML::Key << "m_leafSize" << YAML::Value << m_leafSize;
    out << YAML::Key << "m_distanceToEndNode" << YAML::Value
        << m_distanceToEndNode;

    out << YAML::Key << "m_meshResolution" << YAML::Value << m_meshResolution;
    out << YAML::Key << "m_meshSubdivision" << YAML::Value << m_meshSubdivision;

    m_defaultRayTracingBranchAlbedoTexture.Save(
            "m_defaultRayTracingBranchAlbedoTexture", out);
    m_defaultRayTracingBranchNormalTexture.Save(
            "m_defaultRayTracingBranchNormalTexture", out);
    m_defaultBranchAlbedoTexture.Save("m_defaultBranchAlbedoTexture", out);
    m_defaultBranchNormalTexture.Save("m_defaultBranchNormalTexture", out);

    m_currentFocusingInternode.Save("m_currentFocusingInternode", out);

    m_plantSystem.Save("m_plantSystem", out);
}

void TreeSystem::Deserialize(const YAML::Node &in) {
    m_density = in["m_density"].as<float>();
    m_linearDamping = in["m_linearDamping"].as<float>();
    m_angularDamping = in["m_angularDamping"].as<float>();
    m_positionSolverIteration = in["m_positionSolverIteration"].as<int>();
    m_velocitySolverIteration = in["m_velocitySolverIteration"].as<int>();
    m_jointDriveStiffnessFactor = in["m_jointDriveStiffnessFactor"].as<float>();
    m_jointDriveStiffnessThicknessFactor =
            in["m_jointDriveStiffnessThicknessFactor"].as<float>();
    m_jointDriveDampingFactor = in["m_jointDriveDampingFactor"].as<float>();
    m_jointDriveDampingThicknessFactor =
            in["m_jointDriveDampingThicknessFactor"].as<float>();
    m_enableAccelerationForDrive = in["m_enableAccelerationForDrive"].as<bool>();

    m_displayTime = in["m_displayTime"].as<float>();
    m_previousGlobalTime = in["m_previousGlobalTime"].as<float>();
    m_connectionWidth = in["m_connectionWidth"].as<float>();
    m_displayThickness = in["m_displayThickness"].as<bool>();

    m_crownShynessDiameter = in["m_crownShynessDiameter"].as<float>();

    m_leafAmount = in["m_leafAmount"].as<int>();
    m_radius = in["m_radius"].as<float>();
    m_leafSize = in["m_leafSize"].as<glm::vec2>();
    m_distanceToEndNode = in["m_distanceToEndNode"].as<float>();

    m_meshResolution = in["m_meshResolution"].as<float>();
    m_meshSubdivision = in["m_meshSubdivision"].as<float>();

    m_defaultRayTracingBranchAlbedoTexture.Load(
            "m_defaultRayTracingBranchAlbedoTexture", in);
    m_defaultRayTracingBranchNormalTexture.Load(
            "m_defaultRayTracingBranchNormalTexture", in);
    m_defaultBranchAlbedoTexture.Load("m_defaultBranchAlbedoTexture", in);
    m_defaultBranchNormalTexture.Load("m_defaultBranchNormalTexture", in);

    m_currentFocusingInternode.Load("m_currentFocusingInternode", in);

    m_plantSystem.Load("m_plantSystem", in);
}

void TreeSystem::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_defaultRayTracingBranchAlbedoTexture);
    list.push_back(m_defaultRayTracingBranchNormalTexture);
    list.push_back(m_defaultBranchAlbedoTexture);
    list.push_back(m_defaultBranchNormalTexture);
}

void TreeSystem::OnCreate() {
#pragma region Internode camera
    m_internodeDebuggingCamera =
            SerializationManager::ProduceSerializable<Camera>();
    m_internodeDebuggingCamera->m_useClearColor = true;
    m_internodeDebuggingCamera->m_clearColor = glm::vec3(0.1f);
    m_internodeDebuggingCamera->OnCreate();
#pragma endregion
    Enable();
}

void TreeSystem::LateUpdate() {
#pragma region Internode debugging camera
    ImVec2 viewPortSize;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0});
    ImGui::Begin("Tree Internodes");
    {
        if (ImGui::BeginChild("CameraRenderer", ImVec2(0, 0), false,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Settings")) {
#pragma region Menu
                    ImGui::Checkbox("Force update", &m_alwaysUpdate);
                    ImGui::SliderFloat("Display Time", &m_displayTime, 0.0f,
                                       m_plantSystem.Get<PlantSystem>()->m_globalTime);
                    if (ImGui::ButtonEx(
                            "To present", ImVec2(0, 0),
                            m_displayTime !=
                            m_plantSystem.Get<PlantSystem>()->m_globalTime
                            ? 0
                            : ImGuiButtonFlags_Disabled))
                        m_displayTime = m_plantSystem.Get<PlantSystem>()->m_globalTime;
                    if (m_displayTime != m_plantSystem.Get<PlantSystem>()->m_globalTime) {
                        ImGui::SameLine();
                        if (ImGui::Button("Start from here.")) {
                            ResetTimeForTree(m_displayTime);
                        }
                    }

                    ImGui::Checkbox("Connections", &m_drawBranches);
                    if (m_drawBranches) {
                        if (ImGui::TreeNodeEx("Connection settings",
                                              ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::Combo("Render type", (int *) &m_branchRenderType,
                                         BranchRenderTypes, IM_ARRAYSIZE(BranchRenderTypes));
                            ImGui::Checkbox("As transparency", &m_useTransparency);
                            if (m_useTransparency)
                                ImGui::SliderFloat("Alpha", &m_transparency, 0, 1);
                            ImGui::Checkbox("Compress", &m_enableBranchDataCompress);
                            if (m_enableBranchDataCompress)
                                ImGui::DragFloat("Compress factor", &m_branchCompressFactor,
                                                 0.01f, 0.01f, 1.0f);
                            ImGui::Checkbox("Color Map", &m_useColorMap);
                            if (m_useColorMap) {
                                static int savedAmount = 3;
                                ImGui::SliderInt("Slot amount", &m_colorMapSegmentAmount, 2,
                                                 10);
                                if (savedAmount != m_colorMapSegmentAmount) {
                                    m_colorMapValues.resize(m_colorMapSegmentAmount);
                                    m_colorMapColors.resize(m_colorMapSegmentAmount);
                                    for (int i = 0; i < m_colorMapSegmentAmount; i++) {
                                        if (i != 0 && m_colorMapValues[i] < m_colorMapValues[i - 1])
                                            m_colorMapValues[i] = m_colorMapValues[i - 1] + 1;
                                        if (i >= savedAmount)
                                            m_colorMapColors[i] =
                                                    glm::linearRand(glm::vec3(0.0f), glm::vec3(1.0f));
                                    }
                                    savedAmount = m_colorMapSegmentAmount;
                                }
                                for (int i = 0; i < m_colorMapValues.size(); i++) {
                                    if (i == 0) {
                                        ImGui::DragFloat("Value 0", &m_colorMapValues[0], 0.1f,
                                                         0.0f, m_colorMapValues[1]);
                                        ImGui::ColorEdit3("Color 0", &m_colorMapColors[0].x);
                                    } else if (i == m_colorMapValues.size() - 1) {
                                        ImGui::DragFloat(("Value" + std::to_string(i)).c_str(),
                                                         &m_colorMapValues[i], 0.1f,
                                                         m_colorMapValues[i - 1] + 0.1f, 9999.0f);
                                        ImGui::ColorEdit3(("Color" + std::to_string(i)).c_str(),
                                                          &m_colorMapColors[i].x);
                                    } else {
                                        ImGui::DragFloat(("Value" + std::to_string(i)).c_str(),
                                                         &m_colorMapValues[i], 0.1f,
                                                         m_colorMapValues[i - 1] + 0.1f,
                                                         m_colorMapValues[i + 1]);
                                        ImGui::ColorEdit3(("Color" + std::to_string(i)).c_str(),
                                                          &m_colorMapColors[i].x);
                                    }
                                }
                            }

                            if (ImGui::Checkbox("Display thickness", &m_displayThickness))
                                m_updateBranch = true;
                            if (!m_displayThickness)
                                if (ImGui::DragFloat("Connection width", &m_connectionWidth,
                                                     0.01f, 0.01f, 1.0f))
                                    m_updateBranch = true;
                            ImGui::TreePop();
                        }
                    }
                    ImGui::Checkbox("Pointers", &m_drawPointers);
                    if (m_drawPointers) {
                        if (ImGui::TreeNodeEx("Pointer settings",
                                              ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::Combo("Render type", (int *) &m_pointerRenderType,
                                         PointerRenderTypes,
                                         IM_ARRAYSIZE(PointerRenderTypes));
                            ImGui::Checkbox("Compress", &m_enablePointerDataCompress);
                            if (m_pointerCompressFactor)
                                ImGui::DragFloat("Compress factor", &m_branchCompressFactor,
                                                 0.01f, 0.01f, 1.0f);
                            if (ImGui::ColorEdit4("Pointer color", &m_pointerColor.x))
                                m_updatePointer = true;
                            if (ImGui::DragFloat("Pointer length", &m_pointerLength, 0.01f,
                                                 0.01f, 3.0f))
                                m_updatePointer = true;
                            if (ImGui::DragFloat("Pointer width", &m_pointerWidth, 0.01f,
                                                 0.01f, 1.0f))
                                m_updatePointer = true;
                            ImGui::TreePop();
                        }
                    }
                    m_voxelSpaceModule.OnGui();

#pragma endregion
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
            viewPortSize = ImGui::GetWindowSize();
            viewPortSize.y -= 20;
            if (viewPortSize.y < 0)
                viewPortSize.y = 0;
            m_internodeDebuggingCameraResolutionX = viewPortSize.x;
            m_internodeDebuggingCameraResolutionY = viewPortSize.y;
            ImGui::Image(
                    reinterpret_cast<ImTextureID>(
                            m_internodeDebuggingCamera->GetTexture()->Texture()->Id()),
                    viewPortSize, ImVec2(0, 1), ImVec2(1, 0));
            glm::vec2 mousePosition = glm::vec2(FLT_MAX, FLT_MIN);
            if (ImGui::IsWindowFocused()) {
                bool valid = true;
                mousePosition = InputManager::GetMouseAbsolutePositionInternal(
                        WindowManager::GetWindow());
                float xOffset = 0;
                float yOffset = 0;
                if (valid) {
                    if (!m_startMouse) {
                        m_lastX = mousePosition.x;
                        m_lastY = mousePosition.y;
                        m_startMouse = true;
                    }
                    xOffset = mousePosition.x - m_lastX;
                    yOffset = -mousePosition.y + m_lastY;
                    m_lastX = mousePosition.x;
                    m_lastY = mousePosition.y;
#pragma region Scene Camera Controller
                    if (!m_rightMouseButtonHold &&
                        InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT,
                                                       WindowManager::GetWindow())) {
                        m_rightMouseButtonHold = true;
                    }
                    if (m_rightMouseButtonHold &&
                        !EditorManager::GetInstance().m_lockCamera) {
                        glm::vec3 front =
                                EditorManager::GetInstance().m_sceneCameraRotation *
                                glm::vec3(0, 0, -1);
                        glm::vec3 right =
                                EditorManager::GetInstance().m_sceneCameraRotation *
                                glm::vec3(1, 0, 0);
                        if (InputManager::GetKeyInternal(GLFW_KEY_W,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition +=
                                    front * static_cast<float>(Application::Time().DeltaTime()) *
                                    EditorManager::GetInstance().m_velocity;
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_S,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition -=
                                    front * static_cast<float>(Application::Time().DeltaTime()) *
                                    EditorManager::GetInstance().m_velocity;
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_A,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition -=
                                    right * static_cast<float>(Application::Time().DeltaTime()) *
                                    EditorManager::GetInstance().m_velocity;
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_D,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition +=
                                    right * static_cast<float>(Application::Time().DeltaTime()) *
                                    EditorManager::GetInstance().m_velocity;
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_LEFT_SHIFT,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition.y +=
                                    EditorManager::GetInstance().m_velocity *
                                    static_cast<float>(Application::Time().DeltaTime());
                        }
                        if (InputManager::GetKeyInternal(GLFW_KEY_LEFT_CONTROL,
                                                         WindowManager::GetWindow())) {
                            EditorManager::GetInstance().m_sceneCameraPosition.y -=
                                    EditorManager::GetInstance().m_velocity *
                                    static_cast<float>(Application::Time().DeltaTime());
                        }
                        if (xOffset != 0.0f || yOffset != 0.0f) {
                            EditorManager::GetInstance().m_sceneCameraYawAngle +=
                                    xOffset * EditorManager::GetInstance().m_sensitivity;
                            EditorManager::GetInstance().m_sceneCameraPitchAngle +=
                                    yOffset * EditorManager::GetInstance().m_sensitivity;
                            if (EditorManager::GetInstance().m_sceneCameraPitchAngle > 89.0f)
                                EditorManager::GetInstance().m_sceneCameraPitchAngle = 89.0f;
                            if (EditorManager::GetInstance().m_sceneCameraPitchAngle < -89.0f)
                                EditorManager::GetInstance().m_sceneCameraPitchAngle = -89.0f;

                            EditorManager::GetInstance().m_sceneCameraRotation =
                                    Camera::ProcessMouseMovement(
                                            EditorManager::GetInstance().m_sceneCameraYawAngle,
                                            EditorManager::GetInstance().m_sceneCameraPitchAngle,
                                            false);
                        }
                    }
#pragma endregion
                    if (m_drawBranches) {
#pragma region Ray selection
                        m_currentFocusingInternode = Entity();
                        std::mutex writeMutex;
                        auto windowPos = ImGui::GetWindowPos();
                        auto windowSize = ImGui::GetWindowSize();
                        mousePosition.x -= windowPos.x;
                        mousePosition.x -= windowSize.x;
                        mousePosition.y -= windowPos.y + 20;
                        float minDistance = FLT_MAX;
                        GlobalTransform cameraLtw;
                        cameraLtw.m_value =
                                glm::translate(
                                        EditorManager::GetInstance().m_sceneCameraPosition) *
                                glm::mat4_cast(
                                        EditorManager::GetInstance().m_sceneCameraRotation);
                        const Ray cameraRay = m_internodeDebuggingCamera->ScreenPointToRay(
                                cameraLtw, mousePosition);
                        EntityManager::ForEach<GlobalTransform, BranchCylinderWidth,
                                InternodeGrowth>(
                                JobManager::PrimaryWorkers(),
                                m_plantSystem.Get<PlantSystem>()->m_internodeQuery,
                                [&, cameraLtw, cameraRay](int i, Entity entity,
                                                          GlobalTransform &ltw,
                                                          BranchCylinderWidth &width,
                                                          InternodeGrowth &internodeGrowth) {
                                    const glm::vec3 position = ltw.m_value[3];
                                    const auto parentPosition =
                                            entity.GetParent()
                                                    .GetDataComponent<GlobalTransform>()
                                                    .GetPosition();
                                    const auto center = (position + parentPosition) / 2.0f;
                                    auto dir = cameraRay.m_direction;
                                    auto pos = cameraRay.m_start;
                                    const auto radius = width.m_value;
                                    const auto height = glm::distance(parentPosition, position);
                                    if (internodeGrowth.m_distanceToRoot == 0) {
                                        if (!cameraRay.Intersect(
                                                position,
                                                glm::max(0.2f, internodeGrowth.m_thickness * 4.0f)))
                                            return;
                                    } else {
                                        if (!cameraRay.Intersect(center, height / 2.0f))
                                            return;

#pragma region Line Line intersection
                                        /*
                     * http://geomalgorithms.com/a07-_distance.html
                     */
                                        glm::vec3 u = pos - (pos + dir);
                                        glm::vec3 v = position - parentPosition;
                                        glm::vec3 w = (pos + dir) - parentPosition;
                                        const auto a = dot(u,
                                                           u); // always >= 0
                                        const auto b = dot(u, v);
                                        const auto c = dot(v,
                                                           v); // always >= 0
                                        const auto d = dot(u, w);
                                        const auto e = dot(v, w);
                                        const auto dotP = a * c - b * b; // always >= 0
                                        float sc, tc;
                                        // compute the line parameters of the two closest points
                                        if (dotP < 0.001f) { // the lines are almost parallel
                                            sc = 0.0f;
                                            tc = (b > c ? d / b
                                                        : e / c); // use the largest denominator
                                        } else {
                                            sc = (b * e - c * d) / dotP;
                                            tc = (a * e - b * d) / dotP;
                                        }
                                        // get the difference of the two closest points
                                        glm::vec3 dP = w + sc * u - tc * v; // =  L1(sc) - L2(tc)
                                        if (glm::length(dP) > radius)
                                            return;
#pragma endregion
                                    }
                                    const auto distance = glm::distance(
                                            glm::vec3(cameraLtw.m_value[3]), glm::vec3(center));
                                    std::lock_guard<std::mutex> lock(writeMutex);
                                    if (distance < minDistance) {
                                        minDistance = distance;
                                        m_currentFocusingInternode = entity;
                                    }
                                });
                        if (InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_LEFT,
                                                           WindowManager::GetWindow())) {
                            if (!m_currentFocusingInternode.Get().IsNull()) {
                                EditorManager::SetSelectedEntity(
                                        m_currentFocusingInternode.Get());
                            }
                        }
#pragma endregion
                    }
                }
            }
        }
        ImGui::EndChild();
        auto *window = ImGui::FindWindowByName("Tree Internodes");
        m_internodeDebuggingCamera->SetEnabled(
                !(window->Hidden && !window->Collapsed));
    }
    ImGui::End();
    ImGui::PopStyleVar();

#pragma endregion
}
