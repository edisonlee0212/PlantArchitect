#include "InternodeModel/InternodeLayer.hpp"
#include "DataComponents.hpp"
#include <RadialBoundingVolume.hpp>
#include "InternodeModel/Internode.hpp"
#include "Graphics.hpp"
#include "DefaultResources.hpp"
#include "PointCloud.hpp"

using namespace PlantArchitect;
using namespace UniEngine;

glm::vec3 RadialBoundingVolume::GetRandomPoint() {
    if (!m_meshGenerated)
        return glm::vec3(0);
    float sizePoint = glm::linearRand(0.0f, m_totalSize);
    int layerIndex = 0;
    int sectorIndex = 0;
    bool found = false;
    for (int i = 0; i < m_layerAmount; i++) {
        if (sizePoint > m_sizes[i].first) {
            sizePoint -= m_sizes[i].first;
            continue;
        }
        for (int j = 0; j < m_sectorAmount; j++) {
            auto &sector = m_sizes[i].second;
            if (sizePoint > sector[j]) {
                sizePoint -= sector[j];
                continue;
            } else {
                layerIndex = i;
                sectorIndex = j;
                found = true;
                break;
            }
        }
        break;
    }
    if (!found) {
        layerIndex = m_layerAmount - 1;
        sectorIndex = m_sectorAmount - 1;
    }
    const float heightLevel = m_maxHeight / m_layerAmount;
    const float sliceAngle = 360.0f / m_sectorAmount;
    float height = heightLevel * layerIndex + glm::linearRand(0.0f, heightLevel);
    float angle = sliceAngle * sectorIndex + glm::linearRand(0.0f, sliceAngle);
    float distance = m_layers[layerIndex][sectorIndex].m_maxDistance * glm::length(glm::diskRand(1.0f));
    const auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
    return globalTransform.m_value *
           glm::vec4(distance * glm::sin(glm::radians(angle)), height, distance * glm::cos(glm::radians(angle)), 1.0f);
}

glm::ivec2 RadialBoundingVolume::SelectSlice(glm::vec3 position) const {
    glm::ivec2 retVal;
    const float heightLevel = m_maxHeight / m_layerAmount;
    const float sliceAngle = 360.0f / m_sectorAmount;
    auto x = static_cast<int>(position.y / heightLevel);
    if (x < 0)
        x = 0;
    retVal.x = x;
    if (retVal.x >= m_layerAmount)
        retVal.x = m_layerAmount - 1;
    if (position.x == 0 && position.z == 0)
        retVal.y = 0;
    else
        retVal.y = static_cast<int>(
                (glm::degrees(glm::atan(position.x, position.z)) + 180.0f) /
                sliceAngle);
    if (retVal.y >= m_sectorAmount)
        retVal.y = m_sectorAmount - 1;
    return retVal;
}

void RadialBoundingVolume::GenerateMesh() {
    m_boundMeshes.clear();
    if (m_layers.empty())
        return;
    for (int tierIndex = 0; tierIndex < m_layerAmount; tierIndex++) {
        auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
        std::vector<Vertex> vertices;
        std::vector<unsigned> indices;

        const float sliceAngle = 360.0f / m_sectorAmount;
        const int totalAngleStep = 360.0f / m_sectorAmount;
        const int totalLevelStep = 2;
        const float stepAngle = sliceAngle / (totalAngleStep - 1);
        const float heightLevel = m_maxHeight / m_layerAmount;
        const float stepLevel = heightLevel / (totalLevelStep - 1);
        vertices.resize(totalLevelStep * m_sectorAmount * totalAngleStep * 2 +
                        totalLevelStep);
        indices.resize((12 * (totalLevelStep - 1) * totalAngleStep) *
                       m_sectorAmount);
        for (int levelStep = 0; levelStep < totalLevelStep; levelStep++) {
            const float currentHeight =
                    heightLevel * tierIndex + stepLevel * levelStep;
            for (int sliceIndex = 0; sliceIndex < m_sectorAmount; sliceIndex++) {
                for (int angleStep = 0; angleStep < totalAngleStep; angleStep++) {
                    const int actualAngleStep = sliceIndex * totalAngleStep + angleStep;

                    float currentAngle = sliceAngle * sliceIndex + stepAngle * angleStep;
                    if (currentAngle >= 360)
                        currentAngle = 0;
                    float x = glm::abs(glm::tan(glm::radians(currentAngle)));
                    float z = 1.0f;
                    if (currentAngle >= 0 && currentAngle <= 90) {
                        z *= -1;
                        x *= -1;
                    } else if (currentAngle > 90 && currentAngle <= 180) {
                        x *= -1;
                    } else if (currentAngle > 270 && currentAngle <= 360) {
                        z *= -1;
                    }
                    glm::vec3 position = glm::normalize(glm::vec3(x, 0.0f, z)) *
                                         m_layers[tierIndex][sliceIndex].m_maxDistance;
                    position.y = currentHeight;
                    vertices[levelStep * totalAngleStep * m_sectorAmount +
                             actualAngleStep]
                            .m_position = position;
                    vertices[levelStep * totalAngleStep * m_sectorAmount +
                             actualAngleStep]
                            .m_texCoord = glm::vec2((float) levelStep / (totalLevelStep - 1),
                                                     (float) angleStep / (totalAngleStep - 1));
                    vertices[levelStep * totalAngleStep * m_sectorAmount +
                             actualAngleStep]
                            .m_normal = glm::normalize(position);
                    vertices[totalLevelStep * m_sectorAmount * totalAngleStep +
                             levelStep * totalAngleStep * m_sectorAmount +
                             actualAngleStep]
                            .m_position = position;
                    vertices[totalLevelStep * m_sectorAmount * totalAngleStep +
                             levelStep * totalAngleStep * m_sectorAmount +
                             actualAngleStep]
                            .m_texCoord = glm::vec2((float) levelStep / (totalLevelStep - 1),
                                                     (float) angleStep / (totalAngleStep - 1));
                    vertices[totalLevelStep * m_sectorAmount * totalAngleStep +
                             levelStep * totalAngleStep * m_sectorAmount +
                             actualAngleStep]
                            .m_normal = glm::vec3(0, levelStep == 0 ? -1 : 1, 0);
                }
            }
            vertices[vertices.size() - totalLevelStep + levelStep].m_position =
                    glm::vec3(0, currentHeight, 0);
            vertices[vertices.size() - totalLevelStep + levelStep].m_normal =
                    glm::vec3(0, levelStep == 0 ? -1 : 1, 0);
            vertices[vertices.size() - totalLevelStep + levelStep].m_texCoord =
                    glm::vec2(0.0f);
        }
        for (int levelStep = 0; levelStep < totalLevelStep - 1; levelStep++) {
            for (int sliceIndex = 0; sliceIndex < m_sectorAmount; sliceIndex++) {
                for (int angleStep = 0; angleStep < totalAngleStep; angleStep++) {
                    const int actualAngleStep =
                            sliceIndex * totalAngleStep + angleStep; // 0-5
                    // Fill a quad here.
                    if (actualAngleStep < m_sectorAmount * totalAngleStep - 1) {
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep)] =
                                levelStep * totalAngleStep * m_sectorAmount + actualAngleStep;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                1] = levelStep * totalAngleStep * m_sectorAmount +
                                     actualAngleStep + 1;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                2] = (levelStep + 1) * totalAngleStep * m_sectorAmount +
                                     actualAngleStep;

                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                3] = (levelStep + 1) * totalAngleStep * m_sectorAmount +
                                     actualAngleStep + 1;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                4] = (levelStep + 1) * totalAngleStep * m_sectorAmount +
                                     actualAngleStep;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                5] = levelStep * totalAngleStep * m_sectorAmount +
                                     actualAngleStep + 1;
                        // Connect with center here.
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                6] = totalLevelStep * m_sectorAmount * totalAngleStep +
                                     levelStep * totalAngleStep * m_sectorAmount +
                                     actualAngleStep;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                7] = vertices.size() - totalLevelStep + levelStep;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                8] = totalLevelStep * m_sectorAmount * totalAngleStep +
                                     levelStep * totalAngleStep * m_sectorAmount +
                                     actualAngleStep + 1;

                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                9] = totalLevelStep * m_sectorAmount * totalAngleStep +
                                     (levelStep + 1) * totalAngleStep * m_sectorAmount +
                                     actualAngleStep + 1;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                10] = vertices.size() - totalLevelStep + (levelStep + 1);
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                11] = totalLevelStep * m_sectorAmount * totalAngleStep +
                                      (levelStep + 1) * totalAngleStep * m_sectorAmount +
                                      actualAngleStep;
                    } else {
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep)] =
                                levelStep * totalAngleStep * m_sectorAmount + actualAngleStep;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                1] = levelStep * totalAngleStep * m_sectorAmount;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                2] = (levelStep + 1) * totalAngleStep * m_sectorAmount +
                                     actualAngleStep;

                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                3] = (levelStep + 1) * totalAngleStep * m_sectorAmount;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                4] = (levelStep + 1) * totalAngleStep * m_sectorAmount +
                                     actualAngleStep;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                5] = levelStep * totalAngleStep * m_sectorAmount;
                        // Connect with center here.
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                6] = totalLevelStep * m_sectorAmount * totalAngleStep +
                                     levelStep * totalAngleStep * m_sectorAmount +
                                     actualAngleStep;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                7] = vertices.size() - totalLevelStep + levelStep;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                8] = totalLevelStep * m_sectorAmount * totalAngleStep +
                                     levelStep * totalAngleStep * m_sectorAmount;

                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                9] = totalLevelStep * m_sectorAmount * totalAngleStep +
                                     (levelStep + 1) * totalAngleStep * m_sectorAmount;
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                10] = vertices.size() - totalLevelStep + (levelStep + 1);
                        indices[12 * (levelStep * totalAngleStep * m_sectorAmount +
                                      actualAngleStep) +
                                11] = totalLevelStep * m_sectorAmount * totalAngleStep +
                                      (levelStep + 1) * totalAngleStep * m_sectorAmount +
                                      actualAngleStep;
                    }
                }
            }
        }
        mesh->SetVertices(19, vertices, indices);
        m_boundMeshes.push_back(std::move(mesh));
    }
    m_meshGenerated = true;
}

void RadialBoundingVolume::FormEntity() {
    if (!m_meshGenerated)
        CalculateVolume();
    if (!m_meshGenerated)
        return;
    auto scene = GetScene();
    auto children = scene->GetChildren(GetOwner());
    for (auto &child: children) {
        if(scene->GetEntityName(child) == "RBV Geometry") scene->DeleteEntity(child);
    }
    children.clear();
    auto rbvEntity = scene->CreateEntity("RBV Geometry");
    scene->SetParent(rbvEntity, GetOwner(), false);
    for (auto i = 0; i < m_boundMeshes.size(); i++) {
        auto slice = scene->CreateEntity("RBV_" + std::to_string(i));
        auto mmc = scene->GetOrSetPrivateComponent<MeshRenderer>(slice).lock();
        auto mat = ProjectManager::CreateTemporaryAsset<Material>();
        mmc->m_material = mat;
        mat->SetProgram(DefaultResources::GLPrograms::StandardProgram);
        mmc->m_forwardRendering = false;
        mmc->m_mesh = m_boundMeshes[i];
        scene->SetParent(slice, rbvEntity, false);
    }
}

std::string RadialBoundingVolume::Save() {
    if (!m_meshGenerated)
        CalculateVolume();
    std::string output;
    output += std::to_string(m_layerAmount) + "\n";
    output += std::to_string(m_sectorAmount) + "\n";
    output += std::to_string(m_maxHeight) + "\n";
    output += std::to_string(m_maxRadius) + "\n";
    int tierIndex = 0;
    for (const auto &tier: m_layers) {
        int sliceIndex = 0;
        for (const auto &slice: tier) {
            output += std::to_string(slice.m_maxDistance);
            output += "\n";
            sliceIndex++;
        }
        tierIndex++;
    }
    output += "\n";
    for (const auto &tier: m_layers) {
        int sliceIndex = 0;
        for (const auto &slice: tier) {
            output += std::to_string(slice.m_maxDistance);
            output += ",";
            sliceIndex++;
        }
        tierIndex++;
    }
    output += "\n";
    return output;
}

void RadialBoundingVolume::ExportAsObj(const std::string &filename) {
    if (!m_meshGenerated)
        CalculateVolume();
    auto &meshes = m_boundMeshes;

    std::ofstream of;
    of.open((filename + ".obj").c_str(),
            std::ofstream::out | std::ofstream::trunc);
    if (of.is_open()) {
        std::string o = "o ";
        o += "RBV\n";
        of.write(o.c_str(), o.size());
        of.flush();
        std::string data;
        int offset = 1;
#pragma region Data collection
        for (auto &mesh: meshes) {
            for (const auto &vertices: mesh->UnsafeGetVertices()) {
                data += "v " + std::to_string(vertices.m_position.x) + " " +
                        std::to_string(vertices.m_position.y) + " " +
                        std::to_string(vertices.m_position.z) + "\n";
            }
        }
        for (auto &mesh: meshes) {
            data += "# List of indices for faces vertices, with (x, y, z).\n";
            auto &triangles = mesh->UnsafeGetTriangles();
            for (auto i = 0; i < triangles.size(); i++) {
                auto f1 = triangles.at(i).x + offset;
                auto f2 = triangles.at(i).y + offset;
                auto f3 = triangles.at(i).z + offset;
                data += "f " + std::to_string(f1) + " " + std::to_string(f2) + " " +
                        std::to_string(f3) + "\n";
            }
            offset += mesh->GetVerticesAmount();
        }
#pragma endregion
        of.write(data.c_str(), data.size());
        of.flush();
    }
}

void RadialBoundingVolume::Load(const std::string &path) {
    std::ifstream ifs;
    ifs.open(path.c_str());
    UNIENGINE_LOG("Loading from " + path);
    if (ifs.is_open()) {
        ifs >> m_layerAmount;
        ifs >> m_sectorAmount;
        ifs >> m_maxHeight;
        ifs >> m_maxRadius;
        m_layers.resize(m_layerAmount);
        for (auto &tier: m_layers) {
            tier.resize(m_sectorAmount);
            for (auto &slice: tier) {
                ifs >> slice.m_maxDistance;
            }
        }
        GenerateMesh();
    }
}

void RadialBoundingVolume::CalculateVolume() {
    ResizeVolumes();
    std::vector<Entity> internodes;
    auto internode = m_rootInternode.Get<Internode>();
    if (!internode) {
        UNIENGINE_WARNING("No root internode!");
        return;
    }
    internode->CollectInternodes(internodes);
    m_maxHeight = 0;
    m_maxRadius = 0;
    auto scene = GetScene();
    const auto rootPosition =
            scene->GetDataComponent<GlobalTransform>(internode->GetOwner()).GetPosition();
    std::vector<glm::vec3> positions;
    for (auto &i: internodes) {
        const glm::vec3 position = scene->GetDataComponent<GlobalTransform>(i).GetPosition() - rootPosition;
        positions.push_back(position);
        if (position.y > m_maxHeight)
            m_maxHeight = position.y;
        const float radius = glm::length(glm::vec2(position.x, position.z));
        if (radius > m_maxRadius)
            m_maxRadius = radius;
    }

    auto positionIndex = 0;
    for (auto &internode: internodes) {
        const auto internodeGrowth = scene->GetDataComponent<InternodeInfo>(internode);
        const glm::vec3 position = positions[positionIndex];
        const auto sliceIndex = SelectSlice(position);
        const float currentDistance =
                glm::length(glm::vec2(position.x, position.z));
        if (currentDistance <= internodeGrowth.m_thickness) {
            for (auto &slice: m_layers[sliceIndex.x]) {
                if (slice.m_maxDistance <
                    currentDistance + internodeGrowth.m_thickness)
                    slice.m_maxDistance = currentDistance + internodeGrowth.m_thickness;
            }
        } else if (m_layers[sliceIndex.x][sliceIndex.y].m_maxDistance <
                   currentDistance)
            m_layers[sliceIndex.x][sliceIndex.y].m_maxDistance = currentDistance;
        positionIndex++;
    }
    GenerateMesh();
    CalculateSizes();
}

void RadialBoundingVolume::CalculateVolume(const std::vector<glm::vec3>& points)
{
    ResizeVolumes();
    m_maxHeight = 0;
    m_maxRadius = 0;
    auto scene = GetScene();
    for (const auto& point : points) {
        if (point.y > m_maxHeight)
            m_maxHeight = point.y;
        const float radius = glm::length(glm::vec2(point.x, point.z));
        if (radius > m_maxRadius)
            m_maxRadius = radius;
    }

    auto positionIndex = 0;
    for (const auto& point : points) {
        const auto sliceIndex = SelectSlice(point);
        const float currentDistance =
            glm::length(glm::vec2(point.x, point.z));
        if (currentDistance <= 0.2f) {
            for (auto& slice : m_layers[sliceIndex.x]) {
                if (slice.m_maxDistance <
                    currentDistance + 0.2f)
                    slice.m_maxDistance = currentDistance + 0.2f;
            }
        }
        else if (m_layers[sliceIndex.x][sliceIndex.y].m_maxDistance <
            currentDistance)
            m_layers[sliceIndex.x][sliceIndex.y].m_maxDistance = currentDistance + 0.2f;
        positionIndex++;
    }
    GenerateMesh();
    CalculateSizes();
}

void RadialBoundingVolume::CalculateVolume(float maxHeight) {
    auto owner = GetOwner();
    ResizeVolumes();
    auto scene = GetScene();
    if (!scene->HasPrivateComponent<Internode>(owner)) {
        UNIENGINE_WARNING("Entity is not an internode!");
        return;
    }
    std::vector<Entity> internodes;
    auto internode = scene->GetOrSetPrivateComponent<Internode>(owner).lock();
    internode->CollectInternodes(internodes);
    m_maxHeight = maxHeight;
    m_maxRadius = 0;
    const auto rootPosition =
            scene->GetDataComponent<GlobalTransform>(internode->GetOwner()).GetPosition();
    std::vector<glm::vec3> positions;
    for (auto &i: internodes) {
        const glm::vec3 position = scene->GetDataComponent<GlobalTransform>(i).GetPosition() - rootPosition;
        positions.push_back(position);
        if (position.y > m_maxHeight)
            m_maxHeight = position.y;
        const float radius = glm::length(glm::vec2(position.x, position.z));
        if (radius > m_maxRadius)
            m_maxRadius = radius;
    }
    const auto threadsAmount = Jobs::Workers().Size();
    std::vector<std::vector<std::vector<RadialBoundingVolumeSlice>>>
            tempCakeTowers;
    tempCakeTowers.resize(threadsAmount);
    for (int i = 0; i < threadsAmount; i++) {
        tempCakeTowers[i].resize(m_layerAmount);
        for (auto &tier: tempCakeTowers[i]) {
            tier.resize(m_sectorAmount);
            for (auto &slice: tier) {
                slice.m_maxDistance = 0.0f;
            }
        }
    }
    auto positionIndex = 0;
    for (auto &internode: internodes) {
        const auto internodeGrowth = scene->GetDataComponent<InternodeInfo>(internode);
        const int segments = 3;
        for (int i = 0; i < segments; i++) {
            const glm::vec3 position = positions[positionIndex];
            const auto sliceIndex = SelectSlice(position);
            const float currentDistance =
                    glm::length(glm::vec2(position.x, position.z));
            if (currentDistance <= internodeGrowth.m_thickness) {
                for (auto &slice: m_layers[sliceIndex.x]) {
                    if (slice.m_maxDistance <
                        currentDistance + internodeGrowth.m_thickness)
                        slice.m_maxDistance = currentDistance + internodeGrowth.m_thickness;
                }
            } else if (m_layers[sliceIndex.x][sliceIndex.y].m_maxDistance <
                       currentDistance)
                m_layers[sliceIndex.x][sliceIndex.y].m_maxDistance = currentDistance;
        }
        positionIndex++;
    }
    GenerateMesh();
    CalculateSizes();
}

void RadialBoundingVolume::OnInspect() {
    IVolume::OnInspect();
    Editor::DragAndDropButton<Internode>(m_rootInternode, "Root Internode");
    auto scene = GetScene();
    ImGui::Checkbox("Prune Buds", &m_pruneBuds);
    ImGui::ColorEdit4("Display Color", &m_displayColor.x);
    ImGui::DragFloat("Display Scale", &m_displayScale, 0.01f, 0.01f, 1.0f);
    bool edited = false;
    if (ImGui::DragInt("Layer Amount", &m_layerAmount, 1, 1, 100))
        edited = true;
    if (ImGui::DragInt("Slice Amount", &m_sectorAmount, 1, 1, 100))
        edited = true;
    if (ImGui::Button("Calculate Bounds") || edited)
        CalculateVolume();
    if (ImGui::Button("Form Entity")) {
        FormEntity();
    }

    bool displayLayer = false;
    if (m_meshGenerated) {
        if (ImGui::TreeNodeEx("Transformations")) {
            ImGui::DragFloat("Max height", &m_maxHeight, 0.01f);
            static float augmentation = 1.0f;
            ImGui::DragFloat("Augmentation radius", &augmentation, 0.01f);
            if (ImGui::Button("Process")) {
                Augmentation(augmentation);
            }
            if (ImGui::Button("Generate mesh")) {
                GenerateMesh();
            }
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Layers", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (int i = 0; i < m_layerAmount; i++) {
                if (ImGui::TreeNodeEx(("Layer " + std::to_string(i)).c_str())) {
                    for (int j = 0; j < m_sectorAmount; j++) {
                        if (ImGui::DragFloat(
                                ("Sector " + std::to_string(j) + "##" + std::to_string(i))
                                        .c_str(),
                                &m_layers[i][j].m_maxDistance, 0.1f, 0.0f, 100.0f))
                            GenerateMesh();
                    }

                    Gizmos::DrawGizmoMesh(
                            m_boundMeshes[i], m_displayColor,
                            scene->GetDataComponent<GlobalTransform>(GetOwner()).m_value);
                    displayLayer = true;
                    ImGui::TreePop();
                }
            }
            ImGui::TreePop();
        }
    }
    FileUtils::SaveFile("Save RBV", "RBV", {".rbv"},
                        [this](const std::filesystem::path &path) {
                            const std::string data = Save();
                            std::ofstream ofs;
                            ofs.open(path.string().c_str(),
                                     std::ofstream::out | std::ofstream::trunc);
                            ofs.write(data.c_str(), data.length());
                            ofs.flush();
                            ofs.close();
                        });
    FileUtils::OpenFile(
            "Load RBV", "RBV", {".rbv"},
            [this](const std::filesystem::path &path) { Load(path.string()); });
    FileUtils::SaveFile("Export RBV as OBJ", "3D Model", {".obj"},
                        [this](const std::filesystem::path &path) {
                            ExportAsObj(path.string());
                        });

    static AssetRef pointCloud;
    if (Editor::DragAndDropButton(pointCloud, "Import from Point Cloud",
        { "PointCloud" }, true))
    {
        if (auto pc = pointCloud.Get<PointCloud>())
        {
            std::vector<glm::vec3> points;
            points.resize(pc->m_points.size());
            for (int i = 0; i < pc->m_points.size(); i++)
            {
                auto point = pc->m_points[i] + pc->m_offset;
                points[i] = glm::vec3(point.z, point.x, point.y);
            }
            CalculateVolume(points);
            pointCloud.Clear();
        }
    }

    if (!displayLayer && m_displayBounds && m_meshGenerated) {
        for (auto &i: m_boundMeshes) {
            Gizmos::DrawGizmoMesh(
                    i, m_displayColor,
                    scene->GetDataComponent<GlobalTransform>(GetOwner()).m_value);
        }
    }
}

bool RadialBoundingVolume::InVolume(const glm::vec3 &position) {
    if (glm::any(glm::isnan(position)))
        return true;
    if (m_meshGenerated) {
        const auto sliceIndex = SelectSlice(position);
        const float currentDistance =
                glm::length(glm::vec2(position.x, position.z));
        return glm::max(1.0f, m_layers[sliceIndex.x][sliceIndex.y].m_maxDistance) >=
               currentDistance &&
               position.y <= m_maxHeight;
    }
    return true;
}

bool RadialBoundingVolume::InVolume(const GlobalTransform &globalTransform,
                                    const glm::vec3 &position) {
    const auto &finalPos = glm::vec3(
            (glm::inverse(globalTransform.m_value) * glm::translate(position))[3]);

    if (m_meshGenerated) {
        const auto sliceIndex = SelectSlice(finalPos);
        const float currentDistance =
                glm::length(glm::vec2(finalPos.x, finalPos.z));
        return glm::max(1.0f, m_layers[sliceIndex.x][sliceIndex.y].m_maxDistance) >=
               currentDistance &&
               finalPos.y <= m_maxHeight;
    }
    return true;
}

void RadialBoundingVolume::Deserialize(const YAML::Node &in) {
    IVolume::Deserialize(in);
    m_meshGenerated = false;
    m_center = in["m_center"].as<glm::vec3>();
    m_displayColor = in["m_displayColor"].as<glm::vec4>();
    m_pruneBuds = in["m_pruneBuds"].as<bool>();
    m_maxHeight = in["m_maxHeight"].as<float>();
    m_maxRadius = in["m_maxRadius"].as<float>();
    m_displayScale = in["m_displayScale"].as<float>();
    m_layerAmount = in["m_layerAmount"].as<int>();
    m_sectorAmount = in["m_sectorAmount"].as<int>();

    if (in["m_layers"]) {
        m_layers.resize(m_layerAmount);
        for (auto &i: m_layers) {
            i.resize(m_sectorAmount);
        }
        int index = 0;
        for (const auto &i: in["m_layers"]) {
            m_layers[index / m_sectorAmount][index % m_sectorAmount].m_maxDistance =
                    i["m_maxDistance"].as<float>();
            index++;
        }
    }
}

void RadialBoundingVolume::Serialize(YAML::Emitter &out) {
    IVolume::Serialize(out);
    out << YAML::Key << "m_center" << YAML::Value << m_center;
    out << YAML::Key << "m_displayColor" << YAML::Value << m_displayColor;
    out << YAML::Key << "m_pruneBuds" << YAML::Value << m_pruneBuds;
    out << YAML::Key << "m_maxHeight" << YAML::Value << m_maxHeight;
    out << YAML::Key << "m_maxRadius" << YAML::Value << m_maxRadius;
    out << YAML::Key << "m_displayScale" << YAML::Value << m_displayScale;
    out << YAML::Key << "m_layerAmount" << YAML::Value << m_layerAmount;
    out << YAML::Key << "m_sectorAmount" << YAML::Value << m_sectorAmount;

    if (!m_layers.empty()) {
        out << YAML::Key << "m_layers" << YAML::BeginSeq;
        for (const auto &i: m_layers) {
            for (const auto &j: i) {
                out << YAML::BeginMap;
                out << YAML::Key << "m_maxDistance" << YAML::Value << j.m_maxDistance;
                out << YAML::EndMap;
            }
        }
        out << YAML::EndSeq;
    }
}

void RadialBoundingVolume::Augmentation(float value) {
    m_maxHeight += value * 2.0f;
    for (auto &i: m_layers) {
        for (auto &j: i) {
            j.m_maxDistance += value;
        }
    }

}

void RadialBoundingVolume::CalculateSizes() {
    m_sizes.resize(m_layerAmount);
    m_totalSize = 0.0f;
    for (int i = 0; i < m_layerAmount; i++) {
        auto &layer = m_layers[i];
        m_sizes[i].second.resize(layer.size());
        m_sizes[i].first = 0;
        //1. Calculate each each sector's volume
        for (int j = 0; j < layer.size(); j++) {
            auto &sector = layer[j];
            m_sizes[i].second[j] = sector.m_maxDistance * sector.m_maxDistance;
            m_sizes[i].first += sector.m_maxDistance * sector.m_maxDistance;
        }
        m_totalSize += m_sizes[i].first;
    }
}

void RadialBoundingVolume::ResizeVolumes() {
    m_layers.resize(m_layerAmount);
    for (auto &tier: m_layers) {
        tier.resize(m_sectorAmount);
        for (auto &slice: tier) {
            slice.m_maxDistance = 0.0f;
        }
    }
}

void RadialBoundingVolume::OnDestroy() {
    IVolume::OnDestroy();
}
