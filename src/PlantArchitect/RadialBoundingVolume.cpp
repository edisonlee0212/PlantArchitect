#include <InternodeSystem.hpp>
#include <RadialBoundingVolume.hpp>
#include <RayTracer.hpp>

using namespace PlantArchitect;
using namespace UniEngine;

glm::vec3 RadialBoundingVolume::GetRandomPoint() {
    if (!m_meshGenerated)
        return glm::vec3(0);
    return glm::vec3(0);
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
        auto mesh = AssetManager::CreateAsset<Mesh>();
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
                            .m_texCoords = glm::vec2((float) levelStep / (totalLevelStep - 1),
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
                            .m_texCoords = glm::vec2((float) levelStep / (totalLevelStep - 1),
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
            vertices[vertices.size() - totalLevelStep + levelStep].m_texCoords =
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
    auto children = GetOwner().GetChildren();
    for (auto &child: children) {
        EntityManager::DeleteEntity(child);
    }
    children.clear();
    for (auto i = 0; i < m_boundMeshes.size(); i++) {
        auto slice = EntityManager::CreateEntity("RBV_" + std::to_string(i));
        auto mmc = slice.GetOrSetPrivateComponent<MeshRenderer>().lock();
        mmc->m_material = AssetManager::LoadMaterial(
                DefaultResources::GLPrograms::StandardProgram);
        mmc->m_forwardRendering = false;
        mmc->m_mesh = m_boundMeshes[i];

        slice.SetParent(GetOwner(), false);
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
    auto owner = GetOwner();
    if(!owner.HasPrivateComponent<Internode>()) {
        UNIENGINE_WARNING("Entity is not an internode!");
        return;
    }
    std::vector<Entity> internodes;
    auto internode = owner.GetOrSetPrivateComponent<Internode>().lock();
    internode->CollectInternodes(internodes);
    m_maxHeight = 0;
    m_maxRadius = 0;
    const auto treeGlobalTransform =
            owner.GetDataComponent<GlobalTransform>().m_value;
    std::vector<glm::vec3> positions;
    for (auto &i: internodes) {
        auto globalTransform = i.GetDataComponent<GlobalTransform>().m_value;
        const glm::vec3 position =
                (glm::inverse(treeGlobalTransform) * globalTransform)[3];
        positions.push_back(position);
        if (position.y > m_maxHeight)
            m_maxHeight = position.y;
        const float radius = glm::length(glm::vec2(position.x, position.z));
        if (radius > m_maxRadius)
            m_maxRadius = radius;
    }

    m_layers.resize(m_layerAmount);
    for (auto &tier: m_layers) {
        tier.resize(m_sectorAmount);
        for (auto &slice: tier) {
            slice.m_maxDistance = 0.0f;
        }
    }
    auto positionIndex = 0;
    for (auto &internode: internodes) {
        const auto internodeGrowth = internode.GetDataComponent<InternodeInfo>();
        auto parentGlobalTransform =
                internode.GetParent().GetDataComponent<GlobalTransform>().m_value;
        const glm::vec3 parentNodePosition =
                (glm::inverse(treeGlobalTransform) * parentGlobalTransform)[3];
        const int segments = 3;
        for (int i = 0; i < segments; i++) {
            const glm::vec3 position =
                    positions[positionIndex] +
                    (parentNodePosition - positions[positionIndex]) * (float) i /
                    (float) segments;
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
}

void RadialBoundingVolume::CalculateVolume(float maxHeight) {
    auto owner = GetOwner();
    if(!owner.HasPrivateComponent<Internode>()) {
        UNIENGINE_WARNING("Entity is not an internode!");
        return;
    }
    std::vector<Entity> internodes;
    auto internode = owner.GetOrSetPrivateComponent<Internode>().lock();
    internode->CollectInternodes(internodes);
    m_maxHeight = maxHeight;
    m_maxRadius = 0;
    const auto treeGlobalTransform =
            owner.GetDataComponent<GlobalTransform>().m_value;
    std::vector<glm::vec3> positions;
    for (auto &i: internodes) {
        auto globalTransform = i.GetDataComponent<GlobalTransform>().m_value;
        const glm::vec3 position =
                (glm::inverse(treeGlobalTransform) * globalTransform)[3];
        positions.push_back(position);
        const float radius = glm::length(glm::vec2(position.x, position.z));
        if (radius > m_maxRadius)
            m_maxRadius = radius;
    }

    m_layers.resize(m_layerAmount);
    for (auto &tier: m_layers) {
        tier.resize(m_sectorAmount);
        for (auto &slice: tier) {
            slice.m_maxDistance = 0.0f;
        }
    }
    const auto threadsAmount = JobManager::PrimaryWorkers().Size();
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
        const auto internodeGrowth = internode.GetDataComponent<InternodeInfo>();
        auto parentGlobalTransform =
                internode.GetParent().GetDataComponent<GlobalTransform>().m_value;
        const glm::vec3 parentNodePosition =
                (glm::inverse(treeGlobalTransform) * parentGlobalTransform)[3];
        const int segments = 3;
        for (int i = 0; i < segments; i++) {
            const glm::vec3 position =
                    positions[positionIndex] +
                    (parentNodePosition - positions[positionIndex]) * (float) i /
                    (float) segments;
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
}

void RadialBoundingVolume::OnGui() {
    if (!m_meshGenerated)
        CalculateVolume();
    ImGui::Checkbox("Prune Buds", &m_pruneBuds);
    ImGui::Checkbox("Display bounds", &m_display);
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

                RenderManager::DrawGizmoMesh(
                        m_boundMeshes[i], m_displayColor,
                        GetOwner().GetDataComponent<GlobalTransform>().m_value);
                displayLayer = true;
                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
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
    if (!displayLayer && m_display && m_meshGenerated) {
        for (auto &i: m_boundMeshes) {
            RenderManager::DrawGizmoMesh(
                    i, m_displayColor,
                    GetOwner().GetDataComponent<GlobalTransform>().m_value);
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

void RadialBoundingVolume::Clone(
        const std::shared_ptr<IPrivateComponent> &target) {
    *this = *std::static_pointer_cast<RadialBoundingVolume>(target);
}

void RadialBoundingVolume::Deserialize(const YAML::Node &in) {
    m_meshGenerated = false;

    m_center = in["m_center"].as<glm::vec3>();
    m_displayColor = in["m_displayColor"].as<glm::vec4>();
    m_display = in["m_display"].as<bool>();
    m_pruneBuds = in["m_pruneBuds"].as<bool>();
    m_maxHeight = in["m_maxHeight"].as<float>();
    m_maxRadius = in["m_maxRadius"].as<float>();
    m_displayScale = in["m_displayScale"].as<float>();
    m_layerAmount = in["m_layerAmount"].as<int>();
    m_sectorAmount = in["m_sectorAmount"].as<int>();
    m_displayPoints = in["m_displayPoints"].as<bool>();
    m_displayBounds = in["m_displayBounds"].as<bool>();

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
    out << YAML::Key << "m_center" << YAML::Value << m_center;
    out << YAML::Key << "m_displayColor" << YAML::Value << m_displayColor;
    out << YAML::Key << "m_display" << YAML::Value << m_display;
    out << YAML::Key << "m_pruneBuds" << YAML::Value << m_pruneBuds;
    out << YAML::Key << "m_maxHeight" << YAML::Value << m_maxHeight;
    out << YAML::Key << "m_maxRadius" << YAML::Value << m_maxRadius;
    out << YAML::Key << "m_displayScale" << YAML::Value << m_displayScale;
    out << YAML::Key << "m_layerAmount" << YAML::Value << m_layerAmount;
    out << YAML::Key << "m_sectorAmount" << YAML::Value << m_sectorAmount;
    out << YAML::Key << "m_displayPoints" << YAML::Value << m_displayPoints;
    out << YAML::Key << "m_displayBounds" << YAML::Value << m_displayBounds;

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


