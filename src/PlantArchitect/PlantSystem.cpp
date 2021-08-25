#include <CUDAModule.hpp>
#include <CubeVolume.hpp>
#include <Joint.hpp>
#include <PhysicsManager.hpp>
#include <PlantSystem.hpp>
#include <RayTracedRenderer.hpp>
#include <RayTracerManager.hpp>
#include <RigidBody.hpp>
#include <TreeSystem.hpp>
#include <Utilities.hpp>
#include <Volume.hpp>

using namespace PlantArchitect;

#pragma region GUI Related
void ResourceParcel::OnGui() const {
  ImGui::Text("%s", ("Nutrient: " + std::to_string(m_nutrient)).c_str());
  ImGui::Text("%s", ("Carbon: " + std::to_string(m_carbon)).c_str());
}

void InternodeData::OnGui() {
  EditorManager::DragAndDropButton(m_thickestChild, "Thickest Child");
  EditorManager::DragAndDropButton(m_plant, "Plant");
  ImGui::Checkbox("Display points", &m_displayPoints);
  ImGui::Checkbox("Display KDop", &m_displayHullMesh);
  if (ImGui::Button("Form mesh")) {
    CalculateQuickHull();
    // m_kDop.FormMesh();
  }
  if (m_displayPoints && !m_points.empty()) {
    ImGui::DragFloat("Points size", &m_pointSize, 0.001f, 0.001f, 0.1f);
    ImGui::ColorEdit4("Point Color", &m_pointColor.x);
    RenderManager::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube,
                                          m_pointColor, m_points,
                                          glm::mat4(1.0f), m_pointSize);
    RenderManager::DrawGizmoMeshInstanced(
        DefaultResources::Primitives::Cube,
        EntityManager::GetSystem<TreeSystem>()->m_internodeDebuggingCamera,
        EditorManager::GetInstance().m_sceneCameraPosition,
        EditorManager::GetInstance().m_sceneCameraRotation, m_pointColor,
        m_points, glm::mat4(1.0f), m_pointSize);
  }
  if (m_displayHullMesh && !m_points.empty()) {
    ImGui::ColorEdit4("KDop Color", &m_hullMeshColor.x);
    ImGui::DragFloat("Line size", &m_lineWidth, 0.1f, 1.0f, 10.0f);

    if (m_hullMesh) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glLineWidth(m_lineWidth);
      RenderManager::DrawGizmoMesh(m_hullMesh, m_hullMeshColor);
      RenderManager::DrawGizmoMesh(
          m_hullMesh,
          EntityManager::GetSystem<TreeSystem>()->m_internodeDebuggingCamera,
          EditorManager::GetInstance().m_sceneCameraPosition,
          EditorManager::GetInstance().m_sceneCameraRotation, m_hullMeshColor);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glLineWidth(1.0f);
    } else {
      FormMesh();
    }
  }
  if (ImGui::TreeNode("Display buds")) {
    for (int i = 0; i < m_buds.size(); i++) {
      ImGui::Text("%s", ("Bud: " + std::to_string(i)).c_str());
      if (ImGui::TreeNode("Info")) {
        ImGui::Text(m_buds[i].m_isApical ? "Type: Apical" : "Type: Lateral");
        ImGui::Text(m_buds[i].m_active ? "Status: Active"
                                       : "Status: Not active");
        ImGui::Text(m_buds[i].m_enoughForGrowth ? "Has enough resource: True"
                                                : "Has enough resource: False");
        ImGui::Text("%s", ("ResourceWeight: " +
                           std::to_string(m_buds[i].m_resourceWeight))
                              .c_str());
        if (ImGui::TreeNode("Current Resource")) {
          m_buds[i].m_currentResource.OnGui();
          ImGui::TreePop();
        }
        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }
}

void KDop::Calculate(const std::vector<glm::mat4> &globalTransforms) {
  m_planes.resize(26);
  int index = 0;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        if (x != 0 || y != 0 || z != 0) {
          m_planes[index].m_a = x;
          m_planes[index].m_b = y;
          m_planes[index].m_c = z;
          m_planes[index].Normalize();
          m_planes[index].m_d = FLT_MAX;
          index++;
        }
      }
    }
  }
  for (auto &i : globalTransforms) {
    index = 0;
    glm::vec3 position = i[3];
    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        for (int z = -1; z <= 1; z++) {
          if (x != 0 || y != 0 || z != 0) {
            float distance = -glm::dot(
                position, glm::vec3(m_planes[index].m_a, m_planes[index].m_b,
                                    m_planes[index].m_c));
            if (m_planes[index].m_d > distance) {
              m_planes[index].m_d = distance;
            }
            index++;
          }
        }
      }
    }
  }
}
glm::vec3 KDop::GetIntersection(const Plane &p0, const Plane &p1,
                                const Plane &p2) {
  const auto p0n = glm::vec3(p0.m_a, p0.m_b, p0.m_c);
  const auto p1n = glm::vec3(p1.m_a, p1.m_b, p1.m_c);
  const auto p2n = glm::vec3(p2.m_a, p2.m_b, p2.m_c);
  float det = glm::determinant(glm::mat3(p0n, p1n, p2n));
  assert(det != 0);
  return (glm::cross(p1n, p2n) * -p0.m_d + glm::cross(p2n, p0n) * -p1.m_d +
          glm::cross(p0n, p1n) * -p2.m_d) /
         det;
}
void KDop::Serialize(YAML::Emitter &out) {
  if (!m_planes.empty()) {
    out << YAML::Key << "m_planes" << YAML::Value
        << YAML::Binary((const unsigned char *)m_planes.data(),
                        m_planes.size() * sizeof(Plane));
  }
}
void KDop::Deserialize(const YAML::Node &in) {
  if (in["m_planes"]) {
    YAML::Binary chains = in["m_planes"].as<YAML::Binary>();
    m_planes.resize(chains.size() / sizeof(Plane));
    std::memcpy(m_planes.data(), chains.data(), chains.size());
  }
}
/*
void KDop::FormMesh() {
  m_mesh = AssetManager::CreateAsset<Mesh>();
  m_vertices.clear();
  m_indices.clear();
  Vertex archetype;

  int offset = 0;
  //(-1, 0, 0)
  archetype.m_normal = glm::vec3(-1, 0, 0);
  archetype.m_position = GetIntersection(m_planes[4], m_planes[0], m_planes[1]);
  m_vertices.push_back(archetype);
  archetype.m_position = GetIntersection(m_planes[4], m_planes[1], m_planes[2]);
  m_vertices.push_back(archetype);
  archetype.m_position = GetIntersection(m_planes[4], m_planes[2], m_planes[5]);
  m_vertices.push_back(archetype);
  archetype.m_position = GetIntersection(m_planes[4], m_planes[5], m_planes[8]);
  m_vertices.push_back(archetype);
  archetype.m_position = GetIntersection(m_planes[4], m_planes[8], m_planes[7]);
  m_vertices.push_back(archetype);
  archetype.m_position = GetIntersection(m_planes[4], m_planes[7], m_planes[6]);
  m_vertices.push_back(archetype);
  archetype.m_position = GetIntersection(m_planes[4], m_planes[6], m_planes[3]);
  m_vertices.push_back(archetype);
  archetype.m_position = GetIntersection(m_planes[4], m_planes[3], m_planes[0]);
  m_vertices.push_back(archetype);
  m_indices.emplace_back(0 + offset, 1 + offset, 2 + offset);
  m_indices.emplace_back(0 + offset, 2 + offset, 3 + offset);
  m_indices.emplace_back(0 + offset, 3 + offset, 4 + offset);
  m_indices.emplace_back(0 + offset, 4 + offset, 5 + offset);
  m_indices.emplace_back(0 + offset, 5 + offset, 6 + offset);
  m_indices.emplace_back(0 + offset, 6 + offset, 7 + offset);


  //(1, 0, 0)
  offset += 8;
  archetype.m_normal = glm::vec3(1, 0, 0);
  archetype.m_position =
      GetIntersection(m_planes[21], m_planes[25], m_planes[22]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[21], m_planes[22], m_planes[19]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[21], m_planes[19], m_planes[18]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[21], m_planes[18], m_planes[17]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[21], m_planes[17], m_planes[20]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[21], m_planes[20], m_planes[23]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[21], m_planes[23], m_planes[24]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[21], m_planes[24], m_planes[25]);
  m_vertices.push_back(archetype);
  m_indices.emplace_back(0 + offset, 1 + offset, 2 + offset);
  m_indices.emplace_back(0 + offset, 2 + offset, 3 + offset);
  m_indices.emplace_back(0 + offset, 3 + offset, 4 + offset);
  m_indices.emplace_back(0 + offset, 4 + offset, 5 + offset);
  m_indices.emplace_back(0 + offset, 5 + offset, 6 + offset);
  m_indices.emplace_back(0 + offset, 6 + offset, 7 + offset);

  //(0, -1, 0)
  offset += 8;
  archetype.m_normal = glm::vec3(0, -1, 0);
  archetype.m_position =
      GetIntersection(m_planes[10], m_planes[0], m_planes[9]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[10], m_planes[9], m_planes[17]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[10], m_planes[17], m_planes[18]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[10], m_planes[18], m_planes[19]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[10], m_planes[19], m_planes[11]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[10], m_planes[11], m_planes[2]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[10], m_planes[2], m_planes[1]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[10], m_planes[1], m_planes[0]);
  m_vertices.push_back(archetype);
  m_indices.emplace_back(0 + offset, 1 + offset, 2 + offset);
  m_indices.emplace_back(0 + offset, 2 + offset, 3 + offset);
  m_indices.emplace_back(0 + offset, 3 + offset, 4 + offset);
  m_indices.emplace_back(0 + offset, 4 + offset, 5 + offset);
  m_indices.emplace_back(0 + offset, 5 + offset, 6 + offset);
  m_indices.emplace_back(0 + offset, 6 + offset, 7 + offset);

  //(0, 1, 0)
  offset += 8;
  archetype.m_normal = glm::vec3(0, 1, 0);
  archetype.m_position =
      GetIntersection(m_planes[15], m_planes[8], m_planes[16]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[15], m_planes[16], m_planes[25]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[15], m_planes[25], m_planes[24]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[15], m_planes[24], m_planes[23]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[15], m_planes[23], m_planes[14]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[15], m_planes[14], m_planes[6]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[15], m_planes[6], m_planes[7]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[15], m_planes[7], m_planes[8]);
  m_vertices.push_back(archetype);
  m_indices.emplace_back(0 + offset, 1 + offset, 2 + offset);
  m_indices.emplace_back(0 + offset, 2 + offset, 3 + offset);
  m_indices.emplace_back(0 + offset, 3 + offset, 4 + offset);
  m_indices.emplace_back(0 + offset, 4 + offset, 5 + offset);
  m_indices.emplace_back(0 + offset, 5 + offset, 6 + offset);
  m_indices.emplace_back(0 + offset, 6 + offset, 7 + offset);

  //(0, 0, -1)
  offset += 8;
  archetype.m_normal = glm::vec3(0, 0, -1);
  archetype.m_position =
      GetIntersection(m_planes[12], m_planes[0], m_planes[3]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[12], m_planes[3], m_planes[6]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[12], m_planes[6], m_planes[14]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[12], m_planes[14], m_planes[23]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[12], m_planes[23], m_planes[20]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[12], m_planes[20], m_planes[17]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[12], m_planes[17], m_planes[9]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[12], m_planes[9], m_planes[0]);
  m_vertices.push_back(archetype);
  m_indices.emplace_back(0 + offset, 1 + offset, 2 + offset);
  m_indices.emplace_back(0 + offset, 2 + offset, 3 + offset);
  m_indices.emplace_back(0 + offset, 3 + offset, 4 + offset);
  m_indices.emplace_back(0 + offset, 4 + offset, 5 + offset);
  m_indices.emplace_back(0 + offset, 5 + offset, 6 + offset);
  m_indices.emplace_back(0 + offset, 6 + offset, 7 + offset);

  //(0, 0, 1)
  offset += 8;
  archetype.m_normal = glm::vec3(0, 0, 1);
  archetype.m_position =
      GetIntersection(m_planes[13], m_planes[8], m_planes[5]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[13], m_planes[5], m_planes[2]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[13], m_planes[2], m_planes[11]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[13], m_planes[11], m_planes[19]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[13], m_planes[19], m_planes[22]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[13], m_planes[22], m_planes[25]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[13], m_planes[25], m_planes[16]);
  m_vertices.push_back(archetype);
  archetype.m_position =
      GetIntersection(m_planes[13], m_planes[16], m_planes[8]);
  m_vertices.push_back(archetype);
  m_indices.emplace_back(0 + offset, 1 + offset, 2 + offset);
  m_indices.emplace_back(0 + offset, 2 + offset, 3 + offset);
  m_indices.emplace_back(0 + offset, 3 + offset, 4 + offset);
  m_indices.emplace_back(0 + offset, 4 + offset, 5 + offset);
  m_indices.emplace_back(0 + offset, 5 + offset, 6 + offset);
  m_indices.emplace_back(0 + offset, 6 + offset, 7 + offset);

  m_mesh->SetVertices((unsigned)VertexAttribute::Position |
                          (unsigned)VertexAttribute::Normal |
                          (unsigned)VertexAttribute::TexCoord,
                      m_vertices, m_indices);
}
*/
void InternodeData::CalculateKDop() { m_kDop.Calculate(m_points); }
void InternodeData::CalculateQuickHull() {
  std::vector<quickhull::Vector3<float>> pointCloud;
  for (const auto &i : m_points) {
    pointCloud.emplace_back(i[3].x, i[3].y, i[3].z);
  }
  quickhull::QuickHull<float> qh;
  m_convexHull = qh.getConvexHull(pointCloud, true, false);
}
void InternodeData::FormMesh() {
  if (!m_hullMesh)
    m_hullMesh = AssetManager::CreateAsset<Mesh>();
  const auto &indexBuffer = m_convexHull.getIndexBuffer();
  const auto &vertexBuffer = m_convexHull.getVertexBuffer();
  std::vector<Vertex> vertices;
  std::vector<unsigned> indices;
  for (const auto &i : vertexBuffer) {
    Vertex vertex;
    vertex.m_position = glm::vec3(i.x, i.y, i.z);
    vertices.push_back(vertex);
  }
  for (const auto &i : indexBuffer) {
    indices.push_back(i);
  }
  m_hullMesh->SetVertices((unsigned)VertexAttribute::Position |
                              (unsigned)VertexAttribute::TexCoord,
                          vertices, indices);
}
void InternodeData::Clone(const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<InternodeData>(target);
  m_hullMesh.reset();
}
void InternodeData::Relink(const std::unordered_map<Handle, Handle> &map) {
  m_thickestChild.Relink(map);
  m_plant.Relink(map);
}
void InternodeData::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_normalDir" << YAML::Value << m_normalDir;
  out << YAML::Key << "m_displayPoints" << YAML::Value << m_displayPoints;
  out << YAML::Key << "m_displayHullMesh" << YAML::Value << m_displayHullMesh;
  out << YAML::Key << "m_pointSize" << YAML::Value << m_pointSize;
  out << YAML::Key << "m_lineWidth" << YAML::Value << m_lineWidth;
  out << YAML::Key << "m_pointColor" << YAML::Value << m_pointColor;
  out << YAML::Key << "m_hullMeshColor" << YAML::Value << m_hullMeshColor;
  out << YAML::Key << "m_step" << YAML::Value << m_step;

  if (!m_points.empty()) {
    out << YAML::Key << "m_points" << YAML::Value
        << YAML::Binary((const unsigned char *)m_points.data(),
                        m_points.size() * sizeof(glm::mat4));
  }

  m_thickestChild.Save("m_thickestChild", out);
  m_plant.Save("m_plant", out);

  if (!m_buds.empty()) {
    out << YAML::Key << "m_buds" << YAML::Value << YAML::BeginSeq;
    for (auto &i : m_buds) {
      out << YAML::BeginMap;
      i.Serialize(out);
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }

  out << YAML::Key << "m_kDop" << YAML::BeginMap;
  m_kDop.Serialize(out);
  out << YAML::EndMap;
}
void InternodeData::Deserialize(const YAML::Node &in) {
  m_normalDir = in["m_normalDir"].as<glm::vec3>();
  m_displayPoints = in["m_displayPoints"].as<bool>();
  m_displayHullMesh = in["m_displayHullMesh"].as<bool>();
  m_pointSize = in["m_pointSize"].as<float>();
  m_lineWidth = in["m_lineWidth"].as<float>();
  m_pointColor = in["m_pointColor"].as<glm::vec4>();
  m_hullMeshColor = in["m_hullMeshColor"].as<glm::vec4>();

  if (in["m_points"]) {
    YAML::Binary chains = in["m_points"].as<YAML::Binary>();
    m_points.resize(chains.size() / sizeof(glm::mat4));
    std::memcpy(m_points.data(), chains.data(), chains.size());
  }

  m_thickestChild.Load("m_thickestChild", in);
  m_plant.Load("m_plant", in);

  if (in["m_buds"]) {
    for (const auto &i : in["m_buds"]) {
      Bud bud;
      bud.Deserialize(i);
      m_buds.push_back(bud);
    }
  }

  m_kDop.Deserialize(in["m_kDop"]);
}

void PlantSystem::OnInspect() {
  if(!m_ready){
    ImGui::Text("System not ready!");
    return;
  }

  if (m_iterationsToGrow == 0 && m_physicsSimulationRemainingTime == 0) {
    if (ImGui::Button("Delete all plants")) {
      ImGui::OpenPopup("Delete Warning");
    }
    if (ImGui::BeginPopupModal("Delete Warning", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text("Are you sure? All plants will be removed!");
      if (ImGui::Button("Yes, delete all!", ImVec2(120, 0))) {
        DeleteAllPlants();
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(120, 0))) {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
    ImGui::Text(
        "%s",
        ("Internode amount: " + std::to_string(m_internodes.size())).c_str());
    if (ImGui::CollapsingHeader("Growth", ImGuiTreeNodeFlags_DefaultOpen)) {
      static int pushAmount = 5;
      ImGui::DragInt("Amount", &pushAmount, 1, 0, 120.0f / m_deltaTime);
      if (ImGui::Button("Push and start (grow by iteration)")) {
        m_iterationsToGrow = pushAmount;
        Application::SetPlaying(true);
      }
      if (Application::IsPlaying() &&
          ImGui::Button("Push time (grow instantly)")) {
        const float time = Application::Time().CurrentTime();
        GrowAllPlants(pushAmount);
        const std::string spendTime =
            std::to_string(Application::Time().CurrentTime() - time);
        Debug::Log("Growth finished in " + spendTime + " sec.");
      }

      ImGui::SliderFloat("Time speed", &m_deltaTime, 0.1f, 1.0f);

      if (ImGui::TreeNode("Timers")) {
        ImGui::Text("Mesh Gen: %.3fs", m_meshGenerationTimer);
        ImGui::Text("Resource allocation: %.3fs", m_resourceAllocationTimer);
        ImGui::Text("Form internodes: %.3fs", m_internodeFormTimer);
        ImGui::Text("Create internodes: %.3fs", m_internodeCreateTimer);
        ImGui::Text("Create internodes PostProcessing: %.3fs",
                    m_internodeCreatePostProcessTimer);
        ImGui::Text("Illumination: %.3fs", m_illuminationCalculationTimer);
        ImGui::Text("Pruning: %.3fs", m_pruningTimer);
        ImGui::Text("Metadata: %.3fs", m_metaDataTimer);
        ImGui::TreePop();
      }
    }
    if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
      static float pushPhysicsTime = 20.0f;
      ImGui::DragFloat("Time step", &m_physicsTimeStep, 0.001f, 0.01f, 0.1f);
      ImGui::DragFloat("Time", &pushPhysicsTime, 1.0f, 0.0f, 3600.0f);
      if (ImGui::Button("Add time and start")) {
        m_physicsSimulationRemainingTime = m_physicsSimulationTotalTime =
            pushPhysicsTime;
        for (auto &i : m_plantSkinnedMeshGenerators) {
          i.second();
        }
        Application::SetPlaying(true);
        PhysicsManager::UploadRigidBodyShapes();
        PhysicsManager::UploadTransforms(true, true);
        EntityManager::ForEach<GlobalTransform>(
            JobManager::PrimaryWorkers(), m_internodeQuery,
            [&](int index, Entity internode, GlobalTransform &globalTransform) {
              auto internodeData =
                  internode.GetOrSetPrivateComponent<InternodeData>().lock();
              internodeData->m_points.clear();
            },
            false);
      }
    }
  } else {
    ImGui::Text("Busy...");
  }
}
#pragma endregion
#pragma region Growth related
bool PlantSystem::GrowAllPlants() {
  Refresh();
  m_globalTime += m_deltaTime;
  float time = Application::Time().CurrentTime();
  std::vector<ResourceParcel> totalResources;
  totalResources.resize(m_plants.size());
  std::vector<ResourceParcel> resourceAvailable;
  resourceAvailable.resize(m_plants.size());
  for (auto &i : resourceAvailable)
    i.m_nutrient = 5000000.0f;
  CollectNutrient(m_plants, totalResources, resourceAvailable);
  for (auto &i : m_plantResourceAllocators) {
    i.second(totalResources);
  }
  m_resourceAllocationTimer = Application::Time().CurrentTime() - time;
  for (const auto &plant : m_plants) {
    auto plantInfo = plant.GetDataComponent<PlantInfo>();
    plantInfo.m_age += m_deltaTime;
    plant.SetDataComponent(plantInfo);
  }
  time = Application::Time().CurrentTime();
  std::vector<InternodeCandidate> candidates;
  for (auto &i : m_plantGrowthModels) {
    i.second(candidates);
  }
  m_internodeFormTimer = Application::Time().CurrentTime() - time;
  if (GrowCandidates(candidates)) {
    time = Application::Time().CurrentTime();
    std::vector<std::pair<GlobalTransform, Volume *>> obstacles;
    const auto *entities =
        EntityManager::UnsafeGetPrivateComponentOwnersList<CubeVolume>();
    if (entities) {
      for (const auto &entity : *entities) {
        if (!entity.IsEnabled())
          continue;
        auto volume = entity.GetOrSetPrivateComponent<CubeVolume>().lock();
        if (volume->IsEnabled() && volume->m_asObstacle)
          obstacles.emplace_back(
              volume->GetOwner().GetDataComponent<GlobalTransform>(),
              volume.get());
      }
    }
    for (auto &i : m_plantInternodePruners) {
      i.second(obstacles);
    }
    m_pruningTimer = Application::Time().CurrentTime() - time;
    time = Application::Time().CurrentTime();
    for (auto &i : m_plantMetaDataCalculators) {
      i.second();
    }
    m_metaDataTimer = Application::Time().CurrentTime() - time;
    return true;
  }
  return false;
}
bool PlantSystem::GrowAllPlants(const unsigned &iterations) {
  bool grew = false;
  for (unsigned i = 0; i < iterations; i++) {
    const bool grewInThisIteration = GrowAllPlants();
    grew = grew || grewInThisIteration;
  }
  if (grew)
    Refresh();
  return grew;
}

bool PlantSystem::GrowCandidates(std::vector<InternodeCandidate> &candidates) {
  float time = Application::Time().CurrentTime();
  if (candidates.empty())
    return false;
  auto entities = EntityManager::CreateEntities(m_internodeArchetype,
                                                candidates.size(), "Internode");
  int i = 0;
  for (auto &candidate : candidates) {
    auto newInternode = entities[i];
    newInternode.SetDataComponent(candidate.m_info);
    newInternode.SetDataComponent(candidate.m_growth);
    newInternode.SetDataComponent(candidate.m_statistics);
    newInternode.SetDataComponent(candidate.m_globalTransform);
    i++;
  }
  m_internodeCreateTimer = Application::Time().CurrentTime() - time;
  time = Application::Time().CurrentTime();
  i = 0;
  for (auto &candidate : candidates) {
    auto newInternode = entities[i];
    auto newInternodeData =
        newInternode.GetOrSetPrivateComponent<InternodeData>().lock();
    newInternodeData->m_buds = candidate.m_buds;
    newInternodeData->m_plant = candidate.m_plant;
    newInternode.SetParent(candidate.m_parent);
    const auto search =
        m_plantInternodePostProcessors.find(candidate.m_info.m_plantType);
    if (search != m_plantInternodePostProcessors.end()) {
      search->second(newInternode, candidate);
    }
    i++;
  }
  m_internodeCreatePostProcessTimer = Application::Time().CurrentTime() - time;
  return true;
}

void PlantSystem::CalculateIlluminationForInternodes() {
  if (m_internodeTransforms.empty())
    return;
  const float time = Application::Time().CurrentTime();
  // Upload geometries to OptiX.
  RayTracerFacility::RayTracerManager::GetInstance().UpdateScene();
  RayTracerFacility::IlluminationEstimationProperties properties;
  properties.m_bounceLimit = 1;
  properties.m_numPointSamples = 1000;
  properties.m_numScatterSamples = 1;
  properties.m_seed = glm::linearRand(16384, 32768);
  properties.m_skylightPower = 1.0f;
  properties.m_pushNormal = true;
  std::vector<RayTracerFacility::LightSensor<float>> lightProbes;
  lightProbes.resize(m_internodeQuery.GetEntityAmount());
  EntityManager::ForEach<GlobalTransform>(
      JobManager::PrimaryWorkers(), m_internodeQuery,
      [&](int i, Entity leafEntity, GlobalTransform &globalTransform) {
        lightProbes[i].m_position = globalTransform.GetPosition();
        lightProbes[i].m_surfaceNormal =
            globalTransform.GetRotation() * glm::vec3(0.0f, 0.0f, -1.0f);
      },
      false);
  if (lightProbes.empty())
    return;
  RayTracerFacility::CudaModule::EstimateIlluminationRayTracing(properties,
                                                                lightProbes);

  EntityManager::ForEach<Illumination>(
      JobManager::PrimaryWorkers(), m_internodeQuery,
      [&](int i, Entity leafEntity, Illumination &illumination) {
        const auto &lightProbe = lightProbes[i];
        illumination.m_accumulatedDirection +=
            lightProbe.m_direction * m_deltaTime;
        illumination.m_currentIntensity = lightProbe.m_energy;
      },
      false);

  m_illuminationCalculationTimer = Application::Time().CurrentTime() - time;
}
void PlantSystem::CollectNutrient(
    std::vector<Entity> &trees, std::vector<ResourceParcel> &totalNutrients,
    std::vector<ResourceParcel> &nutrientsAvailable) {
  for (int i = 0; i < trees.size(); i++) {
    totalNutrients[i].m_nutrient = nutrientsAvailable[i].m_nutrient;
  }
}

void PlantSystem::ApplyTropism(const glm::vec3 &targetDir, float tropism,
                               glm::vec3 &front, glm::vec3 &up) {
  const glm::vec3 dir = glm::normalize(targetDir);
  const float dotP = glm::abs(glm::dot(front, dir));
  if (dotP < 0.99f && dotP > -0.99f) {
    const glm::vec3 left = glm::cross(front, dir);
    const float maxAngle = glm::acos(dotP);
    const float rotateAngle = maxAngle * tropism;
    front = glm::normalize(
        glm::rotate(front, glm::min(maxAngle, rotateAngle), left));
    up = glm::normalize(glm::cross(glm::cross(front, up), front));
    // up = glm::normalize(glm::rotate(up, glm::min(maxAngle, rotateAngle),
    // left));
  }
}
#pragma endregion
#pragma region ResourceParcel
ResourceParcel::ResourceParcel() {
  m_nutrient = 0;
  m_carbon = 0;
}

ResourceParcel::ResourceParcel(const float &water, const float &carbon) {
  m_nutrient = water;
  m_carbon = carbon;
}

ResourceParcel &ResourceParcel::operator+=(const ResourceParcel &value) {
  m_nutrient += value.m_nutrient;
  m_carbon += value.m_carbon;
  return *this;
}

bool ResourceParcel::IsEnough() const {
  return m_nutrient > 1.0f && m_carbon > 1.0f;
}
void ResourceParcel::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_nutrient" << YAML::Value << m_nutrient;
  out << YAML::Key << "m_carbon" << YAML::Value << m_carbon;
  out << YAML::Key << "m_globalTime" << YAML::Value << m_globalTime;
}
void ResourceParcel::Deserialize(const YAML::Node &in) {
  m_nutrient = in["m_nutrient"].as<float>();
  m_carbon = in["m_carbon"].as<float>();
  m_globalTime = in["m_globalTime"].as<float>();
}

#pragma endregion
#pragma region Helpers

Entity PlantSystem::CreateCubeObstacle() {
  const auto volumeEntity = EntityManager::CreateEntity("Volume");
  volumeEntity.SetEnabled(false);
  Transform transform;
  transform.SetPosition(glm::vec3(0, 10, 0));
  transform.SetScale(glm::vec3(4, 2, 4));
  GlobalTransform globalTransform;
  globalTransform.m_value = transform.m_value;
  volumeEntity.SetDataComponent(transform);
  volumeEntity.SetDataComponent(globalTransform);

  auto meshRenderer =
      volumeEntity.GetOrSetPrivateComponent<MeshRenderer>().lock();
  meshRenderer->m_mesh = DefaultResources::Primitives::Cube;
  meshRenderer->m_material =
      AssetManager::LoadMaterial(DefaultResources::GLPrograms::StandardProgram);

  auto volume = volumeEntity.GetOrSetPrivateComponent<CubeVolume>().lock();
  volume->ApplyMeshRendererBounds();
  return volumeEntity;
}

void PlantSystem::DeleteAllPlants() {
  m_globalTime = 0;
  std::vector<Entity> trees;
  m_plantQuery.ToEntityArray(trees);
  for (const auto &tree : trees)
    EntityManager::DeleteEntity(tree);
  Refresh();
  for (auto &i : m_deleteAllPlants)
    i.second();
}

Entity PlantSystem::CreatePlant(const PlantType &type,
                                const Transform &transform) {
  const auto entity = EntityManager::CreateEntity(m_plantArchetype);
  entity.SetDataComponent(transform);
  entity.SetName("Tree");
  PlantInfo treeInfo{};
  treeInfo.m_plantType = type;
  treeInfo.m_age = 0;
  treeInfo.m_startTime = m_globalTime;
  entity.SetDataComponent(treeInfo);
#pragma region Set root internode
  const auto rootInternode = EntityManager::CreateEntity(m_internodeArchetype);
  rootInternode.SetName("Internode");
  InternodeInfo internodeInfo;
  internodeInfo.m_plantType = type;
  internodeInfo.m_startAge = 0;
  internodeInfo.m_startGlobalTime = treeInfo.m_startTime;
  InternodeGrowth internodeGrowth;
  internodeGrowth.m_desiredLocalRotation =
      glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));

  Transform internodeTransform;
  internodeTransform.m_value =
      glm::translate(glm::vec3(0.0f)) *
      glm::mat4_cast(internodeGrowth.m_desiredLocalRotation) *
      glm::scale(glm::vec3(1.0f));
  InternodeStatistics internodeStatistics;
  rootInternode.SetDataComponent(internodeInfo);
  rootInternode.SetDataComponent(internodeGrowth);
  rootInternode.SetDataComponent(internodeStatistics);
  rootInternode.SetDataComponent(internodeTransform);

  auto rootInternodeData =
      rootInternode.GetOrSetPrivateComponent<InternodeData>().lock();
  Bud bud;
  bud.m_isApical = true;
  rootInternodeData->m_buds.push_back(bud);
  rootInternodeData->m_plant = entity;
  rootInternode.SetParent(entity);

#pragma endregion
  return entity;
}

Entity PlantSystem::CreateInternode(const PlantType &type,
                                    const Entity &parentEntity) {
  const auto entity = EntityManager::CreateEntity(m_internodeArchetype);
  entity.SetName("Internode");
  InternodeInfo internodeInfo;
  internodeInfo.m_plantType = type;
  auto plant = parentEntity.GetOrSetPrivateComponent<InternodeData>()
                   .lock()
                   ->m_plant.Get();
  internodeInfo.m_startAge = plant.GetDataComponent<PlantInfo>().m_age;
  entity.SetDataComponent(internodeInfo);
  entity.GetOrSetPrivateComponent<InternodeData>().lock()->m_plant = plant;
  entity.SetParent(parentEntity);
  return entity;
}

#pragma endregion
#pragma region Runtime
void PlantSystem::Start() {
  EntityManager::GetSystem<PhysicsSystem>()->Disable();
#pragma region Ground
  if (m_ground.Get().IsNull() && m_anchor.Get().IsNull()) {
    m_ground = EntityManager::CreateEntity("Ground");
    auto ground = m_ground.Get();
    auto meshRenderer = ground.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer->m_mesh = DefaultResources::Primitives::Quad;
    meshRenderer->m_material = AssetManager::LoadMaterial(
        DefaultResources::GLPrograms::StandardProgram);
    auto mat = meshRenderer->m_material.Get<Material>();
    mat->m_name = "Ground mat";
    mat->m_roughness = 1.0f;
    mat->m_metallic = 0.5f;
    mat->m_albedoColor = glm::vec3(1.0f);

    Transform groundTransform;
    groundTransform.SetScale(glm::vec3(500.0f, 1.0f, 500.0f));
    groundTransform.SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    ground.SetDataComponent(groundTransform);

    m_anchor = EntityManager::CreateEntity("Anchor");
    auto anchor = m_anchor.Get();
    Transform anchorTransform;
    anchorTransform.SetScale(glm::vec3(1.0 / 500.0f, 1.0f, 1.0 / 500.0f));
    anchorTransform.SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    anchor.SetDataComponent(anchorTransform);
    anchor.SetParent(ground);

    auto rayTracedRenderer =
        ground.GetOrSetPrivateComponent<RayTracerFacility::RayTracedRenderer>()
            .lock();
    rayTracedRenderer->SyncWithMeshRenderer();

    auto cubeVolume = ground.GetOrSetPrivateComponent<CubeVolume>().lock();
    cubeVolume->m_asObstacle = true;
    cubeVolume->m_minMaxBound.m_max = glm::vec3(1, -1.0f, 1);
    cubeVolume->m_minMaxBound.m_min = glm::vec3(-1, -10.0f, -1);
  }
#pragma endregion
#pragma region Mask material
  /*
  std::string vertShaderCode =
      std::string("#version 460 core\n") +
      *DefaultResources::ShaderIncludes::Uniform + +"\n" +
      FileUtils::LoadFileAsString(AssetManager::GetResourceFolderPath() /
                                  "Shaders/Vertex/Standard.vert");
  std::string fragShaderCode =
      std::string("#version 460 core\n") +
      *DefaultResources::ShaderIncludes::Uniform + "\n" +
      FileUtils::LoadFileAsString(
          std::filesystem::path(PLANT_FACTORY_RESOURCE_FOLDER) /
          "Shaders/Fragment/SemanticBranch.frag");

  auto standardVert =
      std::make_shared<OpenGLUtils::GLShader>(OpenGLUtils::ShaderType::Vertex);
  standardVert->Compile(vertShaderCode);
  auto standardFrag = std::make_shared<OpenGLUtils::GLShader>(
      OpenGLUtils::ShaderType::Fragment);
  standardFrag->Compile(fragShaderCode);
  auto branchProgram = AssetManager::CreateAsset<OpenGLUtils::GLProgram>();
  branchProgram->Link(standardVert, standardFrag);

  vertShaderCode =
      std::string("#version 460 core\n") +
      *DefaultResources::ShaderIncludes::Uniform + +"\n" +
      FileUtils::LoadFileAsString(AssetManager::GetResourceFolderPath() /
                                  "Shaders/Vertex/StandardInstanced.vert");
  fragShaderCode = std::string("#version 460 core\n") +
                   *DefaultResources::ShaderIncludes::Uniform + "\n" +
                   FileUtils::LoadFileAsString(
                       std::filesystem::path(PLANT_FACTORY_RESOURCE_FOLDER) /
                       "Shaders/Fragment/SemanticLeaf.frag");
  standardVert =
      std::make_shared<OpenGLUtils::GLShader>(OpenGLUtils::ShaderType::Vertex);
  standardVert->Compile(vertShaderCode);
  standardFrag = std::make_shared<OpenGLUtils::GLShader>(
      OpenGLUtils::ShaderType::Fragment);
  standardFrag->Compile(fragShaderCode);
  auto leafProgram = AssetManager::CreateAsset<OpenGLUtils::GLProgram>();
  leafProgram->Link(standardVert, standardFrag);
  */
#pragma endregion
#pragma region Entity

  m_internodeArchetype = EntityManager::CreateEntityArchetype(
      "Internode", BranchCylinder(), BranchCylinderWidth(), BranchPointer(),
      BranchColor(), Ray(), Illumination(), InternodeInfo(), InternodeGrowth(),
      InternodeStatistics());

  m_plantArchetype = EntityManager::CreateEntityArchetype("Tree", PlantInfo());

  m_internodeQuery = EntityManager::CreateEntityQuery();

  m_internodeQuery.SetAllFilters(InternodeInfo());

  m_plantQuery = EntityManager::CreateEntityQuery();

  m_plantQuery.SetAllFilters(PlantInfo());
#pragma endregion
#pragma region GUI
  EditorManager::RegisterComponentDataInspector<InternodeStatistics>(
      [](Entity entity, IDataComponent *data, bool isRoot) {
        auto *internodeStatistics = static_cast<InternodeStatistics *>(data);
        ImGui::Text(("MaxChildOrder: " +
                     std::to_string(internodeStatistics->m_maxChildOrder))
                        .c_str());
        ImGui::Text(("MaxChildLevel: " +
                     std::to_string(internodeStatistics->m_maxChildLevel))
                        .c_str());
        ImGui::Text(
            ("ChildrenEndNodeAmount: " +
             std::to_string(internodeStatistics->m_childrenEndNodeAmount))
                .c_str());
        ImGui::Text(("DistanceToBranchEnd: " +
                     std::to_string(internodeStatistics->m_distanceToBranchEnd))
                        .c_str());
        ImGui::Text(
            ("LongestDistanceToAnyEndNode: " +
             std::to_string(internodeStatistics->m_longestDistanceToAnyEndNode))
                .c_str());
        ImGui::Text(("TotalLength: " +
                     std::to_string(internodeStatistics->m_totalLength))
                        .c_str());
        ImGui::Text(
            ("DistanceToBranchStart: " +
             std::to_string(internodeStatistics->m_distanceToBranchStart))
                .c_str());
        ImGui::Checkbox("IsMaxChild: ", &internodeStatistics->m_isMaxChild);
        ImGui::Checkbox("IsEndNode: ", &internodeStatistics->m_isEndNode);
      });

  EditorManager::RegisterComponentDataInspector<
      InternodeGrowth>([](Entity entity, IDataComponent *data, bool isRoot) {
    auto *internodeGrowth = static_cast<InternodeGrowth *>(data);
    ImGui::Text(
        ("Inhibitor: " + std::to_string(internodeGrowth->m_inhibitor)).c_str());
    ImGui::Text(("InhibitorTransmitFactor: " +
                 std::to_string(internodeGrowth->m_inhibitorTransmitFactor))
                    .c_str());
    ImGui::Text(
        ("DistanceToRoot: " + std::to_string(internodeGrowth->m_distanceToRoot))
            .c_str());
    ImGui::Text(
        ("Thickness: " + std::to_string(internodeGrowth->m_thickness)).c_str());
    ImGui::InputFloat("Gravity sagging", &internodeGrowth->m_sagging,
                      ImGuiInputTextFlags_ReadOnly);
    ImGui::InputFloat("Mass of Children", &internodeGrowth->m_MassOfChildren,
                      ImGuiInputTextFlags_ReadOnly);
    ImGui::InputFloat2("Torque",
                       static_cast<float *>(static_cast<void *>(
                           &internodeGrowth->m_childrenTotalTorque)),
                       "%.3f", ImGuiInputTextFlags_ReadOnly);
    ImGui::InputFloat2("Mean position",
                       static_cast<float *>(static_cast<void *>(
                           &internodeGrowth->m_childMeanPosition)),
                       "%.3f", ImGuiInputTextFlags_ReadOnly);
    glm::vec3 desiredAngles =
        glm::degrees(glm::eulerAngles(internodeGrowth->m_desiredLocalRotation));
    ImGui::InputFloat3("Desired Rotation##Internode", &desiredAngles.x, "%.3f",
                       ImGuiInputTextFlags_ReadOnly);
    ImGui::InputFloat3("Desired Position##Internode",
                       &internodeGrowth->m_desiredGlobalPosition.x, "%.3f",
                       ImGuiInputTextFlags_ReadOnly);
  });

  EditorManager::RegisterComponentDataInspector<InternodeInfo>(
      [](Entity entity, IDataComponent *data, bool isRoot) {
        auto *internodeInfo = static_cast<InternodeInfo *>(data);
        ImGui::Checkbox("Activated", &internodeInfo->m_activated);
        ImGui::Text(
            ("StartAge: " + std::to_string(internodeInfo->m_startAge)).c_str());
        ImGui::Text(("StartGlobalTime: " +
                     std::to_string(internodeInfo->m_startGlobalTime))
                        .c_str());
        ImGui::Text(
            ("Order: " + std::to_string(internodeInfo->m_order)).c_str());
        ImGui::Text(
            ("Level: " + std::to_string(internodeInfo->m_level)).c_str());
      });

  EditorManager::RegisterComponentDataInspector<Illumination>(
      [](Entity entity, IDataComponent *data, bool isRoot) {
        auto *illumination = static_cast<Illumination *>(data);
        ImGui::Text(("CurrentIntensity: " +
                     std::to_string(illumination->m_currentIntensity))
                        .c_str());
        ImGui::InputFloat3("Direction", &illumination->m_accumulatedDirection.x,
                           "%.3f", ImGuiInputTextFlags_ReadOnly);
      });

  EditorManager::RegisterComponentDataInspector<PlantInfo>(
      [](Entity entity, IDataComponent *data, bool isRoot) {
        auto *info = static_cast<PlantInfo *>(data);
        ImGui::Text(
            ("Start time: " + std::to_string(info->m_startTime)).c_str());
        ImGui::Text(("Age: " + std::to_string(info->m_age)).c_str());
      });
#pragma endregion

  m_ready = true;
}

void PlantSystem::Update() {
  if (Application::IsPlaying()) {
    if (m_iterationsToGrow > 0) {
      m_iterationsToGrow--;
      m_needUpdateMetadata = true;
      if (GrowAllPlants()) {
        m_endUpdate = true;
      }
    } else if (m_endUpdate) {
      m_needUpdateMetadata = true;
      Refresh();
      m_endUpdate = false;
    } else if (m_physicsSimulationRemainingTime > 0.0f) {
      PhysicsSimulate();
    }
  }
}

void PlantSystem::Refresh() {
  m_plants.resize(0);
  m_plantQuery.ToEntityArray(m_plants);
  m_internodes.resize(0);
  m_internodeTransforms.resize(0);
  m_internodeQuery.ToComponentDataArray(m_internodeTransforms);
  m_internodeQuery.ToEntityArray(m_internodes);
  float time = Application::Time().CurrentTime();
  if (m_needUpdateMetadata) {
    for (auto &i : m_plantMetaDataCalculators) {
      i.second();
    }
  }
  for (auto &i : m_plantMeshGenerators) {
    i.second();
  }
  m_meshGenerationTimer = Application::Time().CurrentTime() - time;
  CalculateIlluminationForInternodes();
}

void PlantSystem::End() {}
void PlantSystem::PhysicsSimulate() {
  EntityManager::ForEach<GlobalTransform>(
      JobManager::PrimaryWorkers(), m_internodeQuery,
      [&](int index, Entity internode, GlobalTransform &globalTransform) {
        auto internodeData =
            internode.GetOrSetPrivateComponent<InternodeData>().lock();
        internodeData->m_points.push_back(globalTransform.m_value);
      },
      false);

  EntityManager::GetSystem<PhysicsSystem>()->Simulate(m_physicsTimeStep);
  m_physicsSimulationRemainingTime -= m_physicsTimeStep;
  if (m_physicsSimulationRemainingTime <= 0) {
    m_physicsSimulationRemainingTime = 0;
    EntityManager::ForEach<GlobalTransform>(
        JobManager::PrimaryWorkers(), m_internodeQuery,
        [&](int index, Entity internode, GlobalTransform &globalTransform) {
          auto internodeData =
              internode.GetOrSetPrivateComponent<InternodeData>().lock();
          internodeData->CalculateKDop();
          internodeData->CalculateQuickHull();
        },
        false);
  }
  float elapsedTime =
      m_physicsSimulationTotalTime - m_physicsSimulationRemainingTime;
  Transform groundTransform = m_ground.Get().GetDataComponent<Transform>();
  groundTransform.SetPosition(glm::vec3(glm::sin(elapsedTime * 6.0f) * 0.2f,
                                        0.0f,
                                        glm::sin(elapsedTime * 6.0f) * 0.2f));
  m_ground.Get().SetDataComponent(groundTransform);
}
void PlantSystem::Relink(const std::unordered_map<Handle, Handle> &map) {
  m_ground.Relink(map);
  m_anchor.Relink(map);
}
void PlantSystem::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_deltaTime" << YAML::Value << m_deltaTime;
  out << YAML::Key << "m_globalTime" << YAML::Value << m_globalTime;
  out << YAML::Key << "m_illuminationFactor" << YAML::Value
      << m_illuminationFactor;
  out << YAML::Key << "m_illuminationAngleFactor" << YAML::Value
      << m_illuminationAngleFactor;
  out << YAML::Key << "m_physicsTimeStep" << YAML::Value << m_physicsTimeStep;
  out << YAML::Key << "m_physicsSimulationTotalTime" << YAML::Value
      << m_physicsSimulationTotalTime;
  out << YAML::Key << "m_physicsSimulationRemainingTime" << YAML::Value
      << m_physicsSimulationRemainingTime;
  out << YAML::Key << "m_iterationsToGrow" << YAML::Value << m_iterationsToGrow;
  m_ground.Save("m_ground", out);
  m_anchor.Save("m_anchor", out);
}
void PlantSystem::Deserialize(const YAML::Node &in) {
  m_deltaTime = in["m_deltaTime"].as<float>();
  m_globalTime = in["m_globalTime"].as<float>();
  m_illuminationFactor = in["m_illuminationFactor"].as<float>();
  m_illuminationAngleFactor = in["m_illuminationAngleFactor"].as<float>();
  m_physicsTimeStep = in["m_physicsTimeStep"].as<float>();
  m_physicsSimulationTotalTime = in["m_physicsSimulationTotalTime"].as<float>();
  m_physicsSimulationRemainingTime =
      in["m_physicsSimulationRemainingTime"].as<float>();
  m_iterationsToGrow = in["m_iterationsToGrow"].as<int>();

  m_ground.Load("m_ground", in);
  m_anchor.Load("m_anchor", in);

  m_ready = false;
}
void PlantSystem::OnCreate() { Enable(); }

#pragma endregion
void Bud::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_enoughForGrowth" << YAML::Value << m_enoughForGrowth;
  out << YAML::Key << "m_resourceWeight" << YAML::Value << m_resourceWeight;
  out << YAML::Key << "m_deathGlobalTime" << YAML::Value << m_deathGlobalTime;
  out << YAML::Key << "m_avoidanceAngle" << YAML::Value << m_avoidanceAngle;
  out << YAML::Key << "m_active" << YAML::Value << m_active;
  out << YAML::Key << "m_isApical" << YAML::Value << m_isApical;
  out << YAML::Key << "m_mainAngle" << YAML::Value << m_mainAngle;

  out << YAML::Key << "m_currentResource" << YAML::BeginMap;
  m_currentResource.Serialize(out);
  out << YAML::EndMap;

  if (!m_resourceLog.empty()) {
    out << YAML::Key << "m_resourceLog" << YAML::Value
        << YAML::Binary((const unsigned char *)m_resourceLog.data(),
                        m_resourceLog.size() * sizeof(ResourceParcel));
  }
}
void Bud::Deserialize(const YAML::Node &in) {
  m_enoughForGrowth = in["m_enoughForGrowth"].as<bool>();
  m_resourceWeight = in["m_resourceWeight"].as<float>();
  m_deathGlobalTime = in["m_deathGlobalTime"].as<float>();
  m_avoidanceAngle = in["m_avoidanceAngle"].as<float>();
  m_active = in["m_active"].as<bool>();
  m_isApical = in["m_isApical"].as<bool>();
  m_mainAngle = in["m_mainAngle"].as<float>();

  m_currentResource.Deserialize(in["m_currentResource"]);

  if (in["m_resourceLog"]) {
    YAML::Binary logs = in["m_resourceLog"].as<YAML::Binary>();
    m_resourceLog.resize(logs.size() / sizeof(ResourceParcel));
    std::memcpy(m_resourceLog.data(), logs.data(), logs.size());
  }
}
