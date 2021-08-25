#include <PlantSystem.hpp>
#include <TreeData.hpp>
#include <TreeSystem.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>

using namespace PlantArchitect;

void TreeData::OnGui() {
  if (ImGui::TreeNodeEx("I/O")) {
    if (m_meshGenerated) {
      FileUtils::SaveFile("Export OBJ", "3D Model", {".obj"},
                          [this](const std::filesystem::path &path) {
                            ExportModel(path.string());
                          });
    }
    FileUtils::SaveFile(
        "Export xml graph", "Graph", {".xml"},
        [this](const std::filesystem::path &path) {
          std::ofstream ofs;
          ofs.open(path.string().c_str(),
                   std::ofstream::out | std::ofstream::trunc);
          if (!ofs.is_open()) {
            Debug::Error("Can't open file!");
            return;
          }
          rapidxml::xml_document<> doc;
          auto *type = doc.allocate_node(rapidxml::node_doctype, 0, "Tree");
          doc.append_node(type);
          auto *scene =
              doc.allocate_node(rapidxml::node_element, "Tree", "Tree");
          doc.append_node(scene);
          TreeSystem::Serialize(GetOwner(), doc, scene);
          ofs << doc;
          ofs.flush();
          ofs.close();
        });
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Runtime Data")) {
    ImGui::Text(
        ("MeshGenerated: " + std::string(m_meshGenerated ? "Yes" : "No"))
            .c_str());
    ImGui::Text(
        ("FoliageGenerated: " + std::string(m_foliageGenerated ? "Yes" : "No"))
            .c_str());
    ImGui::Text(("ActiveLength: " + std::to_string(m_activeLength)).c_str());
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Parameters")) {
    m_parameters.OnGui();
    ImGui::TreePop();
  }
}

void TreeData::ExportModel(const std::string &filename,
                           const bool &includeFoliage) const {
  auto mesh = GetOwner()
                  .GetOrSetPrivateComponent<MeshRenderer>()
                  .lock()
                  ->m_mesh.Get<Mesh>();
  if (!mesh)
    return;
  if (mesh->GetVerticesAmount() == 0) {
    Debug::Log("Mesh not generated!");
    return;
  }
  auto triangles = mesh->UnsafeGetTriangles();
  std::ofstream of;
  of.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
  if (of.is_open()) {
    std::string branchVertices;
    std::string leafVertices;
    std::string branchIndices;
    std::string leafIndices;
#pragma region Data collection
    for (const auto &vertex : mesh->UnsafeGetVertices()) {
      branchVertices += "v " + std::to_string(vertex.m_position.x) + " " +
                        std::to_string(vertex.m_position.y) + " " +
                        std::to_string(vertex.m_position.z) + "\n";
    }
    for (int i = 0; i < triangles.size(); i++) {
      const auto triangle = triangles[i];
      branchIndices += "f " + std::to_string(triangle.x + 1) + " " +
                       std::to_string(triangle.y + 1) + " " +
                       std::to_string(triangle.z + 1) + "\n";
    }
#pragma endregion
    size_t branchVerticesSize = mesh->GetVerticesAmount();
    if (includeFoliage) {
      Entity foliageEntity;
      GetOwner().ForEachChild([&foliageEntity](Entity child) {
        if (child.HasDataComponent<TreeLeavesTag>()) {
          foliageEntity = child;
        }
      });
      size_t branchletVerticesSize = 0;
      if (foliageEntity.HasPrivateComponent<MeshRenderer>()) {
        mesh = foliageEntity.GetOrSetPrivateComponent<MeshRenderer>()
                   .lock()
                   ->m_mesh.Get<Mesh>();
        triangles = mesh->UnsafeGetTriangles();
        branchletVerticesSize += mesh->GetVerticesAmount();
#pragma region Data collection
        for (const auto &vertex : mesh->UnsafeGetVertices()) {
          leafVertices += "v " + std::to_string(vertex.m_position.x) + " " +
                          std::to_string(vertex.m_position.y) + " " +
                          std::to_string(vertex.m_position.z) + "\n";
        }
        for (auto triangle : triangles) {
          leafIndices +=
              "f " + std::to_string(triangle.x + branchVerticesSize + 1) + " " +
              std::to_string(triangle.y + branchVerticesSize + 1) + " " +
              std::to_string(triangle.z + branchVerticesSize + 1) + "\n";
        }
#pragma endregion
      }
    }

    of.write(branchVertices.c_str(), branchVertices.size());
    of.flush();
    if (!leafVertices.empty()) {
      of.write(leafVertices.c_str(), leafVertices.size());
      of.flush();
    }
    std::string group = "o branches\n";
    of.write(group.c_str(), group.size());
    of.write(branchIndices.c_str(), branchIndices.size());
    if (!leafVertices.empty()) {
      group = "o leaves\n";
      of.write(group.c_str(), group.size());
      of.write(leafIndices.c_str(), leafIndices.size());
    }
    of.flush();
    of.close();
    Debug::Log("Model saved as " + filename);
  } else {
    Debug::Error("Can't open file!");
  }
}
void TreeData::Clone(const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<TreeData>(target);
}
void TreeData::OnCreate() {
  m_branchMesh = AssetManager::CreateAsset<Mesh>();
  m_skinnedBranchMesh = AssetManager::CreateAsset<SkinnedMesh>();
  m_meshGenerated = false;
}
void TreeData::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_branchMesh);
  list.push_back(m_skinnedBranchMesh);
}
void TreeData::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_height" << YAML::Value << m_height;
  out << YAML::Key << "m_maxBranchingDepth" << YAML::Value
      << m_maxBranchingDepth;
  out << YAML::Key << "m_lateralBudsCount" << YAML::Value << m_lateralBudsCount;
  out << YAML::Key << "m_totalLength" << YAML::Value << m_totalLength;
  out << YAML::Key << "m_activeLength" << YAML::Value << m_activeLength;
  out << YAML::Key << "m_meshGenerated" << YAML::Value << m_meshGenerated;
  out << YAML::Key << "m_foliageGenerated" << YAML::Value << m_foliageGenerated;
  out << YAML::Key << "m_gravityDirection" << YAML::Value << m_gravityDirection;

  m_branchMesh.Save("m_branchMesh", out);
  m_skinnedBranchMesh.Save("m_skinnedBranchMesh", out);

  out << YAML::Key << "m_parameters" << YAML::BeginMap;
  m_parameters.Serialize(out);
  out << YAML::EndMap;
}
void TreeData::Deserialize(const YAML::Node &in) {
  m_height = in["m_height"].as<float>();
  m_maxBranchingDepth = in["m_maxBranchingDepth"].as<int>();
  m_lateralBudsCount = in["m_lateralBudsCount"].as<int>();
  m_totalLength = in["m_totalLength"].as<float>();
  m_activeLength = in["m_activeLength"].as<float>();
  m_meshGenerated = in["m_meshGenerated"].as<bool>();
  m_foliageGenerated = in["m_foliageGenerated"].as<bool>();
  m_gravityDirection = in["m_gravityDirection"].as<glm::vec3>();

  m_branchMesh.Load("m_branchMesh", in);
  m_skinnedBranchMesh.Load("m_skinnedBranchMesh", in);

  m_parameters.Deserialize(in["m_parameters"]);
}
