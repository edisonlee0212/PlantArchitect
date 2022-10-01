//
// Created by lllll on 5/9/2022.
//

#include "VoxelGrid.hpp"
#include "CubeVolume.hpp"
#include "Graphics.hpp"
#include "SphereVolume.hpp"
#include "CylinderVolume.hpp"
#include "MeshVolume.hpp"
using namespace PlantArchitect;


float &VoxelGrid::GetVoxel(int x, int y, int z) {
    return m_voxels[x * 1024 + y * 32 + z];
}

glm::vec3 VoxelGrid::GetCenter(int x, int y, int z) const {
    return glm::vec3(2.0f * x - 31.0f, 2.0f * y + 1.0f, 2.0f * z - 31.0f);
}

glm::vec3 VoxelGrid::GetCenter(unsigned index) const {
    return glm::vec3(2.0f * (index / 1024) - 31.0f, 2.0f * (index % 1024 / 32) + 1.0f, 2.0f * (index % 32) - 31.0f);
}

void VoxelGrid::Clear() {
    std::memset(m_voxels, 0, sizeof(float) * 32768);
}

void VoxelGrid::FillObstacle(const std::shared_ptr<Scene> &scene) {
    Clear();
    std::vector<std::pair<GlobalTransform, std::shared_ptr<IVolume>>> obstacleVolumes;
    auto *cubeObstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<CubeVolume>();
    if (cubeObstaclesEntities)
        for (const auto &i: *cubeObstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<CubeVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<CubeVolume>(i).lock());
                if (volume->m_asObstacle && volume->IsEnabled())
                    obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
            }
        }
    auto *sphereObstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<SphereVolume>();
    if (sphereObstaclesEntities)
        for (const auto &i: *sphereObstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<SphereVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<SphereVolume>(i).lock());
                if (volume->m_asObstacle && volume->IsEnabled())
                    obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
            }
        }
    auto *cylinderObstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<CylinderVolume>();
    if (cylinderObstaclesEntities)
        for (const auto &i: *cylinderObstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<CylinderVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<CylinderVolume>(i).lock());
                if (volume->m_asObstacle && volume->IsEnabled())
                    obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
            }
        }
    auto *meshObstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<MeshVolume>();
    if (meshObstaclesEntities)
        for (const auto &i: *meshObstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<MeshVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<MeshVolume>(i).lock());
                if (volume->m_asObstacle && volume->IsEnabled())
                    obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
            }
        }
    if(!obstacleVolumes.empty()){
        std::vector<std::shared_future<void>> results;
        Jobs::ParallelFor(32768, [&](unsigned i) {
            auto center = GetCenter(i);
            const int div = 4;
            float fillRatio = 0;
            for (int block = 0; block < div * div * div; block++) {
                auto position = center + 2.0f * glm::vec3((float) (block / (div * div)) / div + 0.5f / div - 0.5f,
                                                          (float) (block % (div * div) / div) / div + 0.5f / div - 0.5f,
                                                          (float) (block % div) / div + 0.5f / div - 0.5f);
                for (const auto &volume: obstacleVolumes) {
                    if (volume.second->InVolume(volume.first, position)) {
                        fillRatio += 1.0f / (div * div * div);
                        break;
                    }
                }
            }
            m_voxels[i] = fillRatio;
        }, results);
        for (const auto &i: results) {
            i.wait();
        }
    }
}

bool VoxelGrid::SaveInternal(const std::filesystem::path &path) {
    std::ofstream ofs;
    ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (ofs.is_open()) {
        std::string output;
        for (int i = 0; i < 32767; i++) {
            output += std::to_string(m_voxels[i]) + ",";
        }
        output += std::to_string(m_voxels[32767]);
        ofs.write(output.c_str(), output.size());
        ofs.flush();
        ofs.close();
        return true;
    } else {
        UNIENGINE_ERROR("Can't open file!");
    }
    return false;
}

bool VoxelGrid::LoadInternal(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
        UNIENGINE_ERROR("Not exist!");
        return false;
    }
    try {
        std::ifstream ifs(path.string());
        std::string line;
        std::getline(ifs, line);
        std::stringstream str(line);
        std::string word;
        for (int i = 0; i < 32768; i++) {
            std::getline(str, word, ',');
            m_voxels[i] = std::stof(word);
        }
    }
    catch (std::exception e) {
        UNIENGINE_ERROR("Failed to load!");
        return false;
    }
    return true;
}

void VoxelGrid::OnCreate() {
    Clear();
}

void VoxelGrid::OnInspect() {
    if (ImGui::Button("Refresh obstacles")) {
        FillObstacle(Application::GetActiveScene());
    }
    if(ImGui::Button("Form mesh")){
        auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
        std::vector<Vertex> vertices;
        std::vector<glm::uvec3> triangles;
        FormMesh(vertices, triangles);
        mesh->SetVertices(17, vertices, triangles);
        auto scene = Application::GetActiveScene();
        auto entity = scene->CreateEntity("Voxels");
        auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(entity).lock();
        meshRenderer->m_mesh = mesh;
        auto material = ProjectManager::CreateTemporaryAsset<Material>();
        material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
        meshRenderer->m_material = material;
    }
}

void VoxelGrid::FormMesh(std::vector<Vertex> &vertices, std::vector<glm::uvec3> &triangles) {
    auto cubeMesh = DefaultResources::Primitives::Cube;
    auto &cubeTriangles = cubeMesh->UnsafeGetTriangles();
    auto cubeVerticesSize = cubeMesh->GetVerticesAmount();
    size_t offset = 0;
    int index = 0;
    for (const auto &voxel: m_voxels) {
        if (voxel > 0.0f) {
            auto matrix =
                    glm::translate(GetCenter(index)) * glm::mat4_cast(glm::quat(glm::vec3(0.0f))) * glm::scale(glm::vec3(1.0f));
            Vertex archetype;
            for (auto i = 0; i < cubeMesh->GetVerticesAmount(); i++) {
                archetype.m_position =
                        matrix * glm::vec4(cubeMesh->UnsafeGetVertices()[i].m_position, 1.0f);
                archetype.m_normal = glm::normalize(glm::vec3(
                        matrix * glm::vec4(cubeMesh->UnsafeGetVertices()[i].m_normal, 0.0f)));
                archetype.m_tangent = glm::normalize(glm::vec3(
                        matrix *
                        glm::vec4(cubeMesh->UnsafeGetVertices()[i].m_tangent, 0.0f)));
                archetype.m_texCoord =
                        cubeMesh->UnsafeGetVertices()[i].m_texCoord;
                vertices.push_back(archetype);
            }
            for (auto triangle: cubeTriangles) {
                triangle.x += offset;
                triangle.y += offset;
                triangle.z += offset;
                triangles.emplace_back(triangle);
            }
            offset += cubeVerticesSize;
        }
        index++;
    }
}
