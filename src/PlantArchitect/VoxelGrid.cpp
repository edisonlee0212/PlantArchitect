//
// Created by lllll on 5/9/2022.
//

#include "VoxelGrid.hpp"
#include "CubeVolume.hpp"
#include "Graphics.hpp"

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
    std::memset(&m_colors[0], 0, sizeof(float) * 32768 * 4);
}

void VoxelGrid::FillObstacle(const std::shared_ptr<Scene> &scene) {
    Clear();
    auto *obstaclesEntities = scene->UnsafeGetPrivateComponentOwnersList<CubeVolume>();
    std::vector<std::pair<GlobalTransform, std::shared_ptr<IVolume>>> obstacleVolumes;
    if (obstaclesEntities) {
        for (const auto &i: *obstaclesEntities) {
            if (scene->IsEntityEnabled(i) && scene->HasPrivateComponent<CubeVolume>(i)) {
                auto volume = std::dynamic_pointer_cast<IVolume>(scene->GetOrSetPrivateComponent<CubeVolume>(i).lock());
                if (volume->m_asObstacle && volume->IsEnabled())
                    obstacleVolumes.emplace_back(scene->GetDataComponent<GlobalTransform>(i), volume);
            }
        }
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
            m_colors[i] = glm::vec4(1, 1, 1, fillRatio);
        }, results);
        for (const auto &i: results) {
            i.wait();
        }
    }
}

void VoxelGrid::RenderGrid() {
    Graphics::DrawGizmoMeshInstancedColored(DefaultResources::Primitives::Cube,
                                            m_colors,
                                            m_matrices, glm::mat4(1.0f), 1.0f);
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
            if(m_voxels[i] != 0){
                UNIENGINE_LOG("Hit");
            }
            m_colors[i] = glm::vec4(1, 1, 1, m_voxels[i]);
        }
    }
    catch (std::exception e) {
        UNIENGINE_ERROR("Failed to load!");
        return false;
    }
    return true;
}

void VoxelGrid::OnCreate() {
    m_colors.resize(32768);
    m_matrices.resize(32768);
    Clear();
    for (int i = 0; i < 32768; i++) {
        m_matrices[i] =
                glm::translate(GetCenter(i)) * glm::mat4_cast(glm::quat(glm::vec3(0.0f))) * glm::scale(glm::vec3(1.0f));
    }
}

void VoxelGrid::OnInspect() {
    static bool renderGrid = true;
    if (ImGui::Button("Refresh obstacles")) {
        FillObstacle(Application::GetActiveScene());
    }
    ImGui::Checkbox("Render grid", &renderGrid);
    if (renderGrid) {
        RenderGrid();
    }
}
