#include <VoxelSpace.hpp>

using namespace PlantArchitect;

Voxel &VoxelSpace::GetVoxel(const glm::vec3 &position) {
    const auto relativePosition = position - m_origin;
    const int x = glm::floor(relativePosition.x / m_diameter);
    const int y = glm::floor(relativePosition.y / m_diameter);
    const int z = glm::floor(relativePosition.z / m_diameter);
    if (x < 0 || y < 0 || z < 0 || x > m_size.x - 1 || y > m_size.y - 1 ||
        z > m_size.z - 1) {
        UNIENGINE_ERROR("VoxelSpace: Out of bound!");
        throw 1;
    }
    return m_layers[x].m_lines[y].m_voxels[z];
}

void VoxelSpace::Reset() {
    m_layers.resize(m_size.x);
    for (auto &layer: m_layers) {
        layer.m_lines.resize(m_size.y);
        for (auto &line: layer.m_lines) {
            line.m_voxels.resize(m_size.z);
            for (auto &voxel: line.m_voxels) {
                voxel.m_mutex = std::make_unique<std::mutex>();
            }
        }
    }
}

void VoxelSpace::Clear() {
    for (int i = 0; i < m_size.x; i++)
        for (int j = 0; j < m_size.y; j++)
            for (int k = 0; k < m_size.z; k++) {
                auto &voxel = m_layers[i][j][k];
                std::lock_guard lock(*voxel.m_mutex);
                voxel.m_positions.clear();
                voxel.m_entities.clear();
            }
}

void VoxelSpace::OnInspect() {
    if(ImGui::Checkbox("Display voxels", &m_display)){
        if(m_display) Freeze();
    }
    if (m_display) {
        if (ImGui::Button("Refresh Voxel"))
            Freeze();
    }
}

void VoxelSpace::Freeze() {
    m_frozenVoxels.clear();
    for (int i = 0; i < m_size.x; i++)
        for (int j = 0; j < m_size.y; j++)
            for (int k = 0; k < m_size.z; k++) {
                const auto &voxel = m_layers[i][j][k];
                std::lock_guard lock(*voxel.m_mutex);
                if (!voxel.m_entities.empty()) {
                    m_frozenVoxels.push_back(
                            glm::translate(m_origin + glm::vec3(m_diameter * (0.5f + i),
                                                                m_diameter * (0.5f + j),
                                                                m_diameter * (0.5f + k))) *
                            glm::scale(glm::vec3(m_diameter / 2.0f)));
                }
            }
}

float VoxelSpace::GetDiameter() const { return m_diameter; }

glm::ivec3 VoxelSpace::GetSize() const { return m_size; }

glm::vec3 VoxelSpace::GetOrigin() const { return m_origin; }

void VoxelSpace::SetDiameter(const float &value) {
    m_diameter = value;
    Reset();
}

void VoxelSpace::SetSize(const glm::ivec3 &value) {
    m_size = value;
    Reset();
}

void VoxelSpace::SetOrigin(const glm::vec3 &value) {
    m_origin = value;
    Reset();
}

void VoxelSpace::Push(const glm::vec3 &position, const Entity &target) {
    auto &voxel = GetVoxel(position);
    std::lock_guard lock(*voxel.m_mutex);
    voxel.m_positions.push_back(position);
    voxel.m_entities.push_back(target);
}

void VoxelSpace::Remove(const glm::vec3 &position, const Entity &target) {
    auto &voxel = GetVoxel(position);
    std::lock_guard lock(*voxel.m_mutex);
    for (int i = 0; i < voxel.m_entities.size(); i++) {
        if (voxel.m_entities[i] == target) {
            voxel.m_positions.erase(voxel.m_positions.begin() + i);
            voxel.m_entities.erase(voxel.m_entities.begin() + i);
            return;
        }
    }
}

bool VoxelSpace::HasVoxel(const glm::vec3 &position) {
    auto &voxel = GetVoxel(position);
    std::lock_guard lock(*voxel.m_mutex);
    return !voxel.m_entities.empty();
}

void
VoxelSpace::ForEachInRange(const glm::vec3 &position, float radius,
                           const std::function<void(const glm::vec3 &position,
                                                    const Entity &entity)> &action) const {
    int span = glm::ceil(radius / m_diameter);
    for (int i = -span; i <= span; i++) {
        for (int j = -span; j <= span; j++) {
            for (int k = -span; k <= span; k++) {
                glm::vec3 tracePosition = position + glm::vec3(i, j, k) * m_diameter;

                const auto relativePosition = tracePosition - m_origin;
                const int x = glm::floor(relativePosition.x / m_diameter);
                const int y = glm::floor(relativePosition.y / m_diameter);
                const int z = glm::floor(relativePosition.z / m_diameter);
                if (x < 0 || y < 0 || z < 0 || x > m_size.x - 1 || y > m_size.y - 1 ||
                    z > m_size.z - 1) {
                    UNIENGINE_ERROR("VoxelSpace: Out of bound!");
                    continue;
                }
                const auto &voxel = m_layers[x].m_lines[y].m_voxels[z];

                for (int p = 0; p < voxel.m_positions.size(); p++) {
                    const glm::vec3 vector = position - voxel.m_positions[p];
                    if (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z >
                        radius * radius)
                        continue;
                    action(voxel.m_positions[p], voxel.m_entities[p]);
                }
            }
        }
    }
}

Voxel &Voxel::operator=(const Voxel &voxel) {
    return *this;
}


Voxel::Voxel(const Voxel &) {

}

Voxel::Voxel() {

}

Voxel::~Voxel() {

}
