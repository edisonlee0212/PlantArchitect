#include <PlantSystem.hpp>
#include <VoxelSpace.hpp>

using namespace PlantArchitect;

Voxel &VoxelSpace::GetVoxel(const glm::vec3 &position) {
  const auto relativePosition = position - m_origin;
  const int x = glm::floor(relativePosition.x / m_diameter);
  const int y = glm::floor(relativePosition.y / m_diameter);
  const int z = glm::floor(relativePosition.z / m_diameter);
  if (x < 0 || y < 0 || z < 0 || x > m_size.x - 1 || y > m_size.y - 1 ||
      z > m_size.z - 1) {
    Debug::Error("VoxelSpace: Out of bound!");
    throw 1;
  }
  return m_layers[x].m_lines[y].m_voxels[z];
}

void VoxelSpace::Reset() {
  m_layers.resize(m_size.x);
  for (auto &layer : m_layers) {
    layer.m_lines.resize(m_size.y);
    for (auto &line : layer.m_lines) {
      line.m_voxels.resize(m_size.z);
      for (auto &voxel : line.m_voxels) {
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
        std::lock_guard lock(*voxel.m_mutex.get());
        voxel.m_positions.clear();
        voxel.m_owners.clear();
        voxel.m_internodes.clear();
      }
}

void VoxelSpace::OnGui() {
  ImGui::Checkbox("Display voxels", &m_display);
  if (m_display) {
    if (ImGui::Button("Freeze Voxel"))
      Freeze();
    RenderManager::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube,
                                          glm::vec4(1, 1, 1, 0.5),
                                          m_frozenVoxels);
  }
}

void VoxelSpace::Freeze() {
  m_frozenVoxels.clear();
  for (int i = 0; i < m_size.x; i++)
    for (int j = 0; j < m_size.y; j++)
      for (int k = 0; k < m_size.z; k++) {
        auto &voxel = m_layers[i][j][k];
        std::lock_guard lock(*voxel.m_mutex.get());
        if (!voxel.m_owners.empty()) {
          m_frozenVoxels.push_back(
              glm::translate(m_origin + glm::vec3(m_diameter * (0.5f + i),
                                                  m_diameter * (0.5f + j),
                                                  m_diameter * (0.5f + k))) *
              glm::scale(glm::vec3(m_diameter / 2.0f)));
        }
      }
}

float VoxelSpace::GetDiameter() const { return m_diameter; }

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

void VoxelSpace::Push(const glm::vec3 &position, const Entity &owner,
                      const Entity &internode) {
  auto &voxel = GetVoxel(position);
  std::lock_guard lock(*voxel.m_mutex.get());
  voxel.m_positions.push_back(position);
  voxel.m_owners.push_back(owner);
  voxel.m_internodes.push_back(internode);
}
void VoxelSpace::Remove(const glm::vec3 &position, const Entity &owner,
                        const Entity &internode) {
  auto &voxel = GetVoxel(position);
  std::lock_guard lock(*voxel.m_mutex.get());
  for (int i = 0; i < voxel.m_internodes.size(); i++) {
    if (voxel.m_internodes[i] == internode) {
      voxel.m_positions.erase(voxel.m_positions.begin() + i);
      voxel.m_owners.erase(voxel.m_owners.begin() + i);
      voxel.m_internodes.erase(voxel.m_internodes.begin() + i);
      return;
    }
  }
}
bool VoxelSpace::HasVoxel(const glm::vec3 &position) {
  auto &voxel = GetVoxel(position);
  std::lock_guard lock(*voxel.m_mutex.get());
  return !voxel.m_owners.empty();
}

bool VoxelSpace::HasNeighbor(const glm::vec3 &position, const Entity &internode,
                             const Entity &parent, float radius) {
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        glm::vec3 tracePosition = position + glm::vec3(i, j, k) * m_diameter;
        auto &voxel = GetVoxel(tracePosition);
        for (int p = 0; p < voxel.m_positions.size(); p++) {
          const glm::vec3 vector = position - voxel.m_positions[p];
          if (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z >
              radius * radius)
            continue;
          const Entity compare = voxel.m_internodes[p];
          if (parent != compare && internode != compare) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool VoxelSpace::HasNeighborFromDifferentOwner(const glm::vec3 &position,
                                               const Entity &owner,
                                               float radius) {
  std::vector<Voxel *> checkedVoxel;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        glm::vec3 tracePosition = position + glm::vec3(i, j, k) * m_diameter;
        auto &voxel = GetVoxel(tracePosition);
        bool duplicate = false;
        for (const auto *i : checkedVoxel) {
          if (&voxel == i) {
            duplicate = true;
            break;
          }
        }
        if (duplicate)
          continue;
        checkedVoxel.push_back(&voxel);
        for (int i = 0; i < voxel.m_positions.size(); i++) {
          const glm::vec3 vector = tracePosition - voxel.m_positions[i];
          if (owner != voxel.m_owners[i] &&
              vector.x * vector.x + vector.y * vector.y + vector.z * vector.z <
                  radius * radius)
            return true;
        }
      }
    }
  }
  return false;
}

bool VoxelSpace::HasNeighborFromSameOwner(const glm::vec3 &position,
                                          const Entity &owner, float radius) {
  std::vector<Voxel *> checkedVoxel;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        glm::vec3 tracePosition = position + glm::vec3(i, j, k) * m_diameter;
        auto &voxel = GetVoxel(tracePosition);
        bool duplicate = false;
        for (const auto *i : checkedVoxel) {
          if (&voxel == i) {
            duplicate = true;
            break;
          }
        }
        if (duplicate)
          continue;
        checkedVoxel.push_back(&voxel);
        for (int i = 0; i < voxel.m_positions.size(); i++) {
          const glm::vec3 vector = tracePosition - voxel.m_positions[i];
          if (owner == voxel.m_owners[i] &&
              vector.x * vector.x + vector.y * vector.y + vector.z * vector.z <
                  radius * radius)
            return true;
        }
      }
    }
  }
  return false;
}

bool VoxelSpace::HasObstacleConeSameOwner(
    const float &angle, const glm::vec3 &position, const glm::vec3 &direction,
    const Entity &owner, const Entity &internode, const Entity &parent,
    float selfRadius) {
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        glm::vec3 tracePosition = position  +
                                  glm::vec3(i, j, k) * m_diameter;
        auto &voxel = GetVoxel(tracePosition);
        std::lock_guard lock(*voxel.m_mutex.get());
        const float cosAngle = glm::cos(glm::radians(angle));
        for (int p = 0; p < voxel.m_positions.size(); p++) {
          const float dot =
              glm::dot(glm::normalize(direction),
                       glm::normalize(voxel.m_positions[p] - tracePosition));
          const glm::vec3 vector = tracePosition - voxel.m_positions[p];
          const float distance2 =
              vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
          const Entity compare = voxel.m_internodes[p];
          if (parent != compare && internode != compare &&
              owner == voxel.m_owners[p] &&
              distance2 < selfRadius * selfRadius && dot > cosAngle) {
            if (internode.GetDataComponent<InternodeInfo>().m_order >
                voxel.m_internodes[p].GetDataComponent<InternodeInfo>().m_order)
              return true;
          }
        }
      }
    }
  }
  return false;
}
bool VoxelSpace::RemoveIfHasObstacleInCone(
    const float &angle, const glm::vec3 &position, const glm::vec3 &direction,
    const Entity &internode, const Entity &parent, float selfRadius) {
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        glm::vec3 tracePosition = position +
                                  glm::vec3(i, j, k) * m_diameter;
        auto &voxel = GetVoxel(tracePosition);
        std::lock_guard lock(*voxel.m_mutex.get());
        const float cosAngle = glm::cos(glm::radians(angle));
        for (int p = 0; p < voxel.m_positions.size(); p++) {
          const float dot =
              glm::dot(glm::normalize(direction),
                       glm::normalize(voxel.m_positions[p] - position));
          if (dot < cosAngle)
            continue;
          const glm::vec3 vector = voxel.m_positions[p] - position;
          const float distance2 =
              vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
          if (distance2 > selfRadius * selfRadius)
            continue;
          const Entity compare = voxel.m_internodes[p];
          if (parent != compare && internode != compare) {
            if (internode.GetDataComponent<InternodeInfo>().m_order >
            voxel.m_internodes[p].GetDataComponent<InternodeInfo>().m_order) {
              //voxel.m_positions.erase(voxel.m_positions.begin() + p);
              //voxel.m_owners.erase(voxel.m_owners.begin() + p);
              //voxel.m_internodes.erase(voxel.m_internodes.begin() + p);
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}
