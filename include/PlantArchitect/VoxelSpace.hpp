#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API Voxel {
        std::unique_ptr<std::mutex> m_mutex;
        std::vector<glm::vec3> m_positions;
        std::vector<Entity> m_owners;
        std::vector<Entity> m_internodes;
    };

    struct PLANT_ARCHITECT_API Line {
        std::vector<Voxel> m_voxels;

        Voxel &operator[](int index) { return m_voxels[index]; }
    };

    struct PLANT_ARCHITECT_API Layer {
        std::vector<Line> m_lines;

        Line &operator[](int index) { return m_lines[index]; }
    };

    class PLANT_ARCHITECT_API VoxelSpace {
        float m_diameter = 2.0f;
        glm::ivec3 m_size = {200, 100, 200};
        glm::vec3 m_origin = glm::vec3(-200, -50, -200);
        std::vector<glm::mat4> m_frozenVoxels;

    public:
        Voxel &GetVoxel(const glm::vec3 &position);

        void Reset();

        void Clear();

        bool m_display = false;

        void OnGui();

        void Freeze();

        [[nodiscard]] float GetDiameter() const;

        void SetDiameter(const float &value);

        void SetSize(const glm::ivec3 &value);

        void SetOrigin(const glm::vec3 &value);

        std::vector<Layer> m_layers;

        void Push(const glm::vec3 &position, const Entity &owner,
                  const Entity &internode);

        void Remove(const glm::vec3 &position, const Entity &owner,
                    const Entity &internode);

        bool HasVoxel(const glm::vec3 &position);

        bool HasNeighbor(const glm::vec3 &position, const Entity &internode, const Entity &parent, float radius);

        bool HasNeighborFromDifferentOwner(const glm::vec3 &position,
                                           const Entity &owner, float radius);

        bool HasNeighborFromSameOwner(const glm::vec3 &position, const Entity &owner,
                                      float radius);

        bool HasObstacleConeSameOwner(const float &angle, const glm::vec3 &position,
                                      const glm::vec3 &direction, const Entity &owner,
                                      const Entity &internode, const Entity &parent,
                                      float selfRadius);

        bool RemoveIfHasObstacleInCone(const float &angle, const glm::vec3 &position,
                                       const glm::vec3 &direction,
                                       const Entity &internode, const Entity &parent,
                                       float selfRadius);

    };
} // namespace PlantFactory
