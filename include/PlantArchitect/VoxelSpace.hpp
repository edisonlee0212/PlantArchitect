#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API Voxel {
        std::unique_ptr<std::mutex> m_mutex;
        std::vector<glm::vec3> m_positions;
        std::vector<Entity> m_entities;
        Voxel();
        ~Voxel();
        Voxel &Voxel::operator =(const Voxel & voxel);
        Voxel::Voxel(const PlantArchitect::Voxel &);
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
        glm::ivec3 m_size = {128, 128, 128};
        glm::vec3 m_origin = glm::vec3(-128, -128, -128);

    public:
        std::vector<glm::mat4> m_frozenVoxels;

        Voxel &GetVoxel(const glm::vec3 &position);

        void Reset();

        void Clear();

        bool m_display = false;

        void OnInspect();

        void Freeze();

        [[nodiscard]] float GetDiameter() const;
        [[nodiscard]] glm::ivec3 GetSize() const;
        [[nodiscard]] glm::vec3 GetOrigin() const;

        void SetDiameter(const float &value);

        void SetSize(const glm::ivec3 &value);

        void SetOrigin(const glm::vec3 &value);

        std::vector<Layer> m_layers;

        void Push(const glm::vec3 &position, const Entity &target);

        void Remove(const glm::vec3 &position, const Entity &target);

        bool HasVoxel(const glm::vec3 &position);

        void ForEachInRange(const glm::vec3 &position,
                             float radius, const std::function<void(const glm::vec3& position, const Entity& entity)>& action) const;

    };
} // namespace PlantFactory
