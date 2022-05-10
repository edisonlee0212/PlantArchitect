#pragma once

#include <plant_architect_export.h>
#include "Application.hpp"
#include "glm/gtc/noise.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    class VoxelGrid : public IAsset {
    protected:
        bool SaveInternal(const std::filesystem::path &path) override;

        bool LoadInternal(const std::filesystem::path &path) override;
    public:

        void FormMesh(std::vector<Vertex> &vertices, std::vector<glm::uvec3> &triangles);

        void OnInspect() override;
        float m_voxels[32768];

        void OnCreate() override;

        float &GetVoxel(int x, int y, int z);

        [[nodiscard]] glm::vec3 GetCenter(int x, int y, int z) const;

        [[nodiscard]] glm::vec3 GetCenter(unsigned index) const;

        void Clear();

        void FillObstacle(const std::shared_ptr<Scene> &scene);
    };
}