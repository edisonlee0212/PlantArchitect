#pragma once

#include <plant_architect_export.h>

#include <IVolume.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API RadialBoundingVolumeSlice {
        float m_maxDistance;
    };

    class PLANT_ARCHITECT_API RadialBoundingVolume : public IVolume {
        std::vector<std::shared_ptr<Mesh>> m_boundMeshes;
        bool m_meshGenerated = false;

        void CalculateSizes();
    public:
        glm::vec3 m_center;
        glm::vec4 m_displayColor = glm::vec4(0.0f, 0.0f, 1.0f, 0.5f);
        bool m_pruneBuds = false;

        [[nodiscard]] glm::vec3 GetRandomPoint() override;

        [[nodiscard]] glm::ivec2 SelectSlice(glm::vec3 position) const;

        float m_maxHeight = 0.0f;
        float m_maxRadius = 0.0f;

        void GenerateMesh();

        void FormEntity();

        std::string Save();

        void ExportAsObj(const std::string &filename);

        void Load(const std::string &path);

        float m_displayScale = 0.2f;
        int m_layerAmount = 8;
        int m_sectorAmount = 8;
        std::vector<std::vector<RadialBoundingVolumeSlice>> m_layers;
        std::vector<std::pair<float, std::vector<float>>> m_sizes;
        float m_totalSize = 0;
        void CalculateVolume();
        void CalculateVolume(const std::vector<glm::vec3>& points);
        PrivateComponentRef m_rootInternode;
        void CalculateVolume(float maxHeight);

        void OnInspect() override;

        void ResizeVolumes();

        bool InVolume(const GlobalTransform &globalTransform, const glm::vec3 &position) override;

        bool InVolume(const glm::vec3 &position) override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void Augmentation(float value);

        void OnDestroy() override;
    };
} // namespace PlantFactory
