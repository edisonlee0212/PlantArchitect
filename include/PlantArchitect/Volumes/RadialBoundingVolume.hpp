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


    public:
        glm::vec3 m_center;
        glm::vec4 m_displayColor = glm::vec4(0.0f, 0.0f, 1.0f, 0.5f);
        bool m_display = false;
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

        void CalculateVolume();

        void CalculateVolume(float maxHeight);

        bool m_displayPoints = true;
        bool m_displayBounds = true;

        void OnInspect() override;

        bool InVolume(const GlobalTransform &globalTransform, const glm::vec3 &position) override;

        bool InVolume(const glm::vec3 &position) override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
    };
} // namespace PlantFactory
