#pragma once

#include <plant_architect_export.h>
#include "Application.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API DepthCamera : public IPrivateComponent, public RenderTarget {
        static std::shared_ptr<OpenGLUtils::GLProgram> m_depthTransferProgram;
        static std::shared_ptr<OpenGLUtils::GLVAO> m_depthTransferVAO;
    public:
        bool m_useCameraResolution = true;
        int m_resX = 1;
        int m_resY = 1;
        float m_factor = 1.0f;

        void Update() override;

        void OnCreate() override;

        std::shared_ptr<Texture2D> m_colorTexture;

        void OnInspect() override;

        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
    };
} // namespace PlantArchitect