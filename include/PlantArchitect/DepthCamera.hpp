#pragma once

#include <plant_architect_export.h>
#include "Application.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API DepthCamera : public IPrivateComponent, public RenderTarget {
        static std::shared_ptr<OpenGLUtils::GLProgram> m_depthTransferProgram;
        static std::shared_ptr<OpenGLUtils::GLVAO> m_depthTransferVAO;
    public:
        void Render();
        DepthCamera& operator=(const DepthCamera& source);
        bool m_useCameraResolution = true;
        int m_resX = 1;
        int m_resY = 1;
        void Update() override;
        void OnCreate() override;
        std::shared_ptr<Texture2D> m_colorTexture;
        void OnInspect() override;
    };
} // namespace PlantArchitect