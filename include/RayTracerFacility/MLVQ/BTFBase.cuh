#pragma once

#include <Optix7.hpp>

#include <PDF6D.cuh>

#include <SharedCoordinates.cuh>

#include <glm/glm.hpp>

namespace RayTracerFacility {
    struct BtfBase {
        SharedCoordinates m_tcTemplate;
        PDF6D<glm::vec3> m_pdf6;
        bool m_hdr = false;
        float m_hdrValue = 1.0f;
        bool m_hasData = false;

        __device__ void GetValueDeg(const glm::vec2 &texCoord,
                                    const float &illuminationTheta,
                                    const float &illuminationPhi,
                                    const float &viewTheta, const float &viewPhi,
                                    glm::vec3 &out, const bool &print) const {
            if (!m_hasData) {
                out = glm::vec3(255, 0, 255);
                return;
            }

                if (illuminationTheta > 90.0f || viewTheta > 90.0f) {
                    out = glm::vec3(0.0f);
                    return;
                }
            SharedCoordinates tc(m_tcTemplate);
            // fast version, pre-computation of interpolation values only once
            if (print)
                printf("Sampling from PDF6...");
            m_pdf6.GetValDeg2(texCoord, illuminationTheta, illuminationPhi, viewTheta,
                              viewPhi, out, tc, print);
            if (print)
                printf("Col6[%.2f, %.2f, %.2f]\n", out.x, out.y, out.z);

            if (m_hdr) {
                // we encode the values multiplied by a user coefficient
                // before it is converted to User Color Model
                // Now we have to multiply it back.
                const float multi = 1.0f / m_hdrValue;
                out *= multi;
            }
        }

        int m_materialOrder; //! order of the material processed
        int m_nColor;        //! number of spectral channels in BTF data

        bool m_useCosBeta; //! use cos angles

        float m_mPostScale;

        int m_numOfBeta;
        int m_numOfAlpha;
        int m_numOfTheta;
        int m_numOfPhi;
        float m_stepAlpha;
        float m_stepTheta;
        float m_stepPhi;

        bool m_allMaterialsInOneDatabase; //! if to compress all materials into one
        //! database
        //! if view direction represented directly by UBO measurement quantization
        bool m_use34ViewRepresentation;
        bool m_usePdf2CompactRep; //! If we do not separate colors and luminances for
        //! 2D functions

        int m_materialCount; //! how many materials are stored in the database

        bool Init(const std::string &materialDirectoryPath);
    };
} // namespace RayTracerFacility
