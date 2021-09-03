#pragma once

#include <CUDABuffer.hpp>
#include <Optix7.hpp>
#include <Vertex.hpp>
#include <cuda.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace RayTracerFacility {
    struct RAY_TRACER_FACILITY_API Camera {
        bool m_modified = false;
        float m_fov = 60;
        /*! camera position - *from* where we are looking */
        glm::vec3 m_from = glm::vec3(0.0f);
        /*! which direction we are looking *at* */
        glm::vec3 m_direction = glm::vec3(0.0f);
        /*! general up-vector */
        glm::vec3 m_horizontal;
        glm::vec3 m_vertical;

        bool operator!=(const Camera &other) const {
            return other.m_fov != this->m_fov || other.m_from != this->m_from ||
                   other.m_direction != this->m_direction ||
                   other.m_horizontal != this->m_horizontal ||
                   other.m_vertical != this->m_vertical;
        }

        void Set(const glm::quat &rotation, const glm::vec3 &position,
                 const float &fov, const glm::ivec2 &size);
    };

#pragma region MyRegion
    enum class EnvironmentalLightingType {
        White,
        EnvironmentalMap,
        CIE
    };

    struct RAY_TRACER_FACILITY_API DefaultRenderingProperties {
        bool m_accumulate = true;
        EnvironmentalLightingType m_environmentalLightingType = EnvironmentalLightingType::White;
        float m_skylightIntensity = 0.8f;
        glm::vec3 m_sunDirection = glm::vec3(0, 1, 0);
        glm::vec3 m_sunColor = glm::vec3(1, 1, 1);
        int m_bounceLimit = 4;
        int m_samplesPerPixel = 1;
        Camera m_camera;
        unsigned m_outputTextureId;
        unsigned m_environmentalMapId;
        glm::ivec2 m_frameSize;

        [[nodiscard]] bool
        Changed(const DefaultRenderingProperties &properties) const {
            return properties.m_accumulate != m_accumulate ||
                   properties.m_environmentalLightingType != m_environmentalLightingType ||
                   properties.m_skylightIntensity != m_skylightIntensity ||
                   properties.m_sunDirection != m_sunDirection ||
                   properties.m_sunColor != m_sunColor ||
                   properties.m_bounceLimit != m_bounceLimit ||
                   properties.m_samplesPerPixel != m_samplesPerPixel ||
                   properties.m_outputTextureId != m_outputTextureId ||
                   properties.m_environmentalMapId != m_environmentalMapId ||
                   properties.m_frameSize != m_frameSize ||
                   properties.m_camera != m_camera;
        }

        void OnGui();
    };

    struct RAY_TRACER_FACILITY_API IlluminationEstimationProperties {
        [[nodiscard]] bool
        Changed(const IlluminationEstimationProperties &properties) const {
            return properties.m_seed != m_seed ||
                   properties.m_bounceLimit != m_bounceLimit ||
                   properties.m_numPointSamples != m_numPointSamples ||
                   properties.m_numScatterSamples != m_numScatterSamples ||
                   properties.m_skylightPower != m_skylightPower ||
                   properties.m_pushNormal != m_pushNormal;
        }

        unsigned m_seed = 0;
        int m_bounceLimit = 2;
        int m_numPointSamples = 100;
        int m_numScatterSamples = 10;
        float m_skylightPower = 1.0f;
        bool m_pushNormal = true;
    };

    enum class DefaultRenderingRayType {
        RadianceRayType,
        SampleSpRayType,
        RayTypeCount
    };
    enum class DefaultIlluminationEstimationRayType {
        RadianceRayType,
        RayTypeCount
    };

    struct VertexInfo;

    struct DefaultRenderingLaunchParams {
        DefaultRenderingProperties m_defaultRenderingProperties;
        struct {
            cudaSurfaceObject_t m_outputTexture;
            size_t m_frameId;
        } m_frame;
        struct {
            cudaTextureObject_t m_environmentalMaps[6];
            float m_lightSize = 1.0f;
            glm::vec3 m_direction = glm::vec3(0, -1, 0);
        } m_skylight;
        OptixTraversableHandle m_traversable;
    };

    template<typename T>
    struct RAY_TRACER_FACILITY_API LightSensor {
        glm::vec3 m_surfaceNormal;
        /**
         * \brief The position of the light probe.
         */
        glm::vec3 m_position;
        /**
         * \brief The calculated overall direction where the point received most
         * light.
         */
        glm::vec3 m_direction;
        /**
         * \brief The total energy received at this point.
         */
        T m_energy = 0;
    };

    struct DefaultIlluminationEstimationLaunchParams {
        size_t m_size;
        IlluminationEstimationProperties m_defaultIlluminationEstimationProperties;
        LightSensor<float> *m_lightProbes;
        OptixTraversableHandle m_traversable;
    };

#pragma endregion
    struct RAY_TRACER_FACILITY_API RayTracerInstance {
        std::vector<Vertex> *m_vertices;
        std::vector<glm::uvec3> *m_triangles;
        size_t m_version;
        size_t m_entityId = 0;
        size_t m_entityVersion = 0;
        glm::vec3 m_surfaceColor;
        float m_roughness;
        float m_metallic;
        bool m_removeTag = false;
        glm::mat4 m_globalTransform;

        unsigned m_albedoTexture = 0;
        unsigned m_normalTexture = 0;
        float m_diffuseIntensity = 0;

        bool m_verticesUpdateFlag = true;
        bool m_transformUpdateFlag = true;
    };

    struct RAY_TRACER_FACILITY_API SkinnedRayTracerInstance {
        std::vector<SkinnedVertex> *m_skinnedVertices;
        std::vector<glm::uvec3> *m_triangles;
        std::vector<glm::mat4> *m_boneMatrices;

        size_t m_version;
        size_t m_entityId = 0;
        size_t m_entityVersion = 0;
        glm::vec3 m_surfaceColor;
        float m_roughness;
        float m_metallic;
        bool m_removeTag = false;
        glm::mat4 m_globalTransform;

        unsigned m_albedoTexture = 0;
        unsigned m_normalTexture = 0;
        float m_diffuseIntensity = 0;

        bool m_verticesUpdateFlag = true;
        bool m_transformUpdateFlag = true;
    };

    enum PipelineType {
        DefaultRendering,
        IlluminationEstimation,

        PipelineSize
    };

    struct RayTracerPipeline {
        std::string m_launchParamsName;
        OptixModule m_module;
        OptixModuleCompileOptions m_moduleCompileOptions = {};

        OptixPipeline m_pipeline;
        OptixPipelineCompileOptions m_pipelineCompileOptions = {};
        OptixPipelineLinkOptions m_pipelineLinkOptions = {};

        std::vector<OptixProgramGroup> m_rayGenProgramGroups;
        CudaBuffer m_rayGenRecordsBuffer;
        std::vector<OptixProgramGroup> m_missProgramGroups;
        CudaBuffer m_missRecordsBuffer;
        std::vector<OptixProgramGroup> m_hitGroupProgramGroups;
        CudaBuffer m_hitGroupRecordsBuffer;
        OptixShaderBindingTable m_sbt = {};
        CudaBuffer m_launchParamsBuffer;
        bool m_accumulate = true;
        bool m_statusChanged = false;
    };

    class RayTracer {
    public:
        std::vector<RayTracerInstance> m_instances;
        std::vector<SkinnedRayTracerInstance> m_skinnedInstances;

        // ------------------------------------------------------------------
        // internal helper functions
        // ------------------------------------------------------------------
        [[nodiscard]] bool
        RenderDefault(const DefaultRenderingProperties &properties);

        void EstimateIllumination(const size_t &size,
                                  const IlluminationEstimationProperties &properties,
                                  CudaBuffer &lightProbes);

        RayTracer();

        /*! build an acceleration structure for the given triangle mesh */
        void BuildAccelerationStructure();

        /*! constructs the shader binding table */
        void BuildShaderBindingTable(
                std::vector<std::pair<unsigned, cudaTextureObject_t>> &boundTextures,
                std::vector<cudaGraphicsResource_t> &boundResources);

        void SetAccumulate(const bool &value);

        void SetSkylightSize(const float &value);

        void SetSkylightDir(const glm::vec3 &value);

        void ClearAccumulate();

    protected:
#pragma region Device and context
        /*! @{ CUDA device context and stream that optix pipeline will run
                on, as well as device properties for this device */
        CUcontext m_cudaContext;
        CUstream m_stream;
        cudaDeviceProp m_deviceProps;
        /*! @} */
        //! the optix context that our pipeline will run in.
        OptixDeviceContext m_optixDeviceContext;

        /*! creates and configures a optix device context (in this simple
          example, only for the primary GPU device) */
        void CreateContext();

#pragma endregion
#pragma region Pipeline setup

        DefaultRenderingLaunchParams m_defaultRenderingLaunchParams;
        DefaultIlluminationEstimationLaunchParams
                m_defaultIlluminationEstimationLaunchParams;

        RayTracerPipeline m_defaultRenderingPipeline;
        RayTracerPipeline m_defaultIlluminationEstimationPipeline;

        /*! creates the module that contains all the programs we are going
          to use. in this simple example, we use a single module from a
          single .cu file, using a single embedded ptx string */
        void CreateModules();

        /*! does all setup for the rayGen program(s) we are going to use */
        void CreateRayGenPrograms();

        /*! does all setup for the miss program(s) we are going to use */
        void CreateMissPrograms();

        /*! does all setup for the hitGroup program(s) we are going to use */
        void CreateHitGroupPrograms();

        /*! assembles the full pipeline of all programs */
        void AssemblePipelines();

        void CreateRayGenProgram(RayTracerPipeline &targetPipeline,
                                 char entryFunctionName[]) const;

        void CreateModule(RayTracerPipeline &targetPipeline, char ptxCode[],
                          char launchParamsName[]) const;

        void AssemblePipeline(RayTracerPipeline &targetPipeline) const;

#pragma endregion

#pragma region Accleration structure
        /*! check if we have build the acceleration structure. */
        bool m_hasAccelerationStructure = false;

        /*! one buffer per input mesh */
        std::vector<CudaBuffer> m_verticesBuffer;
        std::vector<CudaBuffer> m_transformedPositionsBuffer;
        std::vector<CudaBuffer> m_transformedNormalsBuffer;
        std::vector<CudaBuffer> m_transformedTangentBuffer;
        std::vector<CudaBuffer> m_texCoordBuffer;

        std::vector<CudaBuffer> m_boneMatricesBuffer;

        /*! one buffer per input mesh */
        std::vector<CudaBuffer> m_trianglesBuffer;
        //! buffer that keeps the (final, compacted) acceleration structure
        CudaBuffer m_acceleratedStructuresBuffer;
#pragma endregion
    };

} // namespace RayTracerFacility
