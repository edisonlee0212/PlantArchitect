#include <RayTracer.hpp>

#include <optix_function_table_definition.h>

#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <glm/gtc/quaternion.hpp>

#include <glm/gtc/random.hpp>

#include <glm/gtc/type_ptr.hpp>


#define GL_TEXTURE_CUBE_MAP 0x8513

#include <cuda_gl_interop.h>
#include <iostream>
#include <RayDataDefinations.hpp>

#include <functional>
#include <filesystem>
#include <imgui.h>

using namespace RayTracerFacility;

void Camera::Set(const glm::quat &rotation, const glm::vec3 &position, const float &fov, const glm::ivec2 &size) {
    m_from = position;
    m_direction = glm::normalize(rotation * glm::vec3(0, 0, -1));
    const float cosFovY = glm::radians(fov * 0.5f);
    const float aspect = static_cast<float>(size.x) / static_cast<float>(size.y);
    m_horizontal
            = cosFovY * aspect * glm::normalize(glm::cross(m_direction, rotation * glm::vec3(0, 1, 0)));
    m_vertical
            = cosFovY * glm::normalize(glm::cross(m_horizontal, m_direction));
}

void DefaultRenderingProperties::OnGui() {
    ImGui::Begin("Ray:Default");
    {
        if (ImGui::BeginChild("CameraRenderer", ImVec2(0, 0), false,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Settings")) {
                    ImGui::Checkbox("Accumulate", &m_accumulate);
                    ImGui::DragInt("bounce limit", &m_bounceLimit, 1, 1, 8);
                    if (ImGui::DragInt("pixel samples", &m_samplesPerPixel, 1, 1, 64)) {
                        m_samplesPerPixel = glm::clamp(m_samplesPerPixel, 1, 128);
                    }
                    ImGui::Checkbox("Use environmental map", &m_useEnvironmentalMap);
                    ImGui::DragFloat("Skylight intensity", &m_skylightIntensity, 0.01f, 0.0f, 5.0f);
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

bool RayTracer::RenderDefault(const DefaultRenderingProperties &properties) {
    if (properties.m_frameSize.x == 0 | properties.m_frameSize.y == 0) return true;
    if (!m_hasAccelerationStructure) return false;
    std::vector<std::pair<unsigned, cudaTextureObject_t>> boundTextures;
    std::vector<cudaGraphicsResource_t> boundResources;
    BuildShaderBindingTable(boundTextures, boundResources);
    if (m_defaultRenderingLaunchParams.m_defaultRenderingProperties.Changed(properties)) {
        m_defaultRenderingLaunchParams.m_defaultRenderingProperties = properties;
        m_defaultRenderingPipeline.m_statusChanged = true;
    }
    if (!m_defaultRenderingPipeline.m_accumulate || m_defaultRenderingPipeline.m_statusChanged) {
        m_defaultRenderingLaunchParams.m_frame.m_frameId = 0;
        m_defaultRenderingPipeline.m_statusChanged = false;
    }
#pragma region Bind texture
    cudaArray_t outputArray;
    cudaGraphicsResource_t outputTexture;
    cudaArray_t environmentalMapPosXArray;
    cudaArray_t environmentalMapNegXArray;
    cudaArray_t environmentalMapPosYArray;
    cudaArray_t environmentalMapNegYArray;
    cudaArray_t environmentalMapPosZArray;
    cudaArray_t environmentalMapNegZArray;
    cudaGraphicsResource_t environmentalMapTexture;
#pragma region Bind output texture as cudaSurface
    CUDA_CHECK(GraphicsGLRegisterImage(&outputTexture,
                                       m_defaultRenderingLaunchParams.m_defaultRenderingProperties.m_outputTextureId,
                                       GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    CUDA_CHECK(GraphicsMapResources(1, &outputTexture, nullptr));
    CUDA_CHECK(GraphicsSubResourceGetMappedArray(&outputArray, outputTexture, 0, 0));
    // Specify surface
    struct cudaResourceDesc cudaResourceDesc;
    memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
    cudaResourceDesc.resType = cudaResourceTypeArray;
    // Create the surface objects
    cudaResourceDesc.res.array.array = outputArray;
    // Create surface object
    CUDA_CHECK(CreateSurfaceObject(&m_defaultRenderingLaunchParams.m_frame.m_outputTexture, &cudaResourceDesc));
#pragma endregion
#pragma region Bind environmental map as cudaTexture
    CUDA_CHECK(GraphicsGLRegisterImage(&environmentalMapTexture,
                                       m_defaultRenderingLaunchParams.m_defaultRenderingProperties.m_environmentalMapId,
                                       GL_TEXTURE_CUBE_MAP, cudaGraphicsRegisterFlagsNone));
    CUDA_CHECK(GraphicsMapResources(1, &environmentalMapTexture, nullptr));
    CUDA_CHECK(GraphicsSubResourceGetMappedArray(&environmentalMapPosXArray, environmentalMapTexture,
                                                 cudaGraphicsCubeFacePositiveX, 0));
    CUDA_CHECK(GraphicsSubResourceGetMappedArray(&environmentalMapNegXArray, environmentalMapTexture,
                                                 cudaGraphicsCubeFaceNegativeX, 0));
    CUDA_CHECK(GraphicsSubResourceGetMappedArray(&environmentalMapPosYArray, environmentalMapTexture,
                                                 cudaGraphicsCubeFacePositiveY, 0));
    CUDA_CHECK(GraphicsSubResourceGetMappedArray(&environmentalMapNegYArray, environmentalMapTexture,
                                                 cudaGraphicsCubeFaceNegativeY, 0));
    CUDA_CHECK(GraphicsSubResourceGetMappedArray(&environmentalMapPosZArray, environmentalMapTexture,
                                                 cudaGraphicsCubeFacePositiveZ, 0));
    CUDA_CHECK(GraphicsSubResourceGetMappedArray(&environmentalMapNegZArray, environmentalMapTexture,
                                                 cudaGraphicsCubeFaceNegativeZ, 0));
    memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
    cudaResourceDesc.resType = cudaResourceTypeArray;
    struct cudaTextureDesc cudaTextureDesc;
    memset(&cudaTextureDesc, 0, sizeof(cudaTextureDesc));
    cudaTextureDesc.addressMode[0] = cudaAddressModeWrap;
    cudaTextureDesc.addressMode[1] = cudaAddressModeWrap;
    cudaTextureDesc.filterMode = cudaFilterModeLinear;
    cudaTextureDesc.readMode = cudaReadModeElementType;
    cudaTextureDesc.normalizedCoords = 1;
    // Create texture object
    cudaResourceDesc.res.array.array = environmentalMapPosXArray;
    CUDA_CHECK(CreateTextureObject(&m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[0], &cudaResourceDesc,
                                   &cudaTextureDesc, nullptr));
    cudaResourceDesc.res.array.array = environmentalMapNegXArray;
    CUDA_CHECK(CreateTextureObject(&m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[1], &cudaResourceDesc,
                                   &cudaTextureDesc, nullptr));
    cudaResourceDesc.res.array.array = environmentalMapPosYArray;
    CUDA_CHECK(CreateTextureObject(&m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[2], &cudaResourceDesc,
                                   &cudaTextureDesc, nullptr));
    cudaResourceDesc.res.array.array = environmentalMapNegYArray;
    CUDA_CHECK(CreateTextureObject(&m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[3], &cudaResourceDesc,
                                   &cudaTextureDesc, nullptr));
    cudaResourceDesc.res.array.array = environmentalMapPosZArray;
    CUDA_CHECK(CreateTextureObject(&m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[4], &cudaResourceDesc,
                                   &cudaTextureDesc, nullptr));
    cudaResourceDesc.res.array.array = environmentalMapNegZArray;
    CUDA_CHECK(CreateTextureObject(&m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[5], &cudaResourceDesc,
                                   &cudaTextureDesc, nullptr));
#pragma endregion
#pragma endregion
#pragma region Upload parameters
    m_defaultRenderingPipeline.m_launchParamsBuffer.Upload(&m_defaultRenderingLaunchParams, 1);
    m_defaultRenderingLaunchParams.m_frame.m_frameId++;
#pragma endregion
#pragma endregion
#pragma region Launch rays from camera
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
            m_defaultRenderingPipeline.m_pipeline, m_stream,
            /*! parameters and SBT */
            m_defaultRenderingPipeline.m_launchParamsBuffer.DevicePointer(),
            m_defaultRenderingPipeline.m_launchParamsBuffer.m_sizeInBytes,
            &m_defaultRenderingPipeline.m_sbt,
            /*! dimensions of the launch: */
            m_defaultRenderingLaunchParams.m_defaultRenderingProperties.m_frameSize.x,
            m_defaultRenderingLaunchParams.m_defaultRenderingProperties.m_frameSize.y,
            1
    ));
#pragma endregion
    CUDA_SYNC_CHECK();
#pragma region Remove texture binding.
    CUDA_CHECK(DestroySurfaceObject(m_defaultRenderingLaunchParams.m_frame.m_outputTexture));
    m_defaultRenderingLaunchParams.m_frame.m_outputTexture = 0;
    CUDA_CHECK(GraphicsUnmapResources(1, &outputTexture, 0));
    CUDA_CHECK(GraphicsUnregisterResource(outputTexture));

    CUDA_CHECK(DestroyTextureObject(m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[0]));
    m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[0] = 0;
    CUDA_CHECK(DestroyTextureObject(m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[1]));
    m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[1] = 0;
    CUDA_CHECK(DestroyTextureObject(m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[2]));
    m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[2] = 0;
    CUDA_CHECK(DestroyTextureObject(m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[3]));
    m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[3] = 0;
    CUDA_CHECK(DestroyTextureObject(m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[4]));
    m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[4] = 0;
    CUDA_CHECK(DestroyTextureObject(m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[5]));
    m_defaultRenderingLaunchParams.m_skylight.m_environmentalMaps[5] = 0;

    CUDA_CHECK(GraphicsUnmapResources(1, &environmentalMapTexture, 0));
    CUDA_CHECK(GraphicsUnregisterResource(environmentalMapTexture));
#pragma endregion

    for (int i = 0; i < boundResources.size(); i++) {
        CUDA_CHECK(DestroySurfaceObject(boundTextures[i].second));
        CUDA_CHECK(GraphicsUnmapResources(1, &boundResources[i], 0));
        CUDA_CHECK(GraphicsUnregisterResource(boundResources[i]));
    }
    return true;
}

void RayTracer::EstimateIllumination(const size_t &size, const IlluminationEstimationProperties &properties,
                                     CudaBuffer &lightProbes) {
    if (!m_hasAccelerationStructure) return;
    std::vector<std::pair<unsigned, cudaTextureObject_t>> boundTextures;
    std::vector<cudaGraphicsResource_t> boundResources;
    BuildShaderBindingTable(boundTextures, boundResources);

#pragma region Upload parameters
    m_defaultIlluminationEstimationLaunchParams.m_size = size;
    m_defaultIlluminationEstimationLaunchParams.m_defaultIlluminationEstimationProperties = properties;
    m_defaultIlluminationEstimationLaunchParams.m_lightProbes = reinterpret_cast<LightSensor<float> *>(lightProbes.DevicePointer());
    m_defaultIlluminationEstimationPipeline.m_launchParamsBuffer.Upload(&m_defaultIlluminationEstimationLaunchParams,
                                                                        1);
#pragma endregion
#pragma endregion
    if (size == 0) {
        std::cout << "Error!" << std::endl;
        return;
    }
#pragma region Launch rays from camera
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
            m_defaultIlluminationEstimationPipeline.m_pipeline, m_stream,
            /*! parameters and SBT */
            m_defaultIlluminationEstimationPipeline.m_launchParamsBuffer.DevicePointer(),
            m_defaultIlluminationEstimationPipeline.m_launchParamsBuffer.m_sizeInBytes,
            &m_defaultIlluminationEstimationPipeline.m_sbt,
            /*! dimensions of the launch: */
            size,
            1,
            1
    ));
#pragma endregion
    CUDA_SYNC_CHECK();
    for (int i = 0; i < boundResources.size(); i++) {
        CUDA_CHECK(DestroySurfaceObject(boundTextures[i].second));
        CUDA_CHECK(GraphicsUnmapResources(1, &boundResources[i], 0));
        CUDA_CHECK(GraphicsUnregisterResource(boundResources[i]));
    }
}

RayTracer::RayTracer() {
    m_defaultRenderingLaunchParams.m_frame.m_frameId = 0;
    //std::cout << "#Optix: creating optix context ..." << std::endl;
    CreateContext();
    //std::cout << "#Optix: setting up module ..." << std::endl;
    CreateModules();
    //std::cout << "#Optix: creating raygen programs ..." << std::endl;
    CreateRayGenPrograms();
    //std::cout << "#Optix: creating miss programs ..." << std::endl;
    CreateMissPrograms();
    //std::cout << "#Optix: creating hitgroup programs ..." << std::endl;
    CreateHitGroupPrograms();
    //std::cout << "#Optix: setting up optix pipeline ..." << std::endl;
    AssemblePipelines();

    std::cout << "#Optix: context, module, pipeline, etc, all set up ..." << std::endl;

}

void RayTracer::SetSkylightSize(const float &value) {
    m_defaultRenderingLaunchParams.m_skylight.m_lightSize = value;
    m_defaultRenderingPipeline.m_statusChanged = true;

}

void RayTracer::SetSkylightDir(const glm::vec3 &value) {
    m_defaultRenderingLaunchParams.m_skylight.m_direction = value;
    m_defaultRenderingPipeline.m_statusChanged = true;
}

void RayTracer::ClearAccumulate() {
    m_defaultRenderingPipeline.m_statusChanged = true;
}

static void context_log_cb(const unsigned int level,
                           const char *tag,
                           const char *message,
                           void *) {
    fprintf(stderr, "[%2d][%12s]: %s\n", static_cast<int>(level), tag, message);
}

void RayTracer::CreateContext() {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(StreamCreate(&m_stream));
    CUDA_CHECK(GetDeviceProperties(&m_deviceProps, deviceID));
    std::cout << "#Optix: running on device: " << m_deviceProps.name << std::endl;
    const CUresult cuRes = cuCtxGetCurrent(&m_cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);
    OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, nullptr, &m_optixDeviceContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                        (m_optixDeviceContext, context_log_cb, nullptr, 4));
}

extern "C" char DEFAULT_RENDERING_PTX[];
extern "C" char ILLUMINATION_ESTIMATION_PTX[];

void RayTracer::CreateModules() {
    CreateModule(m_defaultRenderingPipeline, DEFAULT_RENDERING_PTX, "defaultRenderingLaunchParams");
    CreateModule(m_defaultIlluminationEstimationPipeline, ILLUMINATION_ESTIMATION_PTX,
                 "defaultIlluminationEstimationLaunchParams");
}

void RayTracer::CreateRayGenPrograms() {
    CreateRayGenProgram(m_defaultRenderingPipeline, "__raygen__renderFrame");
    CreateRayGenProgram(m_defaultIlluminationEstimationPipeline, "__raygen__illuminationEstimation");
}

void RayTracer::CreateMissPrograms() {
    {
        m_defaultRenderingPipeline.m_missProgramGroups.resize(static_cast<int>(DefaultRenderingRayType::RayTypeCount));
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = m_defaultRenderingPipeline.m_module;

        // ------------------------------------------------------------------
        // radiance rays
        // ------------------------------------------------------------------
        pgDesc.miss.entryFunctionName = "__miss__radiance";

        OPTIX_CHECK(optixProgramGroupCreate(m_optixDeviceContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeofLog,
                                            &m_defaultRenderingPipeline.m_missProgramGroups[static_cast<int>(DefaultRenderingRayType::RadianceRayType)]
        ));
        if (sizeofLog > 1) std::cout << log << std::endl;
        // ------------------------------------------------------------------
        // BSSRDF Spatial sampler rays
        // ------------------------------------------------------------------
        pgDesc.miss.entryFunctionName = "__miss__sampleSp";
        OPTIX_CHECK(optixProgramGroupCreate(m_optixDeviceContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeofLog,
                                            &m_defaultRenderingPipeline.m_missProgramGroups[static_cast<int>(DefaultRenderingRayType::SampleSpRayType)]
        ));
        if (sizeofLog > 1) std::cout << log << std::endl;
    }
    {
        m_defaultIlluminationEstimationPipeline.m_missProgramGroups.resize(
                static_cast<int>(DefaultIlluminationEstimationRayType::RayTypeCount));
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = m_defaultIlluminationEstimationPipeline.m_module;

        // ------------------------------------------------------------------
        // radiance rays
        // ------------------------------------------------------------------
        pgDesc.miss.entryFunctionName = "__miss__illuminationEstimation";

        OPTIX_CHECK(optixProgramGroupCreate(m_optixDeviceContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeofLog,
                                            &m_defaultIlluminationEstimationPipeline.m_missProgramGroups[static_cast<int>(DefaultIlluminationEstimationRayType::RadianceRayType)]
        ));
        if (sizeofLog > 1) std::cout << log << std::endl;
    }
}

void RayTracer::CreateHitGroupPrograms() {
    {
        m_defaultRenderingPipeline.m_hitGroupProgramGroups.resize(
                static_cast<int>(DefaultRenderingRayType::RayTypeCount));
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = m_defaultRenderingPipeline.m_module;
        pgDesc.hitgroup.moduleAH = m_defaultRenderingPipeline.m_module;
        // -------------------------------------------------------
        // radiance rays
        // -------------------------------------------------------
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
        OPTIX_CHECK(optixProgramGroupCreate(m_optixDeviceContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeofLog,
                                            &m_defaultRenderingPipeline.m_hitGroupProgramGroups[static_cast<int>(DefaultRenderingRayType::RadianceRayType)]
        ));
        if (sizeofLog > 1) std::cout << log << std::endl;

        // -------------------------------------------------------
        // BSSRDF Sampler ray
        // -------------------------------------------------------
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__sampleSp";
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__sampleSp";

        OPTIX_CHECK(optixProgramGroupCreate(m_optixDeviceContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeofLog,
                                            &m_defaultRenderingPipeline.m_hitGroupProgramGroups[static_cast<int>(DefaultRenderingRayType::SampleSpRayType)]
        ));
        if (sizeofLog > 1) std::cout << log << std::endl;
    }
    {
        m_defaultIlluminationEstimationPipeline.m_hitGroupProgramGroups.resize(
                static_cast<int>(DefaultIlluminationEstimationRayType::RayTypeCount));
        char log[2048];
        size_t sizeofLog = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = m_defaultIlluminationEstimationPipeline.m_module;
        pgDesc.hitgroup.moduleAH = m_defaultIlluminationEstimationPipeline.m_module;
        // -------------------------------------------------------
        // radiance rays
        // -------------------------------------------------------
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__illuminationEstimation";
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__illuminationEstimation";
        OPTIX_CHECK(optixProgramGroupCreate(m_optixDeviceContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeofLog,
                                            &m_defaultIlluminationEstimationPipeline.m_hitGroupProgramGroups[static_cast<int>(DefaultIlluminationEstimationRayType::RadianceRayType)]
        ));
        if (sizeofLog > 1) std::cout << log << std::endl;
    }
}

__global__ void ApplyTransformKernel(
        int size, glm::mat4 globalTransform,
        Vertex *vertices,
        glm::vec3 *targetPositions) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        targetPositions[idx] = globalTransform * glm::vec4(vertices[idx].m_position, 1.0f);
        //targetNormals[idx] = glm::normalize(globalTransform * glm::vec4(vertices[idx].m_normal, 0.0f));
        //targetTangents[idx] = glm::normalize(globalTransform * glm::vec4(vertices[idx].m_tangent, 0.0f));
    }
}

void RayTracer::BuildAccelerationStructure() {
    bool uploadVertices = false;
    if (m_verticesBuffer.size() != m_instances.size()) uploadVertices = true;
    else {
        for (auto &i : m_instances) {
            if (i.m_verticesUpdateFlag) {
                uploadVertices = true;
                break;
            }
        }
    }
    if (uploadVertices) {
        for (auto &i : m_verticesBuffer) i.Free();
        for (auto &i : m_trianglesBuffer) i.Free();
        for (auto &i : m_transformedPositionsBuffer) i.Free();

        m_verticesBuffer.clear();
        m_trianglesBuffer.clear();
        m_transformedPositionsBuffer.clear();

        m_verticesBuffer.resize(m_instances.size());
        m_trianglesBuffer.resize(m_instances.size());
        m_transformedPositionsBuffer.resize(m_instances.size());
    }
    OptixTraversableHandle asHandle = 0;

    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(m_instances.size());
    std::vector<CUdeviceptr> deviceVertexPositions(m_instances.size());
    std::vector<CUdeviceptr> deviceVertexTriangles(m_instances.size());
    std::vector<CUdeviceptr> deviceTransforms(m_instances.size());
    std::vector<uint32_t> triangleInputFlags(m_instances.size());

    for (int meshID = 0; meshID < m_instances.size(); meshID++) {
        // upload the model to the device: the builder
        RayTracerInstance &triangleMesh = m_instances[meshID];
        if (uploadVertices) {
            m_verticesBuffer[meshID].Upload(*triangleMesh.m_vertices);
            m_transformedPositionsBuffer[meshID].Resize(triangleMesh.m_vertices->size() * sizeof(glm::vec3));
        }

        if (uploadVertices || triangleMesh.m_transformUpdateFlag) {
            int blockSize = 0;      // The launch configurator returned block size
            int minGridSize = 0;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
            int gridSize = 0;       // The actual grid size needed, based on input size
            int size = triangleMesh.m_vertices->size();
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ApplyTransformKernel, 0, size);
            gridSize = (size + blockSize - 1) / blockSize;
            ApplyTransformKernel << < gridSize, blockSize >> > (size, triangleMesh.m_globalTransform,
                    static_cast<Vertex *>(m_verticesBuffer[meshID].m_dPtr),
                    static_cast<glm::vec3 *>(m_transformedPositionsBuffer[meshID].m_dPtr));
            CUDA_SYNC_CHECK();
        }

        triangleMesh.m_verticesUpdateFlag = false;
        triangleMesh.m_transformUpdateFlag = false;
        m_trianglesBuffer[meshID].Upload(*triangleMesh.m_triangles);
        triangleInput[meshID] = {};
        triangleInput[meshID].type
                = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        deviceVertexPositions[meshID] = m_transformedPositionsBuffer[meshID].DevicePointer();
        deviceVertexTriangles[meshID] = m_trianglesBuffer[meshID].DevicePointer();

        triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
        triangleInput[meshID].triangleArray.numVertices = static_cast<int>(triangleMesh.m_vertices->size());
        triangleInput[meshID].triangleArray.vertexBuffers = &deviceVertexPositions[meshID];

        //triangleInput[meshID].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
        //triangleInput[meshID].triangleArray.preTransform = deviceTransforms[meshID];

        triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::uvec3);
        triangleInput[meshID].triangleArray.numIndexTriplets = static_cast<int>(triangleMesh.m_triangles->size());
        triangleInput[meshID].triangleArray.indexBuffer = deviceVertexTriangles[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelerateOptions = {};
    accelerateOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
                                   | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelerateOptions.motionOptions.numKeys = 1;
    accelerateOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                        (m_optixDeviceContext,
                         &accelerateOptions,
                         triangleInput.data(),
                         static_cast<int>(m_instances.size()),  // num_build_inputs
                         &blasBufferSizes
                        ));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CudaBuffer compactedSizeBuffer;
    compactedSizeBuffer.Resize(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.DevicePointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CudaBuffer tempBuffer;
    tempBuffer.Resize(blasBufferSizes.tempSizeInBytes);

    CudaBuffer outputBuffer;
    outputBuffer.Resize(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(m_optixDeviceContext,
            /* stream */nullptr,
                                &accelerateOptions,
                                triangleInput.data(),
                                static_cast<int>(m_instances.size()),
                                tempBuffer.DevicePointer(),
                                tempBuffer.m_sizeInBytes,

                                outputBuffer.DevicePointer(),
                                outputBuffer.m_sizeInBytes,

                                &asHandle,

                                &emitDesc, 1
    ));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.Download(&compactedSize, 1);

    m_acceleratedStructuresBuffer.Resize(compactedSize);
    OPTIX_CHECK(optixAccelCompact(m_optixDeviceContext,
            /*stream:*/nullptr,
                                  asHandle,
                                  m_acceleratedStructuresBuffer.DevicePointer(),
                                  m_acceleratedStructuresBuffer.m_sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.Free(); // << the Uncompacted, temporary output buffer
    tempBuffer.Free();
    compactedSizeBuffer.Free();

    m_defaultRenderingLaunchParams.m_traversable = asHandle;
    m_defaultIlluminationEstimationLaunchParams.m_traversable = asHandle;
    m_hasAccelerationStructure = true;
}

void RayTracer::SetAccumulate(const bool &value) {
    m_defaultRenderingPipeline.m_accumulate = value;
    m_defaultRenderingPipeline.m_statusChanged = true;
}

void RayTracer::AssemblePipelines() {
    AssemblePipeline(m_defaultRenderingPipeline);
    AssemblePipeline(m_defaultIlluminationEstimationPipeline);
}

void RayTracer::CreateRayGenProgram(RayTracerPipeline &targetPipeline, char entryFunctionName[]) const {
    targetPipeline.m_rayGenProgramGroups.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = targetPipeline.m_module;
    pgDesc.raygen.entryFunctionName = entryFunctionName;
    char log[2048];
    size_t sizeofLog = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(m_optixDeviceContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeofLog,
                                        &targetPipeline.m_rayGenProgramGroups[0]
    ));
    if (sizeofLog > 1) std::cout << log << std::endl;
}

void RayTracer::CreateModule(RayTracerPipeline &targetPipeline, char ptxCode[],
                             char launchParamsName[]) const {
    targetPipeline.m_launchParamsName = launchParamsName;

    targetPipeline.m_moduleCompileOptions.maxRegisterCount = 50;
    targetPipeline.m_moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    targetPipeline.m_moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    targetPipeline.m_pipelineCompileOptions = {};
    targetPipeline.m_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    targetPipeline.m_pipelineCompileOptions.usesMotionBlur = false;
    targetPipeline.m_pipelineCompileOptions.numPayloadValues = 2;
    targetPipeline.m_pipelineCompileOptions.numAttributeValues = 2;
    targetPipeline.m_pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    targetPipeline.m_pipelineCompileOptions.pipelineLaunchParamsVariableName = launchParamsName;

    targetPipeline.m_pipelineLinkOptions.maxTraceDepth = 31;

    const std::string code = ptxCode;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(m_optixDeviceContext,
                                         &targetPipeline.m_moduleCompileOptions,
                                         &targetPipeline.m_pipelineCompileOptions,
                                         code.c_str(),
                                         code.size(),
                                         log, &sizeof_log,
                                         &targetPipeline.m_module
    ));
    if (sizeof_log > 1) std::cout << log << std::endl;
}

void RayTracer::AssemblePipeline(RayTracerPipeline &targetPipeline) const {
    std::vector<OptixProgramGroup> programGroups;
    for (auto *pg : targetPipeline.m_rayGenProgramGroups)
        programGroups.push_back(pg);
    for (auto *pg : targetPipeline.m_missProgramGroups)
        programGroups.push_back(pg);
    for (auto *pg : targetPipeline.m_hitGroupProgramGroups)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeofLog = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(m_optixDeviceContext,
                                    &targetPipeline.m_pipelineCompileOptions,
                                    &targetPipeline.m_pipelineLinkOptions,
                                    programGroups.data(),
                                    static_cast<int>(programGroups.size()),
                                    log, &sizeofLog,
                                    &targetPipeline.m_pipeline
    ));
    if (sizeofLog > 1) std::cout << log << std::endl;

    OPTIX_CHECK(optixPipelineSetStackSize
                        (/* [in] The pipeline to configure the stack size for */
                                targetPipeline.m_pipeline,
                                /* [in] The direct stack size requirement for direct
                                   callables invoked from IS or AH. */
                                2 * 1024,
                                /* [in] The direct stack size requirement for direct
                                   callables invoked from RG, MS, or CH.  */
                                2 * 1024,
                                /* [in] The continuation stack requirement. */
                                2 * 1024,
                                /* [in] The maximum depth of a traversable graph
                                   passed to trace. */
                                1));
    if (sizeofLog > 1) std::cout << log << std::endl;
}

void RayTracer::BuildShaderBindingTable(std::vector<std::pair<unsigned, cudaTextureObject_t>> &boundTextures,
                                        std::vector<cudaGraphicsResource_t> &boundResources) {
    {
        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        std::vector<DefaultRenderingRayGenRecord> raygenRecords;
        for (int i = 0; i < m_defaultRenderingPipeline.m_rayGenProgramGroups.size(); i++) {
            DefaultRenderingRayGenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_defaultRenderingPipeline.m_rayGenProgramGroups[i], &rec));
            rec.m_data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        m_defaultRenderingPipeline.m_rayGenRecordsBuffer.Upload(raygenRecords);
        m_defaultRenderingPipeline.m_sbt.raygenRecord = m_defaultRenderingPipeline.m_rayGenRecordsBuffer.DevicePointer();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        std::vector<DefaultRenderingRayMissRecord> missRecords;
        for (int i = 0; i < m_defaultRenderingPipeline.m_missProgramGroups.size(); i++) {
            DefaultRenderingRayMissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_defaultRenderingPipeline.m_missProgramGroups[i], &rec));
            rec.m_data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        m_defaultRenderingPipeline.m_missRecordsBuffer.Upload(missRecords);
        m_defaultRenderingPipeline.m_sbt.missRecordBase = m_defaultRenderingPipeline.m_missRecordsBuffer.DevicePointer();
        m_defaultRenderingPipeline.m_sbt.missRecordStrideInBytes = sizeof(DefaultRenderingRayMissRecord);
        m_defaultRenderingPipeline.m_sbt.missRecordCount = static_cast<int>(missRecords.size());

        // ------------------------------------------------------------------
        // build hit records
        // ------------------------------------------------------------------

        // we don't actually have any objects in this example, but let's
        // create a dummy one so the SBT doesn't have any null pointers
        // (which the sanity checks in compilation would complain about)
        const int numObjects = m_verticesBuffer.size();
        std::vector<DefaultRenderingRayHitRecord> hitGroupRecords;
        for (int i = 0; i < numObjects; i++) {
            for (int rayID = 0; rayID < static_cast<int>(DefaultRenderingRayType::RayTypeCount); rayID++) {
                DefaultRenderingRayHitRecord rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(m_defaultRenderingPipeline.m_hitGroupProgramGroups[rayID], &rec));
                rec.m_data.m_mesh.m_vertices = reinterpret_cast<Vertex *>(m_verticesBuffer[i].DevicePointer());
                rec.m_data.m_mesh.m_triangles = reinterpret_cast<glm::uvec3 *>(m_trianglesBuffer[i].DevicePointer());
                rec.m_data.m_mesh.m_transform = m_instances[i].m_globalTransform;
                rec.m_data.m_material.m_surfaceColor = m_instances[i].m_surfaceColor;
                rec.m_data.m_material.m_roughness = m_instances[i].m_roughness;
                rec.m_data.m_material.m_metallic = m_instances[i].m_metallic;
                rec.m_data.m_material.m_albedoTexture = 0;
                rec.m_data.m_material.m_normalTexture = 0;
                rec.m_data.m_material.m_diffuseIntensity = m_instances[i].m_diffuseIntensity;
                if (m_instances[i].m_albedoTexture != 0) {
                    bool duplicate = false;
                    for (auto &boundTexture : boundTextures) {
                        if (boundTexture.first == m_instances[i].m_albedoTexture) {
                            rec.m_data.m_material.m_albedoTexture = boundTexture.second;
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
#pragma region Bind output texture
                        cudaArray_t textureArray;
                        cudaGraphicsResource_t graphicsResource;
                        CUDA_CHECK(GraphicsGLRegisterImage(&graphicsResource, m_instances[i].m_albedoTexture,
                                                           GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
                        CUDA_CHECK(GraphicsMapResources(1, &graphicsResource, nullptr));
                        CUDA_CHECK(GraphicsSubResourceGetMappedArray(&textureArray, graphicsResource, 0, 0));
                        struct cudaResourceDesc cudaResourceDesc;
                        memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
                        cudaResourceDesc.resType = cudaResourceTypeArray;
                        cudaResourceDesc.res.array.array = textureArray;
                        struct cudaTextureDesc cudaTextureDesc;
                        memset(&cudaTextureDesc, 0, sizeof(cudaTextureDesc));
                        cudaTextureDesc.addressMode[0] = cudaAddressModeWrap;
                        cudaTextureDesc.addressMode[1] = cudaAddressModeWrap;
                        cudaTextureDesc.filterMode = cudaFilterModeLinear;
                        cudaTextureDesc.readMode = cudaReadModeElementType;
                        cudaTextureDesc.normalizedCoords = 1;
                        CUDA_CHECK(CreateTextureObject(&rec.m_data.m_material.m_albedoTexture, &cudaResourceDesc,
                                                       &cudaTextureDesc, nullptr));
#pragma endregion
                        boundResources.push_back(graphicsResource);
                        boundTextures.emplace_back(m_instances[i].m_albedoTexture,
                                                   rec.m_data.m_material.m_albedoTexture);
                    }
                }
                if (m_instances[i].m_normalTexture != 0) {
                    bool duplicate = false;
                    for (auto &boundTexture : boundTextures) {
                        if (boundTexture.first == m_instances[i].m_normalTexture) {
                            rec.m_data.m_material.m_normalTexture = boundTexture.second;
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
#pragma region Bind output texture
                        cudaArray_t textureArray;
                        cudaGraphicsResource_t graphicsResource;
                        CUDA_CHECK(GraphicsGLRegisterImage(&graphicsResource, m_instances[i].m_normalTexture,
                                                           GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
                        CUDA_CHECK(GraphicsMapResources(1, &graphicsResource, nullptr));
                        CUDA_CHECK(GraphicsSubResourceGetMappedArray(&textureArray, graphicsResource, 0, 0));
                        struct cudaResourceDesc cudaResourceDesc;
                        memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
                        cudaResourceDesc.resType = cudaResourceTypeArray;
                        cudaResourceDesc.res.array.array = textureArray;
                        struct cudaTextureDesc cudaTextureDesc;
                        memset(&cudaTextureDesc, 0, sizeof(cudaTextureDesc));
                        cudaTextureDesc.addressMode[0] = cudaAddressModeWrap;
                        cudaTextureDesc.addressMode[1] = cudaAddressModeWrap;
                        cudaTextureDesc.filterMode = cudaFilterModeLinear;
                        cudaTextureDesc.readMode = cudaReadModeElementType;
                        cudaTextureDesc.normalizedCoords = 1;
                        CUDA_CHECK(CreateTextureObject(&rec.m_data.m_material.m_normalTexture, &cudaResourceDesc,
                                                       &cudaTextureDesc, nullptr));
#pragma endregion
                        boundResources.push_back(graphicsResource);
                        boundTextures.emplace_back(m_instances[i].m_normalTexture,
                                                   rec.m_data.m_material.m_normalTexture);
                    }
                }
                hitGroupRecords.push_back(rec);
            }
        }
        m_defaultRenderingPipeline.m_hitGroupRecordsBuffer.Upload(hitGroupRecords);
        m_defaultRenderingPipeline.m_sbt.hitgroupRecordBase = m_defaultRenderingPipeline.m_hitGroupRecordsBuffer.DevicePointer();
        m_defaultRenderingPipeline.m_sbt.hitgroupRecordStrideInBytes = sizeof(DefaultRenderingRayHitRecord);
        m_defaultRenderingPipeline.m_sbt.hitgroupRecordCount = static_cast<int>(hitGroupRecords.size());
    }
    {
        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        std::vector<DefaultIlluminationEstimationRayGenRecord> raygenRecords;
        for (int i = 0; i < m_defaultIlluminationEstimationPipeline.m_rayGenProgramGroups.size(); i++) {
            DefaultIlluminationEstimationRayGenRecord rec;
            OPTIX_CHECK(
                    optixSbtRecordPackHeader(m_defaultIlluminationEstimationPipeline.m_rayGenProgramGroups[i], &rec));
            rec.m_data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        m_defaultIlluminationEstimationPipeline.m_rayGenRecordsBuffer.Upload(raygenRecords);
        m_defaultIlluminationEstimationPipeline.m_sbt.raygenRecord = m_defaultIlluminationEstimationPipeline.m_rayGenRecordsBuffer.DevicePointer();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        std::vector<DefaultIlluminationEstimationRayMissRecord> missRecords;
        for (int i = 0; i < m_defaultIlluminationEstimationPipeline.m_missProgramGroups.size(); i++) {
            DefaultIlluminationEstimationRayMissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_defaultIlluminationEstimationPipeline.m_missProgramGroups[i], &rec));
            rec.m_data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        m_defaultIlluminationEstimationPipeline.m_missRecordsBuffer.Upload(missRecords);
        m_defaultIlluminationEstimationPipeline.m_sbt.missRecordBase = m_defaultIlluminationEstimationPipeline.m_missRecordsBuffer.DevicePointer();
        m_defaultIlluminationEstimationPipeline.m_sbt.missRecordStrideInBytes = sizeof(DefaultIlluminationEstimationRayMissRecord);
        m_defaultIlluminationEstimationPipeline.m_sbt.missRecordCount = static_cast<int>(missRecords.size());

        // ------------------------------------------------------------------
        // build hit records
        // ------------------------------------------------------------------

        // we don't actually have any objects in this example, but let's
        // create a dummy one so the SBT doesn't have any null pointers
        // (which the sanity checks in compilation would complain about)
        const int numObjects = m_verticesBuffer.size();
        std::vector<DefaultIlluminationEstimationRayHitRecord> hitGroupRecords;
        for (int i = 0; i < numObjects; i++) {
            for (int rayID = 0; rayID < static_cast<int>(DefaultIlluminationEstimationRayType::RayTypeCount); rayID++) {
                DefaultIlluminationEstimationRayHitRecord rec;
                OPTIX_CHECK(
                        optixSbtRecordPackHeader(m_defaultIlluminationEstimationPipeline.m_hitGroupProgramGroups[rayID],
                                                 &rec));
                rec.m_data.m_mesh.m_vertices = reinterpret_cast<Vertex *>(m_verticesBuffer[i].DevicePointer());
                rec.m_data.m_mesh.m_triangles = reinterpret_cast<glm::uvec3 *>(m_trianglesBuffer[i].DevicePointer());
                rec.m_data.m_mesh.m_transform = m_instances[i].m_globalTransform;
                rec.m_data.m_material.m_surfaceColor = m_instances[i].m_surfaceColor;
                rec.m_data.m_material.m_roughness = m_instances[i].m_roughness;
                rec.m_data.m_material.m_metallic = m_instances[i].m_metallic;
                rec.m_data.m_material.m_albedoTexture = 0;
                rec.m_data.m_material.m_normalTexture = 0;
                rec.m_data.m_material.m_diffuseIntensity = m_instances[i].m_diffuseIntensity;
                hitGroupRecords.push_back(rec);
            }
        }
        m_defaultIlluminationEstimationPipeline.m_hitGroupRecordsBuffer.Upload(hitGroupRecords);
        m_defaultIlluminationEstimationPipeline.m_sbt.hitgroupRecordBase = m_defaultIlluminationEstimationPipeline.m_hitGroupRecordsBuffer.DevicePointer();
        m_defaultIlluminationEstimationPipeline.m_sbt.hitgroupRecordStrideInBytes = sizeof(DefaultIlluminationEstimationRayHitRecord);
        m_defaultIlluminationEstimationPipeline.m_sbt.hitgroupRecordCount = static_cast<int>(hitGroupRecords.size());
    }
}
