#pragma once
#include <IVolume.hpp>
#include <plant_architect_export.h>
using namespace UniEngine;
namespace PlantArchitect {
	class PLANT_ARCHITECT_API PointCloudVolume : public IVolume
	{
		std::vector<glm::dvec3> m_points;
		int m_counter = 0;
	public:
		void OnInspect() override;
		glm::vec3 GetRandomPoint() override;
	};
}