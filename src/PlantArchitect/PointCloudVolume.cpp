#include "PointCloudVolume.hpp"
#include "PointCloud.hpp"
using namespace PlantArchitect;

void PointCloudVolume::OnInspect()
{
	static AssetRef pointCloud;
	if(Editor::DragAndDropButton(pointCloud, "Point Cloud",
		{ "PointCloud"}, true))
	{
		if(auto pc = pointCloud.Get<PointCloud>())
		{
			m_points.resize(pc->m_points.size());
			for(int i = 0; i < pc->m_points.size(); i++)
			{
				auto point = pc->m_points[i] + pc->m_offset;
				m_points[i] = glm::vec3(point.y, point.x, point.z);
			}
			pointCloud.Clear();
		}
	}
	ImGui::Text("Size: %d", m_points.size());
}

glm::vec3 PointCloudVolume::GetRandomPoint()
{
	return m_points[m_counter++ % m_points.size()];
}

