//
// Created by lllll on 9/11/2022.
//

#include "Strand.hpp"

using namespace PlantArchitect;

void StrandPlant::GenerateStrands(float pointDistance) {
    m_pointDistance = pointDistance;
    m_strands.clear();
    m_root.reset();
    m_root = std::make_shared<StrandsIntersection>();
    int yRange = glm::ceil(m_rootRegion.m_radius.y / (m_pointDistance * glm::cos(glm::radians(30.0f))));
    int xRange = glm::ceil(m_rootRegion.m_radius.x / m_pointDistance);
    for (int i = -xRange; i <= xRange; i++) {
        for (int j = -yRange; j <= yRange; j++) {
            glm::vec2 point;
            point.x = i * m_pointDistance + (j % 2 == 0 ? 0 : 0.5f * m_pointDistance);
            point.y = j * m_pointDistance * glm::cos(glm::radians(30.0f));
            if (m_rootRegion.IsInRegion(point)) {
                //Generate strand here.
                std::shared_ptr<StrandKnot> knot = std::make_shared<StrandKnot>();
                knot->m_position = glm::vec3(point, 0);
                m_root->m_knots.push_back(knot);
                auto strand = std::make_shared<Strand>();
                strand->m_start = knot;
                m_strands.push_back(std::move(strand));
            }
        }
    }
}

void StrandPlant::OnInspect() {
    static bool drawRegion = true;
    if(m_root) ImGui::Text("Strand count: %d", m_root->m_knots.size());
    if (ImGui::Button("Generate Strands")) GenerateStrands();
    ImGui::Checkbox("Draw Region", &drawRegion);
    if (ImGui::Begin("Region", &drawRegion)) {
        static auto scrolling = glm::vec2(0.0f);
        static float zoomFactor = 1000.0f;
        if (ImGui::Button("Recenter")) {
            scrolling = glm::vec2(0.0f);
        }
        ImGui::DragFloat("Zoom", &zoomFactor, 1.0f, 100.0f, 2500.0f);
        zoomFactor = glm::clamp(zoomFactor, 100.0f, 2500.0f);
        ImGuiIO &io = ImGui::GetIO();
        ImDrawList *draw_list = ImGui::GetWindowDrawList();

        ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
        ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
        if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
        if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
        ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
        const ImVec2 origin(canvas_p0.x + canvas_sz.x / 2.0f + scrolling.x,
                            canvas_p0.y + canvas_sz.y / 2.0f + scrolling.y); // Lock scrolled origin
        const ImVec2 mousePosInCanvas((io.MousePos.x - origin.x) / zoomFactor, (io.MousePos.y - origin.y) / zoomFactor);

        // Draw border and background color
        draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
        draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

        // This will catch our interactions
        ImGui::InvisibleButton("canvas", canvas_sz,
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
        const bool isMouseHovered = ImGui::IsItemHovered(); // Hovered
        const bool isMouseActive = ImGui::IsItemActive();   // Held

        // Pan (we use a zero mouse threshold when there's no context menu)
        // You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
        const float mouse_threshold_for_pan = -1.0f;
        if (isMouseActive && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan)) {
            scrolling.x += io.MouseDelta.x;
            scrolling.y += io.MouseDelta.y;
        }
        static bool addingLine = false;
        // Context menu (under default mouse threshold)
        ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
        if (drag_delta.x == 0.0f && drag_delta.y == 0.0f)
            ImGui::OpenPopupOnItemClick("context", ImGuiPopupFlags_MouseButtonRight);
        static std::vector<glm::vec2> points = {};
        if (ImGui::BeginPopup("context")) {

            ImGui::EndPopup();
        }

        // Draw grid + all lines in the canvas
        draw_list->PushClipRect(canvas_p0, canvas_p1, true);
        if (false) {
            const float GRID_STEP = 64.0f;
            for (float x = fmodf(scrolling.x, GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
                draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y),
                                   IM_COL32(200, 200, 200, 40));
            for (float y = fmodf(scrolling.y, GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
                draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y),
                                   IM_COL32(200, 200, 200, 40));
        }
        // Add first and second point
        if (isMouseHovered && !addingLine && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            points.clear();
            points.push_back({mousePosInCanvas.x, mousePosInCanvas.y});
            addingLine = true;
        }
        if (!addingLine) {
            const auto size = m_rootRegion.m_regionPoints.size();
            for (int i = 0; i < size - 1; i++) {
                draw_list->AddLine(ImVec2(origin.x + m_rootRegion.m_regionPoints[i].x * zoomFactor,
                                          origin.y + m_rootRegion.m_regionPoints[i].y * zoomFactor),
                                   ImVec2(origin.x + m_rootRegion.m_regionPoints[i + 1].x * zoomFactor,
                                          origin.y + m_rootRegion.m_regionPoints[i + 1].y * zoomFactor),
                                   IM_COL32(255, 255, 0, 255), 2.0f);
            }
            draw_list->AddLine(ImVec2(origin.x + m_rootRegion.m_regionPoints[size - 1].x * zoomFactor,
                                      origin.y + m_rootRegion.m_regionPoints[size - 1].y * zoomFactor),
                               ImVec2(origin.x + m_rootRegion.m_regionPoints[0].x * zoomFactor,
                                      origin.y + m_rootRegion.m_regionPoints[0].y * zoomFactor),
                               IM_COL32(255, 255, 0, 255), 2.0f);
            if (m_root) {
                for (const auto &point: m_root->m_knots) {
                    if (point) {
                        auto position = ImVec2(origin.x + point->m_position.x * zoomFactor,
                                               origin.y + point->m_position.y * zoomFactor);
                        draw_list->AddCircleFilled(position, glm::clamp(m_pointDistance * 0.4f * zoomFactor, 1.0f, 100.0f),
                                                   IM_COL32(255, 255, 255, 255));
                    }
                }
            }
        } else {
            const auto size = points.size();
            for (int i = 0; i < size - 1; i++) {
                draw_list->AddLine(ImVec2(origin.x + points[i].x * zoomFactor,
                                          origin.y + points[i].y * zoomFactor),
                                   ImVec2(origin.x + points[i + 1].x * zoomFactor,
                                          origin.y + points[i + 1].y * zoomFactor),
                                   IM_COL32(255, 255, 0, 255), 2.0f);
            }

            if (glm::distance(points.back(), {mousePosInCanvas.x, mousePosInCanvas.y}) >= 10.0f / zoomFactor)
                points.push_back({mousePosInCanvas.x, mousePosInCanvas.y});
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                addingLine = false;
                m_rootRegion.Construct(points);
                GenerateStrands();
            }
        }
        draw_list->PopClipRect();
    }
    ImGui::End();

}

void StrandPlant::OnCreate() {
    GenerateStrands();
}

void StrandsIntersectionRegion::Construct(const std::vector<glm::vec2> &points) {
    m_regionPoints = points;
    //1. Calculate min/max bound
    auto max = glm::vec2(FLT_MIN);
    auto min = glm::vec2(FLT_MAX);
    for (const auto &point: m_regionPoints) {
        if (max.x < point.x) max.x = point.x;
        if (max.y < point.y) max.y = point.y;
        if (min.x > point.x) min.x = point.x;
        if (min.y > point.y) min.y = point.y;
    }
    auto center = (max + min) / 2.0f;
    m_radius = (max - min) / 2.0f;
    max -= center;
    min -= center;
    for (auto &point: m_regionPoints) {
        point -= center;
    }
}

StrandsIntersectionRegion::StrandsIntersectionRegion() {
    std::vector<glm::vec2> points;
    points.emplace_back(-0.1f, 0.1f);
    points.emplace_back(0.1f, 0.1f);
    points.emplace_back(0.1f, -0.1f);
    points.emplace_back(-0.1f, -0.1f);
    Construct(points);
}

bool RayLineIntersect(glm::vec2 rayOrigin, glm::vec2 rayDirection, glm::vec2 point1, glm::vec2 point2) {
    const auto v1 = rayOrigin - point1;
    const auto v2 = point2 - point1;
    const auto v3 = glm::vec2(-rayDirection.y, rayDirection.x);

    float dot = glm::dot(v2, v3);
    if (dot == 0.0f)
        return false;

    float t1 = (v2.x * v1.y - v2.y * v1.x) / dot;
    float t2 = glm::dot(v1, v3) / dot;

    //!!!!Check t2 >= 0 if we allow intersect on point 1
    if (t1 >= 0.0f && t2 > 0.0f && 1.0f - t2 >= 0.0f)
        return true;

    return false;
}

bool StrandsIntersectionRegion::IsInRegion(const glm::vec2 &point) const {
    const auto point2 = glm::vec2(1.0f);
    int windingNumber = 0;
    const auto size = m_regionPoints.size();
    for (int i = 0; i < size - 1; i++) {
        if (RayLineIntersect(point, point2, m_regionPoints[i], m_regionPoints[i + 1])) {
            windingNumber++;
        }
    }
    if (RayLineIntersect(point, point2, m_regionPoints[size - 1], m_regionPoints[0])) windingNumber++;
    if (windingNumber % 2 == 1) {
        return true;
    }
    return false;
}
