//
// Created by lllll on 9/11/2022.
//

#include "Strand.hpp"
#include "DataComponents.hpp"

using namespace PlantArchitect;
#pragma region Helpers

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

bool LineLineIntersect(glm::vec2 pa, glm::vec2 pb, glm::vec2 pc, glm::vec2 pd) {
    const auto v1 = pa - pc;
    const auto v2 = pd - pc;
    const auto v3 = glm::vec2(-(pb.y - pa.y), (pb.x - pa.x));

    float dot = glm::dot(v2, v3);
    if (dot == 0.0f)
        return false;

    float t1 = (v2.x * v1.y - v2.y * v1.x) / dot;
    float t2 = glm::dot(v1, v3) / dot;

    if (t1 > 0.0f && t1 < 1.0f && t2 > 0.0f && t2 < 1.0f)
        return true;

    return false;
}

bool InBoundary(const std::vector<glm::vec2> &boundary, const glm::vec2 &point) {
    const auto point2 = glm::vec2(1.0f, 0.0f);
    const auto point3 = glm::vec2(1.0f, 0.0f);
    int windingNumber = 0;
    const auto size = boundary.size();
    if (size < 3) return false;
    for (int i = 0; i < size - 1; i++) {
        if (RayLineIntersect(point, point2, boundary[i], boundary[i + 1]) &&
            RayLineIntersect(point, point3, boundary[i], boundary[i + 1])) {
            windingNumber++;
        }
    }
    if (RayLineIntersect(point, point2, boundary[size - 1], boundary[0]) &&
        RayLineIntersect(point, point3, boundary[size - 1], boundary[0]))
        windingNumber++;
    if (windingNumber % 2 == 1) {
        return true;
    }
    return false;
}

#pragma endregion

glm::vec2 GetPosition(const glm::ivec2 &coordinate) {
    return glm::vec2(coordinate.x + coordinate.y / 2.0f,
                     coordinate.y * glm::cos(glm::radians(30.0f)));
}

glm::ivec2 GetCoordinate(const glm::vec2 &position) {
    int y = glm::round(position.y / glm::cos(glm::radians(30.0f)));
    return glm::ivec2(glm::round((position.x - y / 2.0f)), y);
}

void StrandPlant::GenerateStrands(float pointDistance) {
    auto scene = GetScene();
    m_strands.clear();
    auto rootEntity = GetRoot();
    auto rootIntersection = scene->GetOrSetPrivateComponent<StrandsIntersection>(rootEntity).lock();

}

void StrandPlant::OnInspect() {
    if (ImGui::Button("Generate Strands")) GenerateStrands();
    static bool displayRootIntersection = true;
    ImGui::Checkbox("Display root intersection", &displayRootIntersection);
    if (displayRootIntersection) {
        auto scene = GetScene();
        auto rootEntity = GetRoot();
        auto rootIntersection = scene->GetOrSetPrivateComponent<StrandsIntersection>(rootEntity).lock();
        rootIntersection->DisplayIntersection("Root Intersection", true);
    }
}

void StrandPlant::OnCreate() {
}

Entity StrandPlant::GetRoot() {
    auto scene = GetScene();
    auto self = GetOwner();
    auto children = scene->GetChildren(self);
    bool found = false;
    Entity retVal;
    for (const auto &child: children) {
        if (scene->HasPrivateComponent<StrandsIntersection>(child)) {
            if (found) scene->DeleteEntity(child);
            else {
                found = true;
                retVal = child;
            }
        }
    }
    if (!found) {
        retVal = scene->CreateEntity("Root");
        scene->AddDataComponent<StrandIntersectionInfo>(retVal, StrandIntersectionInfo());
        auto rootIntersection = scene->GetOrSetPrivateComponent<StrandsIntersection>(retVal).lock();
        rootIntersection->m_isRoot = true;
        scene->SetParent(retVal, self);
    }
    return retVal;
}

void StrandsIntersection::Construct(const std::vector<glm::vec2> &points) {
    auto copiedPoints = points;
    //1. Calculate min/max bound
    auto max = glm::vec2(FLT_MIN);
    auto min = glm::vec2(FLT_MAX);
    for (const auto &point: copiedPoints) {
        if (max.x < point.x) max.x = point.x;
        if (max.y < point.y) max.y = point.y;
        if (min.x > point.x) min.x = point.x;
        if (min.y > point.y) min.y = point.y;
    }
    auto center = (max + min) / 2.0f;
    auto boundaryRadius = (max - min) / 2.0f;
    max -= center;
    min -= center;
    for (auto &point: copiedPoints) {
        point -= center;
    }
    m_strandKnots.clear();
    glm::ivec2 sum = glm::ivec2(0);
    int yRange = glm::ceil(boundaryRadius.y / glm::cos(glm::radians(30.0f)));
    int xRange = glm::ceil(boundaryRadius.x);
    for (int i = -xRange; i <= xRange; i++) {
        for (int j = -yRange; j <= yRange; j++) {
            glm::ivec2 coordinate;
            coordinate.y = j;
            coordinate.x = i - j / 2;
            if (InBoundary(copiedPoints, GetPosition(coordinate))) {
                m_strandKnots.push_back(std::make_shared<StrandKnot>());
                m_strandKnots.back()->m_coordinate = coordinate;
                sum += coordinate;
            }
        }
    }
    if (m_strandKnots.empty())return;
    sum /= m_strandKnots.size();
    for (auto &knot: m_strandKnots) {
        knot->m_coordinate -= sum;
    }
    CalculateConnectivity();
}

void StrandsIntersection::OnCreate() {
    std::vector<glm::vec2> points;
    points.emplace_back(-10.0f, 10.0f);
    points.emplace_back(10.0f, 10.0f);
    points.emplace_back(10.0f, -10.0f);
    points.emplace_back(-10.0f, -10.0f);
    Construct(points);
}


void StrandsIntersection::DisplayIntersection(const std::string &title, bool editable) {
    if (ImGui::Begin(title.c_str())) {
        static auto scrolling = glm::vec2(0.0f);
        static float zoomFactor = 10.0f;
        if (ImGui::Button("Recenter")) {
            scrolling = glm::vec2(0.0f);
        }
        ImGui::DragFloat("Zoom", &zoomFactor, zoomFactor / 100.0f, 1.0f, 50.0f);
        zoomFactor = glm::clamp(zoomFactor, 1.0f, 50.0f);
        ImGuiIO &io = ImGui::GetIO();
        ImDrawList *draw_list = ImGui::GetWindowDrawList();

        ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
        ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
        if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
        if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
        ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
        const ImVec2 origin(canvas_p0.x + canvas_sz.x / 2.0f + scrolling.x,
                            canvas_p0.y + canvas_sz.y / 2.0f + scrolling.y); // Lock scrolled origin
        const ImVec2 mousePosInCanvas((io.MousePos.x - origin.x) / zoomFactor,
                                      (io.MousePos.y - origin.y) / zoomFactor);

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
        if (editable && isMouseHovered && !addingLine && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            points.clear();
            points.emplace_back(mousePosInCanvas.x, mousePosInCanvas.y);
            addingLine = true;
        }
        {
            draw_list->AddCircle(origin,
                                 glm::clamp(0.5f * zoomFactor, 1.0f, 100.0f),
                                 IM_COL32(255,
                                          0,
                                          0, 255));
            for (const auto &knot: m_strandKnots) {
                auto pointPosition = GetPosition(knot->m_coordinate);
                auto canvasPosition = ImVec2(origin.x + pointPosition.x * zoomFactor,
                                             origin.y + pointPosition.y * zoomFactor);
                draw_list->AddCircleFilled(canvasPosition,
                                           glm::clamp(0.4f * zoomFactor, 1.0f, 100.0f),
                                           IM_COL32(255.0f * knot->m_distanceToBoundary / m_maxDistanceToBoundary,
                                                    255.0f * knot->m_distanceToBoundary / m_maxDistanceToBoundary,
                                                    255.0f * knot->m_distanceToBoundary / m_maxDistanceToBoundary, 255));

                if (zoomFactor > 20) {
                    auto textCanvasPosition = ImVec2(origin.x + pointPosition.x * zoomFactor - 0.3f * zoomFactor,
                                                     origin.y + pointPosition.y * zoomFactor - 0.3f * zoomFactor);
                    auto text = std::to_string(knot->m_distanceToBoundary);
                    draw_list->AddText(0, 0.5f * zoomFactor, textCanvasPosition, IM_COL32(255, 0, 0, 255),
                                       text.c_str());
                }
            }
        }
        if (addingLine) {
            const auto size = points.size();
            for (int i = 0; i < size - 1; i++) {
                draw_list->AddLine(ImVec2(origin.x + points[i].x * zoomFactor,
                                          origin.y + points[i].y * zoomFactor),
                                   ImVec2(origin.x + points[i + 1].x * zoomFactor,
                                          origin.y + points[i + 1].y * zoomFactor),
                                   IM_COL32(255, 0, 0, 255), 2.0f);
            }
            if (glm::distance(points.back(), {mousePosInCanvas.x, mousePosInCanvas.y}) >= 10.0f / zoomFactor)
                points.emplace_back(mousePosInCanvas.x, mousePosInCanvas.y);
            if (editable && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                addingLine = false;
                if (!CheckBoundary(points)) {
                    Construct(points);

                }
            }
        }
        draw_list->PopClipRect();
    }
    ImGui::End();
}

void StrandsIntersection::OnInspect() {
    static bool displayRegion = true;
    ImGui::Text("Knot size: %d", m_strandKnots.size());
    ImGui::Text("Max distance to boundary: %d", m_maxDistanceToBoundary);
    ImGui::Checkbox("Display Intersection", &displayRegion);
    if (displayRegion) {
        DisplayIntersection("Intersection", false);
    }
}

void StrandsIntersection::CalculateConnectivity() {
    m_maxDistanceToBoundary = 0;
    std::map<std::pair<int, int>, std::shared_ptr<StrandKnot>> knotsMap;
    for (const auto &knot: m_strandKnots) {
        knot->m_distanceToBoundary = 99999;
        knotsMap[{knot->m_coordinate.x, knot->m_coordinate.y}] = knot;
    }

    std::queue<std::shared_ptr<StrandKnot>> knotDistanceCalculationQueue;
    std::set<std::shared_ptr<StrandKnot>> visitedKnots;
    for (auto &knot: m_strandKnots) {
        int x = knot->m_coordinate.x;
        int y = knot->m_coordinate.y;
        knot->m_upLeft.reset();
        knot->m_upRight.reset();
        knot->m_right.reset();
        knot->m_downRight.reset();
        knot->m_downLeft.reset();
        knot->m_left.reset();
        auto search = knotsMap.find({x - 1, y + 1});
        if (search != knotsMap.end()) {
            knot->m_upLeft = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x, y + 1});
        if (search != knotsMap.end()) {
            knot->m_upRight = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x + 1, y});
        if (search != knotsMap.end()) {
            knot->m_right = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x + 1, y - 1});
        if (search != knotsMap.end()) {
            knot->m_downRight = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x, y - 1});
        if (search != knotsMap.end()) {
            knot->m_downLeft = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x - 1, y});
        if (search != knotsMap.end()) {
            knot->m_left = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
    }

    while (!knotDistanceCalculationQueue.empty()) {
        auto currentKnot = knotDistanceCalculationQueue.front();
        knotDistanceCalculationQueue.pop();
        if (!currentKnot->m_upLeft.expired()) {
            auto visitingKnot = currentKnot->m_upLeft.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_upRight.expired()) {
            auto visitingKnot = currentKnot->m_upRight.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_right.expired()) {
            auto visitingKnot = currentKnot->m_right.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_downRight.expired()) {
            auto visitingKnot = currentKnot->m_downRight.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_downLeft.expired()) {
            auto visitingKnot = currentKnot->m_downLeft.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_left.expired()) {
            auto visitingKnot = currentKnot->m_left.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (currentKnot->m_distanceToBoundary > m_maxDistanceToBoundary)
            m_maxDistanceToBoundary = currentKnot->m_distanceToBoundary;
    }
}

std::vector<std::shared_ptr<StrandKnot>> StrandsIntersection::GetBoundaryKnots() const {
    std::vector<std::shared_ptr<StrandKnot>> retVal;
    for (const auto &knot: m_strandKnots) {
        if (knot->m_distanceToBoundary == 0) {
            auto walker = knot;
            if (!knot->m_upLeft.expired() && knot->m_upLeft.lock()->m_distanceToBoundary == 0)
                walker = knot->m_upLeft.lock();
            else if (!knot->m_upRight.expired() && knot->m_upRight.lock()->m_distanceToBoundary == 0)
                walker = knot->m_upRight.lock();
            else if (!knot->m_right.expired() && knot->m_right.lock()->m_distanceToBoundary == 0)
                walker = knot->m_right.lock();
            else if (!knot->m_downRight.expired() && knot->m_downRight.lock()->m_distanceToBoundary == 0)
                walker = knot->m_downRight.lock();
            else if (!knot->m_downLeft.expired() && knot->m_downLeft.lock()->m_distanceToBoundary == 0)
                walker = knot->m_downLeft.lock();
            else if (!knot->m_left.expired() && knot->m_left.lock()->m_distanceToBoundary == 0)
                walker = knot->m_left.lock();
            auto prev = knot;
            while (walker != knot) {
                retVal.emplace_back(walker);
                auto save = walker;
                if (!walker->m_upLeft.expired() && walker->m_upLeft.lock() != prev &&
                    walker->m_upLeft.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_upLeft.lock();
                else if (!walker->m_upRight.expired() && walker->m_upRight.lock() != prev &&
                         walker->m_upRight.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_upRight.lock();
                else if (!walker->m_right.expired() && walker->m_right.lock() != prev &&
                         walker->m_right.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_right.lock();
                else if (!walker->m_downRight.expired() && walker->m_downRight.lock() != prev &&
                         walker->m_downRight.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_downRight.lock();
                else if (!walker->m_downLeft.expired() && walker->m_downLeft.lock() != prev &&
                         walker->m_downLeft.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_downLeft.lock();
                else if (!walker->m_left.expired() && walker->m_left.lock() != prev &&
                         walker->m_left.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_left.lock();
                prev = save;
            }
            retVal.emplace_back(walker);
            return retVal;
        }
    }
}

void StrandsIntersection::Extract(const glm::vec2 &direction, int numOfKnots,
                                  std::vector<std::shared_ptr<StrandKnot>> &extractedKnots) {

}

bool StrandsIntersection::CheckBoundary(const std::vector<glm::vec2> &points) {
    for (int i = 0; i < points.size(); i++) {
        auto &pa = points[(i == 0 ? points.size() - 1 : i - 1)];
        auto &pb = points[i];
        for (int j = 0; j < points.size(); j++) {
            auto &pc = points[(j == 0 ? points.size() - 1 : j - 1)];
            auto &pd = points[j];
            if (LineLineIntersect(pa, pb, pc, pd)) return true;
        }
    }
    return false;
}
