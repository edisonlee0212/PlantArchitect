//
// Created by lllll on 10/18/2022.
//

#include "PlantStructure.hpp"

using namespace Orchards;

Branch::Branch(BranchHandle handle) {
    m_handle = handle;
    m_recycled = false;
}

Internode::Internode(InternodeHandle handle) {
    m_handle = handle;
    m_recycled = false;
    m_endNode = true;
}

void Plant::RecycleInternode(InternodeHandle handle) {
    assert(!m_internodes[handle].m_recycled);
    m_internodes[handle].m_recycled = true;
    m_internodePool.emplace(handle);
}

void Plant::RecycleBranch(BranchHandle handle) {
    assert(!m_branches[handle].m_recycled);
    m_branches[handle].m_recycled = true;
    m_branchPool.emplace(handle);
}

InternodeHandle Plant::AllocateInternode() {
    if (m_internodePool.empty()) {
        auto newInternode = m_internodes.emplace_back(m_internodes.size());
        return newInternode.m_handle;
    }
    auto handle = m_internodePool.front();
    m_internodePool.pop();
    m_internodes[handle].m_recycled = false;
    return handle;
}

BranchHandle Plant::AllocateBranch() {
    if (m_branchPool.empty()) {
        auto newBranch = m_branches.emplace_back(m_branches.size());
        return newBranch.m_handle;
    }
    auto handle = m_branchPool.front();
    m_branchPool.pop();
    m_branches[handle].m_recycled = false;
    return handle;
}

void Plant::SetParent(BranchHandle target, BranchHandle parent) {
    assert(target < m_branches.size() && parent < m_branches.size());
    auto& targetBranch = m_branches[target];
    auto& parentBranch = m_branches[parent];
    assert(!targetBranch.m_recycled);
    assert(!parentBranch.m_recycled);
    targetBranch.m_parent = parent;
#ifdef _DEBUG
    auto &children = parentBranch.m_children;
    for (int i = 0; i < children.size(); i++) {
        if (target == parent) {
            UNIENGINE_ERROR("Children exists!")
            return;
        }
    }
#endif
    parentBranch.m_children.emplace_back(target);
}

void Plant::RemoveChild(BranchHandle target, BranchHandle child) {
    assert(target < m_branches.size() && child < m_branches.size());
    auto& targetBranch = m_branches[target];
    auto& childBranch = m_branches[child];
    assert(!targetBranch.m_recycled);
    assert(!childBranch.m_recycled);
    auto &children = targetBranch.m_children;
    for (int i = 0; i < children.size(); i++) {
        if (children[i] == child) {
            children[i] = children.back();
            children.pop_back();
            return;
        }
    }
    assert(false);
}

InternodeHandle Plant::Extend(InternodeHandle target) {
    assert(target < m_internodes.size());
    auto& targetInternode = m_internodes[target];
    assert(!targetInternode.m_recycled);
    assert(targetInternode.m_branchHandle < m_branches.size());
    auto& branch = m_branches[targetInternode.m_branchHandle];
    assert(!branch.m_recycled);

    auto newInternodeHandle = AllocateInternode();
    auto& newInternode = m_internodes[target];
    //If this internode is end node, we simply add this to current branch.
    if(targetInternode.m_endNode){
        branch.m_internodes.emplace_back(newInternodeHandle);
        newInternode.m_branchHandle = targetInternode.m_branchHandle;
    }
    //If this internode is not the end node, we need to break the original branch into 2 branch segments, and also create a new branch from the original branch.


    return newInternodeHandle;
}



