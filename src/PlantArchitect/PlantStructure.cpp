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
    auto& internode = m_internodes[handle];
    internode.m_recycled = true;
    internode.m_children.clear();

    //TODO: Remove subsequent internodes and branches.
    auto& branch = m_branches[internode.m_branchHandle];

    m_internodePool.emplace(handle);
}

void Plant::RecycleBranch(BranchHandle handle) {
    assert(!m_branches[handle].m_recycled);
    auto& branch = m_branches[handle];
    branch.m_recycled = true;
    for(const auto& child : branch.m_children){
        RecycleBranch(child);
    }
    branch.m_children.clear();
    if(!branch.m_internodes.empty()) {
        auto internodes = branch.m_internodes;
        for (const auto &i: internodes) {
            RecycleInternode(i);
        }
    }
    m_branchPool.emplace(handle);
}

InternodeHandle Plant::AllocateInternode() {
    if (m_internodePool.empty()) {
        auto newInternode = m_internodes.emplace_back(m_internodes.size());
        return newInternode.m_handle;
    }
    auto handle = m_internodePool.front();
    m_internodePool.pop();
    auto& internode = m_internodes[handle];
    internode.m_recycled = false;
    return handle;
}

BranchHandle Plant::AllocateBranch() {
    if (m_branchPool.empty()) {
        auto newBranch = m_branches.emplace_back(m_branches.size());
        return newBranch.m_handle;
    }
    auto handle = m_branchPool.front();
    m_branchPool.pop();
    auto& branch = m_branches[handle];
    branch.m_recycled = false;
    return handle;
}

void Plant::SetParentBranch(BranchHandle targetHandle, BranchHandle parentHandle) {
    assert(targetHandle < m_branches.size() && parentHandle < m_branches.size());
    auto& targetBranch = m_branches[targetHandle];
    auto& parentBranch = m_branches[parentHandle];
    assert(!targetBranch.m_recycled);
    assert(!parentBranch.m_recycled);
    targetBranch.m_parent = parentHandle;
#ifdef _DEBUG
    auto &children = parentBranch.m_children;
    for (int i = 0; i < children.size(); i++) {
        if (targetHandle == parentHandle) {
            UNIENGINE_ERROR("Children branch exists!")
            return;
        }
    }
#endif
    parentBranch.m_children.emplace_back(targetHandle);
}

void Plant::RemoveChildBranch(BranchHandle targetHandle, BranchHandle childHandle) {
    assert(targetHandle < m_branches.size() && childHandle < m_branches.size());
    auto& targetBranch = m_branches[targetHandle];
    auto& childBranch = m_branches[childHandle];
    assert(!targetBranch.m_recycled);
    assert(!childBranch.m_recycled);
    auto &children = targetBranch.m_children;
    for (int i = 0; i < children.size(); i++) {
        if (children[i] == childHandle) {
            children[i] = children.back();
            children.pop_back();
            return;
        }
    }
    UNIENGINE_ERROR("Children branch not exist!")
}

void Plant::SetParentInternode(InternodeHandle targetHandle, InternodeHandle parentHandle) {
    assert(targetHandle < m_internodes.size() && parentHandle < m_internodes.size());
    auto& targetInternode = m_internodes[targetHandle];
    auto& parentInternode = m_internodes[parentHandle];
    assert(!targetInternode.m_recycled);
    assert(!parentInternode.m_recycled);
    targetInternode.m_parent = parentHandle;
#ifdef _DEBUG
    auto &children = parentInternode.m_children;
    for (int i = 0; i < children.size(); i++) {
        if (targetHandle == parentHandle) {
            UNIENGINE_ERROR("Children internode exists!")
            return;
        }
    }
#endif
    parentInternode.m_children.emplace_back(targetHandle);
}

void Plant::RemoveChildInternode(InternodeHandle targetHandle, InternodeHandle childHandle) {
    assert(targetHandle < m_internodes.size() && childHandle < m_internodes.size());
    auto& targetInternode = m_internodes[targetHandle];
    auto& childInternode = m_internodes[childHandle];
    assert(!targetInternode.m_recycled);
    assert(!childInternode.m_recycled);
    auto &children = targetInternode.m_children;
    for (int i = 0; i < children.size(); i++) {
        if (children[i] == childHandle) {
            children[i] = children.back();
            children.pop_back();
            return;
        }
    }
    UNIENGINE_ERROR("Children internode not exist!")
}

InternodeHandle Plant::Extend(InternodeHandle targetHandle) {
    assert(targetHandle < m_internodes.size());
    auto& targetInternode = m_internodes[targetHandle];
    assert(!targetInternode.m_recycled);
    assert(targetInternode.m_branchHandle < m_branches.size());
    auto& branch = m_branches[targetInternode.m_branchHandle];
    assert(!branch.m_recycled);

    auto newInternodeHandle = AllocateInternode();
    SetParentInternode(newInternodeHandle, targetHandle);
    auto& newInternode = m_internodes[targetHandle];

    //If this internode is end node, we simply add this to current branch.
    if(targetInternode.m_endNode){
        targetInternode.m_endNode = false;
        branch.m_internodes.emplace_back(newInternodeHandle);
        newInternode.m_branchHandle = targetInternode.m_branchHandle;
    }else{
        //If this internode is not the end node, we need to break the original branch into 2 branch segments, and also create a new branch from the original branch.
        auto newBranchHandle = AllocateBranch();
        auto& newBranch = m_branches[newInternodeHandle];
        SetParentBranch(newBranchHandle, targetInternode.m_branchHandle);
        newInternode.m_branchHandle = newBranchHandle;
        newBranch.m_internodes.emplace_back(newInternodeHandle);
    }

    return newInternodeHandle;
}



