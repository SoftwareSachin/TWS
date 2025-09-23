"use client";
import React, { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Cookies from "universal-cookie";

import {
  deleteWorkSpace,
  getWorkSpace,
  updateWorkSpace,
} from "@/api/Workspace/workspace";
import { Button } from "@/components/ui/button";
import WorkspaceCard from "@/components/workspace/workspaceCard";
import Paginator from "@/components/utility/paginator";
import DeleteModal from "@/components/forms/deleteModal";
import { EditWorkspaceModal } from "@/components/forms/EditWorkspaceModal";
import { decodeToken } from "@/components/utility/decodeJwtToken";
import { getCookie, removeCookie } from "@/utils/cookieHelper";
import { constants } from "@/lib/constants";
import { showError, showSuccess } from "@/utils/toastUtils";
import SearchBox from "@/design_components/utility/search-box";

const WorkspacePage = () => {
  const [hasMounted, setHasMounted] = useState(false);
  const [listWorkSpace, setListWorkSpace] = useState([]);
  const [totalWorkList, setTotalWorkList] = useState(0);
  const [deletingId, setDeletingId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState(null);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [editingWorkspace, setEditingWorkspace] = useState(null);
  const [expiredToken, setExpiredToken] = useState(false);
  const [pagination, setPagination] = useState(null);
  const [totalPages, setTotalPages] = useState(1);

  const router = useRouter();
  const searchParams = useSearchParams();

  // Mount flag to safely access browser-only APIs
  useEffect(() => {
    setHasMounted(true);
  }, []);

  // AUTHENTICATION BYPASSED - Use mock data instead of tokens
  const token = hasMounted ? "mock-token" : null;
  const userDetails = hasMounted ? { clientId: "test-org-123", email: "test@amplifi.com" } : null;
  const [searchText, setSearchText] = useState("");


  useEffect(() => {
    if (hasMounted) {
      const currentPage = page ? +page : 1;
      if (!pagination || pagination.page !== currentPage) {
        setPagination({ page: currentPage, size: 8 });
      }
    }
  }, [hasMounted]);
  
  // Authentication bypassed - no redirect needed
  // useEffect(() => {
  //   if (hasMounted && (!token || !userDetails)) {
  //     showError("Session expired. Please login again.");
  //     router.push("/login");
  //   }
  // }, [hasMounted, token, userDetails]);

  // Handle pagination from query params
  const page = searchParams.get("page");
  let search = searchParams.get("id");
  if (!search && userDetails) {
    search = userDetails.clientId;
  }

  useEffect(() => {
    if (pagination && token && userDetails && hasMounted) {
      changePage();
      getWorkspaceList();
    }
  }, [pagination, searchText]);

  useEffect(() => {
    if (page && hasMounted) {
      const currentPage = +page;
      if (!pagination || pagination.page !== currentPage) {
        setPagination({ page: currentPage || 1, size: 8 });
      }
    }
  }, [page, hasMounted]);

  const changePage = () => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("page", pagination.page.toString());
    router.push(`?${params.toString()}`);
  };

  const getWorkspaceList = async () => {
    try {
      setLoading(true);
      const response = await getWorkSpace(search, pagination, searchText);
      setExpiredToken(false);
      setListWorkSpace(response?.data?.data?.items || []);
      setTotalWorkList(response?.data?.data?.total || 0);
      setTotalPages(response?.data?.data?.pages || 0);
    } catch (error) {
      console.error(error);
      const errMsg = error?.response?.data?.detail;
      if (error?.response?.status === 403) {
        if (!expiredToken) {
          getWorkspaceList(); // Retry once
          setExpiredToken(true);
        } else {
          // Authentication bypassed - show generic error instead
          showError("Unable to load workspace data. Please try again.");
        }
      } else {
        showError(errMsg || "Failed to fetch workspace.");
      }
    } finally {
      setLoading(false);
    }
  };

  const openDeleteModal = (id) => {
    setDeletingId(null);
    setSelectedWorkspaceId(id);
    setIsDeleteModalOpen(true);
  };

  const openEditModal = (workspace) => {
    setEditingWorkspace(workspace);
    setEditModalOpen(true);
  };

  const handleUpdateWorkspace = async (updatedData) => {
    if (!editingWorkspace) return;
    try {
      const payload = {
        orgId: search,
        workSpaceId: editingWorkspace.id,
        body: {
          name: updatedData.name,
          description:
            updatedData.description === "" ? null : updatedData.description,
          is_active: editingWorkspace.is_active,
        },
      };
      const response = await updateWorkSpace(payload);
      if (response.status === 200) {
        const updatedWorkspace = response.data.data;
        setListWorkSpace((prev) =>
          prev.map((ws) =>
            ws.id === updatedWorkspace.id ? updatedWorkspace : ws,
          ),
        );
        showSuccess("Workspace updated successfully");
      }
    } catch (err) {
      showError(err?.response?.data?.detail || "Failed to update workspace.");
    } finally {
      setEditModalOpen(false);
      setEditingWorkspace(null);
    }
  };

  const confirmDelete = async () => {
    if (!selectedWorkspaceId) return;
    try {
      setDeletingId(selectedWorkspaceId);
      const response = await deleteWorkSpace(search, selectedWorkspaceId);
      if (response.status === 200) {
        setListWorkSpace((prevList) =>
          prevList.filter((item) => item.id !== selectedWorkspaceId),
        );
        setTotalWorkList((prevCount) => prevCount - 1);
        showSuccess("Workspace deleted successfully");
      }
    } catch (error) {
      console.error(error);
      showError("Failed to delete workspace.");
    } finally {
      setDeletingId(null);
      setIsDeleteModalOpen(false);
      setSelectedWorkspaceId(null);
    }
  };

  const handleCreateWorkspace = () => {
    router.push(`/get-started/?oId=${userDetails.clientId}`);
  };

  // Prevent rendering until mounted and authenticated
  if (!hasMounted || !token || !userDetails) return null;

  return (
    <div className="m-8">
      <div className="rounded-lg flex justify-between items-baseline">
        <div className="font-medium text-2xl">
          Workspace
          <span className="bg-gray-300 rounded-3xl font-normal text-sm px-2 py-1 ms-2">
            {totalWorkList}
          </span>
        </div>
        <div className={"flex gap-2"}>
          <SearchBox
            value={searchText}
            onDebouncedChange={(e) => {
              setSearchText(e);
            }}
            placeholder="Search workspaces"
          ></SearchBox>
          <Button className="bg-blue-10" onClick={handleCreateWorkspace}>
            + Create Workspace
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-4 w-full gap-4 mt-6">
        {loading
          ? Array.from({ length: 8 }).map((_, index) => (
              <div key={"A" + index} className="relative">
                <WorkspaceCard KeyToChild={"A" + index} loading={true} />
              </div>
            ))
          : listWorkSpace.map((item) => (
              <div key={item?.id} className="relative">
                <WorkspaceCard
                  item={item}
                  onClick={() =>
                    router.push(`/workspace/${item?.id}/files/${0}`)
                  }
                  KeyToChild={item?.id}
                  onDelete={() => openDeleteModal(item?.id)}
                  onEdit={() => openEditModal(item)}
                  loading={isDeleteModalOpen && deletingId === item?.id}
                />
              </div>
            ))}
      </div>

      {editingWorkspace && (
        <EditWorkspaceModal
          key={editingWorkspace.id}
          open={editModalOpen}
          onClose={() => setEditModalOpen(false)}
          initialData={{
            name: editingWorkspace.name,
            description: editingWorkspace.description,
          }}
          onSubmit={handleUpdateWorkspace}
        />
      )}

      <DeleteModal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        onDelete={confirmDelete}
        title="Are you sure you want to delete this workspace?"
      />

      {pagination && (
        <Paginator
          page={pagination}
          totalPages={totalPages}
          showPageSize={false}
          onChange={(opts) => setPagination(opts)}
        />
      )}
    </div>
  );
};

export default WorkspacePage;
