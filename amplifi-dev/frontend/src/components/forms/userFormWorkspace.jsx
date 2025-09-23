"use client";
import React, { useEffect, useState } from "react";

import { organisationUserListData } from "@/api/Users";
import { addUsersInWorkspace } from "@/api/Workspace";
import { showError, showSuccess } from "@/utils/toastUtils";
import Paginator from "@/components/utility/paginator"; // Assumed paginator component
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const UserFormWorkspace = ({
  setIsOpen,
  existingUserIDs,
  workspaceId,
  organisationId,
  setNewDataAdded,
  newDataAdded,
}) => {
  const { user } = useUser();
  const [userList, setUserList] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedUserID, setSelectedUserID] = useState(existingUserIDs || []);
  const [pagination, setPagination] = useState({ page: 1, size: 10 });
  const [totalPages, setTotalPages] = useState(1);

  const handleCancel = () => {
    setIsOpen(false);
  };

  const fetchAllUsers = async () => {
    setIsLoading(true);
    try {
      const resp = await organisationUserListData(organisationId, pagination);
      if (resp?.status === 200) {
        setUserList(resp.data.data.items);
        setTotalPages(resp.data.data.total || 1);
      } else {
        console.error(`Unexpected response status: ${resp?.status}`);
      }
    } catch (e) {
      console.error("Error fetching users:", e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCheckboxChange = (id, checked) => {
    setSelectedUserID((prevState) =>
      checked ? [...prevState, id] : prevState.filter((user) => user !== id),
    );
  };

  const handleAddUsers = async () => {
    try {
      const newUserIds = selectedUserID.filter(
        (item) => !existingUserIDs.includes(item),
      );

      const data = {
        orgId: organisationId,
        id: workspaceId,
        body: {
          user_ids: newUserIds,
        },
      };

      const resp = await addUsersInWorkspace(data);
      if (resp?.status === 200) {
        // Track user invitation events for workspace
        identifyUserFromObject(user);

        // Track each user invitation
        newUserIds.forEach((userId) => {
          const invitedUser = userList.find((u) => u.id === userId);
          captureEvent("user_invited", {
            invited_user_email_hash: hashString(invitedUser?.email || ""),
            invited_user_id_hash: hashString(userId || ""),
            inviter_id_hash: hashString(user?.clientId || ""),
            workspace_id_hash: hashString(workspaceId || ""),
            organization_id_hash: hashString(organisationId || ""),
            invitation_type: "workspace",
            user_count: newUserIds.length,
            description: "User invites users to workspace",
          });
        });

        showSuccess(`${resp.data.message}`);
        setNewDataAdded(!newDataAdded);
      } else {
        console.error(`Unexpected response status: ${resp?.status}`);
      }
    } catch (e) {
      console.error("Error adding users:", e);
    }
    setIsOpen(false);
  };

  useEffect(() => {
    fetchAllUsers();
  }, [pagination]);

  return (
    <>
      <div
        style={{ "max-height": "calc(100vh - 160px)" }}
        className="flex flex-col gap-4 p-4 max-h-[calc(100vh - 160px)] overflow-y-auto"
      >
        {userList?.map((item, index) => (
          <div key={index}>
            <div className="flex gap-2 items-center">
              <input
                type="checkbox"
                className="w-4 h-4"
                disabled={existingUserIDs.includes(item.id)}
                checked={selectedUserID.includes(item.id)}
                onChange={(e) =>
                  handleCheckboxChange(item.id, e.target.checked)
                }
              />
              <div className="font-medium text-sm">
                {item?.first_name + " " + item?.last_name}
              </div>
            </div>
            <div className="text-xs text-gray-400 font-normal mt-2 ms-6">
              {item?.email}
            </div>
          </div>
        ))}
      </div>

      <Paginator
        page={pagination}
        totalPages={totalPages}
        onChange={(opts) => setPagination(opts)}
        size="small"
        showPageSize={true}
      />

      <div className="flex gap-4 mt-4 fixed bottom-0 left-0 right-0 bg-white p-4 border-t text-sm font-medium justify-end">
        <button className="border py-1 px-3 rounded" onClick={handleCancel}>
          Cancel
        </button>
        <button
          onClick={handleAddUsers}
          className="bg-blue-10 text-white py-1 px-3 rounded"
          disabled={selectedUserID?.length - existingUserIDs?.length === 0}
        >
          Add {selectedUserID?.length - existingUserIDs?.length} users
        </button>
      </div>
    </>
  );
};

export default UserFormWorkspace;
