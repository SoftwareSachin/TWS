"use client";
import { Button } from "@/components/ui/button";
import React, { useEffect, useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import dots from "@/assets/icons/dots-vertical.svg";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import DeleteModal from "@/components/forms/deleteModal";
import DrawerVertical from "@/components/forms/drawervertical";
import UserFormWorkspace from "@/components/forms/userFormWorkspace";
import Image from "next/image";
import { useParams } from "next/navigation";
import { useUser } from "@/context_api/userContext";
import { showError, showSuccess } from "@/utils/toastUtils";
import {
  removeUsersFromWorkspace,
  getUsersFromWorkspace,
} from "@/api/Workspace/index";
import Paginator from "@/components/utility/paginator";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";

const UsersPage = () => {
  const [isDelete, setIsDelete] = useState(false);
  const [isOpen, setIsOpen] = React.useState(false);
  const params = useParams();
  const [selectedUserIDs, setSelectedUserIDs] = useState([]);
  const [existingUserIDs, setExistingUserIDs] = useState([]);
  const { user } = useUser();
  const orgId = user?.clientId;
  const [userList, setUserList] = useState([]);
  const [loader, setLoader] = useState(false);
  const [pagination, setPagination] = useState({ page: 1, size: 10 });
  const [totalPages, setTotalPages] = useState(1);
  const [newDataAdded, setNewDataAdded] = useState(false);
  const fetchUsersFromWorkspace = async () => {
    const data = {
      orgId: orgId,
      id: params?.workspaceId,
    };

    try {
      setLoader(true);
      const response = await getUsersFromWorkspace(data, pagination);

      if (response.status === 200) {
        setUserList(response.data?.data?.items);
        const itemIDs = response.data?.data?.items.map((item) => item.id);
        setExistingUserIDs(response.data?.data?.items.map((item) => item.id));
        //setUserList(userData);
        setTotalPages(response.data.data.total || 1);
        setLoader(false);
      }
    } catch (error) {
      showError(`${error?.response?.data?.detail}`);
    }
  };
  const handleCreateUser = () => {
    setIsOpen(true);
  };
  const handleDelete = async () => {
    const data = {
      orgId: orgId,
      id: params?.workspaceId,
      body: { user_ids: selectedUserIDs },
    };
    try {
      const response = await removeUsersFromWorkspace(data);

      if (response.status === 200) {
        // Track user removal events for workspace
        identifyUserFromObject(user);

        // Track each user removal
        selectedUserIDs.forEach((userId) => {
          const removedUser = userList.find((u) => u.id === userId);
          captureEvent("user_removed", {
            removed_user_id_hash: hashString(userId || ""),
            removed_user_email_hash: hashString(removedUser?.email || ""),
            remover_id_hash: hashString(user?.clientId || ""),
            workspace_id_hash: hashString(params?.workspaceId || ""),
            organization_id_hash: hashString(orgId || ""),
            removal_type: "workspace",
            user_count: selectedUserIDs.length,
            user_role: removedUser?.role || "",
            description: "User removes users from workspace",
          });
        });

        setNewDataAdded(!newDataAdded);
        showSuccess(`${response?.data?.message}`);
        setIsDelete(false);
        setSelectedUserIDs([]);
      }
    } catch (error) {
      showError(`${error?.response?.data?.detail}`);
    }
  };
  useEffect(() => {
    fetchUsersFromWorkspace();
  }, [newDataAdded, pagination]);
  return (
    <>
      <div className="p-8">
        <div className="flex justify-between">
          <div className="font-semibold text-base">
            Users
            <span className="bg-gray-300 rounded-3xl font-normal text-sm px-2 py-1 ms-2">
              {userList?.length < 10
                ? `0${userList?.length}`
                : userList?.length}
            </span>
          </div>
          <Button className="bg-blue-10" onClick={() => handleCreateUser()}>
            + Add Users
          </Button>
        </div>
        <Table className="border border-gray-300 rounded-2xl mt-2">
          <TableHeader>
            <TableRow className="border-b-2 border-gray-300">
              <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
                Name
              </TableHead>
              <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
                Email
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {userList?.length > 0 ? (
              userList?.map((items, idx) => (
                <TableRow
                  className="border-b-2 border-gray-300 bg-white"
                  key={idx}
                >
                  <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                    {items.full_name}
                  </TableCell>
                  <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap flex justify-between items-center text-ellipsis text-sm">
                    {items.email}
                    <DropdownMenu>
                      <DropdownMenuTrigger className="focus:outline-none">
                        <Image
                          src={dots}
                          alt="options"
                          className="self-start cursor-pointer"
                        />
                      </DropdownMenuTrigger>
                      <DropdownMenuContent
                        align="start"
                        className="w-28 absolute"
                      >
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            setIsDelete(true);
                            setSelectedUserIDs((prev) => [...prev, items.id]);
                          }}
                          className="hover:!bg-blue-100"
                        >
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow className="border-b-2 border-gray-300 bg-white">
                <TableCell
                  colSpan={4}
                  className="py-3 px-4 text-center text-gray-500"
                >
                  No User found.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        {pagination && (
          <Paginator
            page={pagination}
            size={"full"}
            totalPages={totalPages}
            showPageSize={true}
            onChange={(opts) => setPagination(opts)}
          ></Paginator>
        )}
      </div>
      <DeleteModal
        title={"Are you sure you want to delete this user?"}
        isOpen={isDelete}
        onClose={() => setIsDelete(false)}
        onDelete={() => handleDelete()}
      />
      <DrawerVertical
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title="Add users"
      >
        <UserFormWorkspace
          setIsOpen={setIsOpen}
          existingUserIDs={existingUserIDs}
          workspaceId={params?.workspaceId}
          organisationId={orgId}
          setNewDataAdded={setNewDataAdded}
          newDataAdded={newDataAdded}
        />
      </DrawerVertical>
    </>
  );
};

export default UsersPage;
