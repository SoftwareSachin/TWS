"use client";
import { Button } from "@/components/ui/button";
import React, { useEffect, useRef, useState } from "react";
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
import Image from "next/image";
import { ChevronDown } from "lucide-react";
import Modal from "@/components/forms/modal";
import Edituser from "@/components/forms/edituser";
import AddUserForm from "@/components/forms/addUserForm";
import Cookies from "universal-cookie";
import { addUser, deleteUser, organisationUserListData } from "@/api/Users";
import { showError, showSuccess } from "@/utils/toastUtils";
import { useUser } from "@/context_api/userContext";
import Paginator from "@/components/utility/paginator";
import { Page } from "@/types/Paginated";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";

const UsersPage = () => {
  const [addUserModal, setAddUserModal] = useState(false);
  const [editUserModal, setEditUserModal] = useState(false);
  const [deleteUserModal, setDeleteUserModal] = useState(false);
  const [userData, setUserData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedUserId, setSelectedUserId] = useState(null);
  const { user } = useUser();
  const [pagination, setPagination] = useState({ page: 1, size: 25 });
  const [totalPages, setTotalPages] = useState(1);

  const orgId = user?.clientId;

  // Store updated data for role changes
  const cookies = new Cookies();
  const hasFetched = useRef(false);

  const handleCreateUser = () => {
    setAddUserModal(true);
  };

  // const handleDelete = () => {
  //   setDeleteUserModal(true);
  // };

  const handleDelete = (userId) => {
    setSelectedUserId(userId);
    setDeleteUserModal(true);
  };

  // Role options for the custom dropdown
  const roleOptions = [
    { label: "Admin", value: "admin" },
    { label: "Editor", value: "editor" },
    { label: "Viewer", value: "viewer" },
  ];

  const handleRoleChange = (selectedRole, idx) => {
    const updatedUsers = [...userData];
    updatedUsers[idx].role.name = selectedRole.value; // Update role name
    setUserData(updatedUsers);
  };

  const authRole = cookies?.get("role");

  // Function to get status badge classes
  const getStatusBadge = (status) => {
    switch (status) {
      case "Accepted":
        return "bg-green-100 text-green-800";
      case "Invited":
        return "bg-yellow-100 text-yellow-800";
      case "Pending":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };
  const fetchAllUsers = async (page = 1) => {
    setIsLoading(true);
    try {
      console.log(page);
      console.log(pagination.size);
      const resp = await organisationUserListData(orgId, pagination); // Ensure API accepts these
      if (resp?.status === 200) {
        const { data } = resp.data;
        setUserData(data.items);
        setTotalPages(resp.data.data.total || 1); // assuming `total` is in response
      } else {
        console.error(`Unexpected response status: ${resp?.status}`);
      }
    } catch (e) {
      console.error("Error fetching users:", e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAllUsers(pagination.page);
  }, [pagination]);

  const onSaveUser = async () => {
    setAddUserModal(false);
    fetchAllUsers();
  };

  const handleDeleteUser = async (userId) => {
    try {
      // Find user details before deletion
      const userToDelete = userData.find((u) => u.id === userId);

      const response = await deleteUser(userId);
      if (response.status === 200) {
        // Track user removal event
        identifyUserFromObject(user);
        captureEvent("user_removed", {
          removed_user_id_hash: hashString(userId || ""),
          removed_user_email_hash: hashString(userToDelete?.email || ""),
          remover_id_hash: hashString(user?.clientId || ""),
          organization_id_hash: hashString(user?.clientId || ""), // Assuming organization context
          removal_type: "organization",
          user_role: userToDelete?.role || "",
          description: "User removes another user from organization",
        });

        showSuccess(`${response?.data?.message}`);
        fetchAllUsers();
        setDeleteUserModal(false);
      } else {
      }
    } catch (error) {
      setDeleteUserModal(false);
      showError(`${error?.response?.data?.detail} ||'Something went wrong'`);
    } finally {
    }
  };

  return (
    <div className="p-8">
      <div className="flex justify-between">
        <div className="font-semibold text-base">
          Users
          <span className="bg-gray-300 rounded-3xl font-normal text-sm px-2 py-1 ms-2">
            {userData?.length < 10 ? `0${userData?.length}` : userData?.length}
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
            <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
              Status
            </TableHead>
            <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
              Role
            </TableHead>
          </TableRow>
        </TableHeader>

        <TableBody>
          {isLoading
            ? // Render skeleton loader rows while loading
              Array(5)
                .fill("")
                .map((_, idx) => (
                  <TableRow
                    key={idx}
                    className="border-b-2 border-gray-300 bg-white animate-pulse"
                  >
                    <TableCell className="py-3 px-4">
                      <div className="h-4 bg-gray-300 rounded w-3/4"></div>
                    </TableCell>
                    <TableCell className="py-3 px-4">
                      <div className="h-4 bg-gray-300 rounded w-2/3"></div>
                    </TableCell>
                    <TableCell className="py-3 px-4">
                      <div className="h-4 bg-gray-300 rounded w-1/4"></div>
                    </TableCell>
                    <TableCell className="py-3 px-4">
                      <div className="h-4 bg-gray-300 rounded w-1/3"></div>
                    </TableCell>
                  </TableRow>
                ))
            : // Render actual data when not loading
              userData?.map((items, idx) => (
                <TableRow
                  className="border-b-2 border-gray-300 bg-white"
                  key={idx}
                >
                  <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                    {items?.first_name} {items?.last_name}
                  </TableCell>
                  <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                    {items?.email}
                  </TableCell>

                  {/* Status Badge */}
                  <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                    <span
                      className={`inline-block py-1 px-3 rounded-md text-xs ${getStatusBadge(
                        items?.state,
                      )}`}
                    >
                      {items?.state}
                    </span>
                  </TableCell>

                  <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm flex justify-between items-center">
                    {/* Role Dropdown */}
                    {authRole === "admin" ? (
                      <DropdownMenu>
                        <DropdownMenuTrigger className="focus:outline-none">
                          <span className="text-sm cursor-pointer flex items-center gap-4 capitalize">
                            {items?.role?.name}{" "}
                            <ChevronDown className="h-4 w-4" />
                          </span>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="center"
                          className="w-auto min-w-[150px]"
                        >
                          {roleOptions.map((option) => (
                            <DropdownMenuItem
                              key={option.value}
                              onClick={() => handleRoleChange(option, idx)}
                              className="hover:!bg-blue-100"
                            >
                              {option.label}
                            </DropdownMenuItem>
                          ))}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    ) : (
                      items?.role?.name
                    )}

                    {/* Dots for Actions (Edit/Delete) */}
                    <DropdownMenu>
                      <DropdownMenuTrigger className="focus:outline-none">
                        <Image
                          src={dots}
                          alt="options"
                          className="self-start cursor-pointer"
                        />
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="w-28">
                        {/* <DropdownMenuItem
                        onClick={() => setEditUserModal(true)}
                        className="hover:!bg-blue-100"
                      >
                        Edit
                      </DropdownMenuItem> */}
                        <DropdownMenuItem
                          onClick={() => handleDelete(items?.id)}
                          className="hover:!bg-blue-100"
                        >
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))}
        </TableBody>
      </Table>

      <Paginator
        page={pagination}
        size={"full"}
        totalPages={totalPages}
        showPageSize={true}
        onChange={(opts) => setPagination(opts)}
      ></Paginator>

      <DeleteModal
        title={"Are you sure you want to delete this?"}
        isOpen={deleteUserModal}
        onClose={() => setDeleteUserModal(false)}
        onDelete={() => handleDeleteUser(selectedUserId)}
      />

      <Modal
        isOpen={addUserModal}
        onClose={() => {
          setAddUserModal(false);
        }}
        title={"Add User"}
        size={"!max-w-2xl"}
      >
        <div className="max-h-[80vh] overflow-y-auto">
          <AddUserForm onSave={onSaveUser} setIsOpen={setAddUserModal} />
        </div>
      </Modal>
      <Modal
        isOpen={editUserModal}
        onClose={() => setEditUserModal(false)}
        title={"Add User"}
        size={"!max-w-2xl"}
      >
        {/* <Edituser setIsOpen={setEditUserModal} details={data[0]}/> */}
        <Edituser setIsOpen={setEditUserModal} />
      </Modal>
    </div>
  );
};

export default UsersPage;
