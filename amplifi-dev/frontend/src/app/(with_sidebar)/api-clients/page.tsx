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
import Image from "next/image";
import Modal from "@/components/forms/modal";
import { showError, showSuccess } from "@/utils/toastUtils";
import { useUser } from "@/context_api/userContext";
import { useRouter } from "next/navigation";
import { ROLES } from "@/lib/constants";
import Paginator from "@/components/utility/paginator";
import {
  deleteApiClient,
  getApiClientsData,
  regenerateSecret,
} from "@/api/api-client";
import HandleAPIClientForm from "@/components/forms/handleAPIClientForm";
import { ApiClientItem } from "@/types/ApiClient";
import APIClientSecretModal from "@/components/forms/apiClientSecretModal";
import ConfirmModal from "@/components/forms/confirmModal";
import { Page } from "@/types/Paginated";

const APIClients = () => {
  const [addApiClientModal, setAddApiClientModal] = useState(false);
  const [editApiClientModal, setEditApiClientModal] = useState(false);
  const [deleteApiClientModal, setDeleteApiClientModal] = useState(false);
  const [secret, setSecret] = useState("");
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [showConfirmModalLoading, setShowConfirmModalLoading] = useState(false);
  const [apiClientData, setApiClientData] = useState<ApiClientItem[]>();
  const [isLoading, setIsLoading] = useState(false);
  const [selectedApiClientId, setSelectedApiClientId] = useState("");
  const [selectedApiClientData, setSelectedApiClientData] =
    useState<ApiClientItem>();
  const { user } = useUser();
  const router = useRouter();
  const [pagination, setPagination] = useState<Page>({
    page: 1,
    size: 25,
  });
  const [totalPages, setTotalPages] = useState(1);

  const orgId = user?.clientId || "";

  const handleCreateApiClient = () => {
    setAddApiClientModal(true);
  };

  const handleDelete = (apiClientId: string) => {
    setSelectedApiClientId(apiClientId);
    setDeleteApiClientModal(true);
  };

  const handleEdit = (data: ApiClientItem) => {
    setSelectedApiClientData(data);
    setEditApiClientModal(true);
  };

  const fetchAllApiClients = async () => {
    setIsLoading(true);
    try {
      const resp = await getApiClientsData(orgId, pagination); // Ensure API accepts these
      if (resp?.status === 200) {
        const { data } = resp.data;
        setApiClientData(data.items);
        const pages = Math.ceil(resp.data.data.total / pagination.size);
        setTotalPages(pages);
      } else {
        console.error(`Unexpected response status: ${resp?.status}`);
      }
    } catch (e) {
      console.error("Error fetching api clients:", e);
    } finally {
      setIsLoading(false);
    }
  };

  // Redirect if user does not have Amplifi_Admin role
  useEffect(() => {
    const userRoles: string[] = Array.isArray(user?.roles) ? user.roles : [];
    const hasAmplifiAdmin = userRoles.some((role) => role === ROLES.ADMIN);
    if (!hasAmplifiAdmin) {
      router.replace("/");
    }
  }, [user]);

  useEffect(() => {
    fetchAllApiClients();
  }, [pagination]);

  const onSaveApiClient = (clientSecret?: string) => {
    setAddApiClientModal(false);
    setSecret(clientSecret!);
    fetchAllApiClients();
  };

  const onUpdateApiClient = () => {
    setEditApiClientModal(false);
    setSelectedApiClientData({} as ApiClientItem);
    fetchAllApiClients();
  };

  const handleDeleteApiClient = async (id: string) => {
    try {
      const response = await deleteApiClient(orgId, id);
      if (response.status === 200) {
        showSuccess(`${response?.data?.message}`);
        fetchAllApiClients();
        setDeleteApiClientModal(false);
      } else {
      }
    } catch (error: unknown) {
      setDeleteApiClientModal(false);
      if (
        error &&
        typeof error === "object" &&
        "response" in error &&
        error.response &&
        typeof error.response === "object" &&
        "data" in error.response
      ) {
        const errorMessage =
          error.response.data &&
          typeof error.response.data === "object" &&
          "message" in error.response.data
            ? (error.response.data as { message: string }).message
            : "Something went wrong";
        showError(`${errorMessage}`);
      } else if (typeof error === "string") {
        showError(error);
      } else {
        showError("Something went wrong");
      }
    } finally {
    }
  };

  const handleRegenerateSecret = async () => {
    setShowConfirmModalLoading(true);
    try {
      const resp = await regenerateSecret(orgId, selectedApiClientData!.id);
      if (resp?.status === 200) {
        showSuccess(`${resp?.data.message}`);
        setShowConfirmModal(false);
        setSecret(resp?.data?.data?.client_secret);
        setSelectedApiClientData({} as ApiClientItem);
      } else {
        console.error(`Unexpected response status: ${resp?.status}`);
      }
    } catch (e) {
      console.error("Error regenerate secret:", e);
    } finally {
      setShowConfirmModalLoading(false);
    }
  };

  const handleShowConfirm = (data: ApiClientItem) => {
    setSelectedApiClientData(data);
    setShowConfirmModal(true);
  };

  return (
    <div className="p-8">
      <div className="flex justify-between">
        <div className="font-semibold text-base">
          API Clients
          <span className="bg-gray-300 rounded-3xl font-normal text-sm px-2 py-1 ms-2">
            {apiClientData && apiClientData?.length < 10
              ? `0${apiClientData?.length}`
              : apiClientData?.length}
          </span>
        </div>
        <Button className="bg-blue-10" onClick={() => handleCreateApiClient()}>
          + Add API Client
        </Button>
      </div>
      <Table className="border border-gray-300 rounded-2xl mt-2">
        <TableHeader>
          <TableRow className="border-b-2 border-gray-300">
            <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
              Name
            </TableHead>
            <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
              Client ID
            </TableHead>
            <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
              Created At
            </TableHead>
            <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
              Expires At
            </TableHead>
            <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
              Action
            </TableHead>
          </TableRow>
        </TableHeader>

        <TableBody>
          {isLoading ? (
            // Render skeleton loader rows while loading
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
          ) : // Render actual data when not loading
          apiClientData && apiClientData!.length > 0 ? (
            apiClientData.map((items, idx) => (
              <TableRow
                className="border-b-2 border-gray-300 bg-white"
                key={idx}
              >
                <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                  {items?.name}
                </TableCell>
                <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                  {items?.client_id}
                </TableCell>
                <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                  {items?.created_at
                    ? new Date(items.created_at).toLocaleDateString()
                    : "N/A"}
                </TableCell>
                <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                  {items?.expires_at
                    ? new Date(items.expires_at).toLocaleDateString()
                    : "N/A"}
                </TableCell>
                <TableCell className="py-3 px-4 text-sm flex justify-between items-center">
                  <DropdownMenu>
                    <DropdownMenuTrigger className="focus:outline-none">
                      <Image
                        src={dots}
                        alt="options"
                        className="self-start cursor-pointer"
                      />
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-40">
                      <DropdownMenuItem
                        onClick={() => handleShowConfirm(items)}
                        className="hover:!bg-blue-100"
                      >
                        Regenerate Secret
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => handleEdit(items)}
                        className="hover:!bg-blue-100"
                      >
                        Edit
                      </DropdownMenuItem>
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
            ))
          ) : (
            <TableRow>
              <TableCell
                colSpan={5}
                className="text-center py-6 text-gray-500 text-sm"
              >
                No data found
              </TableCell>
            </TableRow>
          )}
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
        isOpen={deleteApiClientModal}
        onClose={() => setDeleteApiClientModal(false)}
        onDelete={() => handleDeleteApiClient(selectedApiClientId)}
      />

      <Modal
        isOpen={addApiClientModal}
        onClose={() => {
          setAddApiClientModal(false);
        }}
        title={"Add API Client"}
        size={"!max-w-2xl"}
      >
        <div className="max-h-[80vh] overflow-y-auto">
          <HandleAPIClientForm
            onSave={onSaveApiClient}
            setIsOpen={setAddApiClientModal}
          />
        </div>
      </Modal>
      <Modal
        isOpen={editApiClientModal}
        onClose={() => setEditApiClientModal(false)}
        title={"Edit API Client"}
        size={"!max-w-2xl"}
      >
        <HandleAPIClientForm
          onSave={onUpdateApiClient}
          setIsOpen={setEditApiClientModal}
          apiClientData={selectedApiClientData}
        />
      </Modal>
      <Modal
        isOpen={!!secret}
        onClose={() => setSecret("")}
        title={"Your Client Secret"}
        size={"!max-w-2xl"}
      >
        <APIClientSecretModal secret={secret!} close={setSecret} />
      </Modal>
      <Modal
        isOpen={showConfirmModal}
        onClose={() => setShowConfirmModal(false)}
        title={"Confirm"}
        size={"!max-w-2xl"}
      >
        <ConfirmModal
          message="Are you sure you want to regenerate your secret? Your current secret will no longer work once a new one is created."
          onConfirm={() => handleRegenerateSecret()}
          onCancel={() => setShowConfirmModal(false)}
          isLoading={showConfirmModalLoading}
        />
      </Modal>
    </div>
  );
};

export default APIClients;
