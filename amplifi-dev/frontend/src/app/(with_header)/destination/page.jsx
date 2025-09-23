/* The above code is a React component called `DestinationPage` that serves as a page for managing
destinations. Here is a summary of what the code is doing: */

"use client";
import React, { useEffect, useState } from "react";
import searchIcon from "@/assets/icons/search-icon.svg";
import DetailCard from "@/components/destination/detailCard";
import { Button } from "@/components/ui/button";
import NoDataScreen from "@/components/empty-screens/noData";
import Image from "next/image";
import emptyscreen from "@/assets/images/empty-screens/destination-empty-screen.svg";
import Modal from "@/components/forms/modal";
import CreateDestinationForm from "@/components/forms/createDestinationForm";
import ViewDestinationForm from "@/components/forms/viewDestinationForm";
import DeleteModal from "@/components/forms/deleteModal";
import { deleteDestination, destinationCardData } from "@/api/destination";
import { useSearchParams } from "next/navigation";
import { showError, showSuccess } from "@/utils/toastUtils";
import { decodeToken } from "@/components/utility/decodeJwtToken";
import { getCookie } from "@/utils/cookieHelper";
import { constants } from "@/lib/constants";

const DestinationPage = () => {
  const [data, setData] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isDelete, setIsDelete] = useState(false);
  const [openViewModal, setOpenViewModal] = useState(false);
  const [openForm, setOpenForm] = useState(false);
  const [selectedDestination, setSelectedDestination] = useState(null);
  const [loading, setLoading] = useState(true);
  const searchParams = useSearchParams();
  const [organizationId, setOrganizationId] = useState(null);
  const [totalPages, setTotalPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  if (!organizationId) {
    const token = getCookie(constants.JWT_TOKEN);
    const userDetails = decodeToken(token);
    setOrganizationId(userDetails.clientId);
  }

  // Fetch the organization_id from the URL search params
  useEffect(() => {
    const id = searchParams.get("id");
    setOrganizationId(id);
  }, [searchParams]);

  // Open and Close modal handlers
  const openModal = () => setIsOpen(true);
  const closeModal = () => setIsOpen(false);

  // Fetch destination data from API
  const fetchDestinationData = async () => {
    setLoading(true);

    try {
      if (!organizationId) {
        console.error("organization_id is required");
        return;
      }

      const resp = await destinationCardData(organizationId);
      if (resp.status === 200) {
        setLoading(false);
        setData(resp?.data?.data?.items);
        setTotalPages(resp?.data?.data?.pages);
      }
    } catch (error) {
      setLoading(false);
      console.error("Error fetching destination data:", error);
    }
  };

  useEffect(() => {
    if (organizationId) {
      fetchDestinationData();
    }
  }, [organizationId, currentPage]);

  const handleDelete = async () => {
    try {
      if (!selectedDestination || !organizationId) {
        showError("Something went wrong please try again");
        setIsDelete(false);
        return;
      }

      await deleteDestination(organizationId, selectedDestination);
      showSuccess(`Destination deleted successfully`);
      setIsDelete(false);
      await fetchDestinationData();
    } catch (error) {
      showError(`${error?.response?.data?.detail}`);
    }
  };

  const handlePageChange = (page) => {
    if (page > 0 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  return (
    <>
      <div className="m-8">
        <div className="flex justify-between">
          <div className="font-medium text-2xl">
            Destinations
            <span className="bg-gray-300 rounded-3xl font-normal text-sm px-2 py-1 ms-2">
              {data?.length < 10 ? `0${data?.length}` : data?.length}
            </span>
          </div>
          {data?.length > 0 && (
            <div className="rounded-lg flex gap-2">
              <span className="bg-white px-2 flex items-center gap-2 rounded-lg border border-gray-400">
                <Image src={searchIcon} alt="search icon" />
                <input
                  placeholder="Search here"
                  className="bg-white outline-none"
                />
              </span>
              <Button className="bg-blue-10" onClick={openModal}>
                + Create Destination
              </Button>
            </div>
          )}
        </div>
        {loading ? (
          <div className="grid grid-cols-4 w-full gap-4 mt-4">
            {Array.from({ length: 8 }).map((_, index) => (
              <div key={"A" + index}>
                <DetailCard loading={true} />
              </div>
            ))}
          </div>
        ) : data?.length > 0 ? (
          <>
            <div className="grid grid-cols-4 w-full gap-4 mt-4">
              {data?.map((item) => (
                <div key={item?.id}>
                  <DetailCard
                    item={item}
                    onClick={() => {
                      setOpenViewModal(true);
                      setSelectedDestination(item);
                    }}
                    onEdit={() => console.log("editing")}
                    onDelete={() => {
                      setSelectedDestination(item.id); // Set the ID of the destination to be deleted
                      setIsDelete(true); // Open the delete confirmation modal
                    }}
                  />
                </div>
              ))}
            </div>

            {totalPages > 1 && (
              <div className="flex justify-center items-right gap-2 mt-6">
                <button
                  onClick={() => handlePageChange(currentPage - 1)}
                  className={`w-10 h-10 flex items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:bg-gray-200 transition-all ${
                    currentPage === 1 ? "opacity-50 cursor-not-allowed" : ""
                  }`}
                  disabled={currentPage === 1}
                >
                  {"<"}
                </button>
                {Array.from({ length: totalPages }).map((_, index) => {
                  const page = index + 1;
                  return (
                    <button
                      key={page}
                      onClick={() => handlePageChange(page)}
                      className={`w-10 h-10 flex items-center justify-center rounded-full text-gray-700 border border-gray-300 hover:bg-blue-100 transition-all ${
                        currentPage === page ? "bg-blue-500 text-white" : ""
                      }`}
                    >
                      {page}
                    </button>
                  );
                })}
                <button
                  onClick={() => handlePageChange(currentPage + 1)}
                  className={`w-10 h-10 flex items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:bg-gray-200 transition-all ${
                    currentPage === totalPages
                      ? "opacity-50 cursor-not-allowed"
                      : ""
                  }`}
                  disabled={currentPage === totalPages}
                >
                  {">"}
                </button>
              </div>
            )}
          </>
        ) : (
          <NoDataScreen
            title="No Destinations Yet"
            subtitle="You haven’t added any destinations yet. Start by creating your first destination!"
            buttonText="Create Destination"
            image={emptyscreen}
            onClick={() => setOpenForm(true)}
          />
        )}

        {/* Modal Components */}
        <Modal
          isOpen={isOpen}
          onClose={closeModal}
          title="Connect to Databricks vector search"
        >
          {/* <CreateDestinationForm /> */}
          <CreateDestinationForm
            onSuccess={() => {
              closeModal(); // Close the modal
              fetchDestinationData(); // Refresh the destination list
            }}
          />
        </Modal>

        <Modal
          isOpen={openViewModal}
          onClose={() => setOpenViewModal(false)}
          title={`${selectedDestination?.name}`}
        >
          <ViewDestinationForm data={selectedDestination} />
        </Modal>

        <DeleteModal
          title="Are you sure you want to delete this file?"
          isOpen={isDelete}
          onClose={() => setIsDelete(false)}
          onDelete={handleDelete}
        />
      </div>

      <Modal
        isOpen={openForm}
        onClose={() => setOpenForm(false)}
        title="Create Destination"
      >
        <CreateDestinationForm
          onSuccess={() => {
            closeModal(); // Close the modal
            fetchDestinationData(); // Refresh the destination list
          }}
        />
      </Modal>
    </>
  );
};

export default DestinationPage;
