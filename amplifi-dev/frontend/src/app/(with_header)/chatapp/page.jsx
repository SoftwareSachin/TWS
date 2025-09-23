/* The provided code is a React component named `ChatAppPage`. Here is a breakdown of what the code
is doing: */
"use client";
import { getLoggedInUserChatApps } from "@/api/chatApp";
import ChatAppCard from "@/components/chat-app/chatAppCard";
import { useRouter } from "next/navigation";
import React, { useEffect, useState } from "react";
const ChatAppPage = () => {
  const [listChatApp, setListChatApp] = useState([]);
  const [totalWorkList, setTotalWorkList] = useState(0);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  const [totalPages, setTotalPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    getChatAppList();
  }, [currentPage]);

  const getChatAppList = async () => {
    const data = {
      page: currentPage,
      size: 8,
    };
    try {
      setLoading(true);
      const response = await getLoggedInUserChatApps(data);
      setListChatApp(response?.data?.data?.items || []);
      setTotalWorkList(response?.data?.data?.total || 0);
      setTotalPages(response?.data?.data?.pages || 0);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (page) => {
    if (page > 0 && page <= totalPages) {
      setCurrentPage(page);
    }
  };
  return (
    <div className="m-8">
      <div className="rounded-lg flex justify-between">
        <div className="font-medium text-2xl">
          Chat Apps
          <span className="bg-gray-300 rounded-3xl font-normal text-sm px-2 py-1 ms-2">
            {totalWorkList}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-4 w-full gap-4 mt-4">
        {/* Render Skeleton UI when loading */}
        {loading
          ? Array.from({ length: 8 }).map((_, index) => (
              <div key={"A" + index}>
                <ChatAppCard KeyToChild={"A" + index} loading={true} />
              </div>
            ))
          : listChatApp.map((item) => (
              <div key={item?.id}>
                <ChatAppCard
                  item={item}
                  onClick={() => router.push(`/ChatApp/${item?.id}`)}
                  KeyToChild={item?.id}
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
              currentPage === totalPages ? "opacity-50 cursor-not-allowed" : ""
            }`}
            disabled={currentPage === totalPages}
          >
            {">"}
          </button>
        </div>
      )}
    </div>
  );
};

export default ChatAppPage;
