import React from "react";
const ChatNotFoundPage = () => {
  return (
    <div
      style={{ minHeight: "100vh" }}
      className="flex flex-col justify-center items-center text-center px-4"
    >
      <h1 className="text-4xl font-bold mb-4">Chat App Not Found</h1>
      <p className="text-gray-600 text-lg">
        The chat you are trying to access does not exist or was removed.
      </p>
    </div>
  );
};

export default ChatNotFoundPage;
