import { useState } from "react";
import { constants } from "@/lib/constants";

export default function ToggleAppSwitch({ onToggle }) {
  const [isSqlApp, setIsSqlApp] = useState(false);
  const handleToggle = () => {
    const newValue = !isSqlApp;
    setIsSqlApp(newValue);
    onToggle(
      newValue ? constants.SQL_CHAT_APP : constants.UNSTRUCTURED_CHAT_APP,
    );
  };
  return (
    <div
      onClick={handleToggle}
      className="relative w-80 h-10 flex items-center bg-gray-200 rounded-xl p-1 cursor-pointer"
    >
      {/* Sliding Background */}
      <div
        className={`absolute top-1 bottom-1 w-1/2 bg-blue-500 rounded-lg transition-all duration-300 ${
          !isSqlApp ? "left-1" : "left-1/2"
        }`}
      ></div>

      {/* SQL Chat App (Left) */}
      <span
        className={`flex-1 text-center text-sm font-medium relative z-10 transition-all ${
          !isSqlApp ? "text-white" : "text-gray-500"
        }`}
      >
        Unstructured Chat App
      </span>

      {/* Unstructured Chat App (Right) */}
      <span
        className={`flex-1 text-center text-sm font-medium relative z-10 transition-all ${
          isSqlApp ? "text-white" : "text-gray-500"
        }`}
      >
        SQL Chat App
      </span>
    </div>
  );
}
