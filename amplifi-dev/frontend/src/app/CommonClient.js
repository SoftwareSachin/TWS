/**
 * The CommonClient function renders a ToastContainer component from the react-toastify library in a
 * React application.
 * @returns The CommonClient component is being returned, which includes the ToastContainer component
 * from the react-toastify library.
 */
"use client";

import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const CommonClient = () => {
  return (
    <>
      <ToastContainer
        position="bottom-right"
        autoClose={3000}
        closeOnClick
        pauseOnHover
        draggable
        toastClassName={() => "bg-transparent shadow-none p-0 m-0"}
        bodyClassName={() => "p-0 m-0"}
      />
    </>
  );
};

export default CommonClient;
