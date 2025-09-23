import React from "react";
import AdminPanelLayout from "../../components/admin-panel/admin-panel-layout";
import Navbar from "../../components/admin-panel/navbar-header";

const layout = ({ children }) => {
  return (
    <>
      <div className="">
        <Navbar />
      </div>
      <div className="mt-20 p-4 sm:p-4 lg:p-6">
        <AdminPanelLayout>{children}</AdminPanelLayout>
      </div>
    </>
  );
};

export default layout;
