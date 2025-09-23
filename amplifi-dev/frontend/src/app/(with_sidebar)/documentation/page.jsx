import React from "react";

const DocumentationPage = () => {
  return (
    <div className={"h-[85vh] w-100"}>
      <iframe
        src="https://dev-docs.dataamplifi.com/"
        width="100%"
        height="100%"
        style={{ border: "none" }}
      />
    </div>
  );
};

export default DocumentationPage;
