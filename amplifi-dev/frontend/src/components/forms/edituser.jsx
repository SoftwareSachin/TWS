import React, { useState } from "react";
import { Input } from "../ui/input";
import Select from "react-select";
import { userSelectCustomStyles } from "../styles/selectStyles";

const Edituser = ({ setIsOpen, details }) => {
  const [fullName, setFullName] = useState(
    details?.first_name + " " + details?.last_name,
  );
  const [email, setEmail] = useState(details?.email);
  const roleOptions = [
    { value: "Admin", label: "Admin" },
    { value: "Editor", label: "Editor" },
    { value: "Viewer", label: "Viewer" },
  ];
  const [role, setRole] = useState({
    value: details?.role?.name,
    label: details?.role?.name,
  });

  const handleSave = () => {
    const userData = { fullName, email, role: role.value };
  };

  return (
    <form className="space-y-4 pt-4">
      <div className="grid grid-cols-3 gap-4 p-4 pt-0">
        <div className="col-span-1">
          <label className="block text-sm font-medium">Full name</label>
          <Input
            type="text"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            placeholder="Enter name"
            className="h-10 px-2 shadow-none focus-visible:outline-none border-gray-30" // Set height and padding to match select
          />
        </div>
        <div className="col-span-1">
          <label className="block text-sm font-medium">Email</label>
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value.toLowerCase())}
            placeholder="Enter email"
            className="h-10 px-2 shadow-none focus:outline-none border-gray-30" // Set height and padding to match select
          />
        </div>
        <div className="col-span-1">
          <label className="block text-sm font-medium">Role</label>
          <Select
            options={roleOptions}
            value={role}
            onChange={(selectedOption) => setRole(selectedOption)}
            placeholder="Select role"
            styles={userSelectCustomStyles}
            components={{
              IndicatorSeparator: () => null,
            }}
            className="capitalize"
          />
        </div>
      </div>
      <div className="flex justify-end space-x-2 p-4 border-t">
        <button
          type="button"
          onClick={() => setIsOpen(false)}
          className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={handleSave}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Save
        </button>
      </div>
    </form>
  );
};

export default Edituser;
