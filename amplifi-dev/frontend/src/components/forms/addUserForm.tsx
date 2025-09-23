import React, { useEffect, useState } from "react";
import { useForm, useFieldArray, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import Select from "react-select";
import { Input } from "../ui/input";
import { Plus, CircleMinus } from "lucide-react";
import { getOrganization } from "@/api/common";
import { ROLES } from "@/lib/constants";
import { userSelectCustomStyles } from "../styles/selectStyles";
import { AddUsersFormSchema, AddUsersFormValues, Users } from "@/types/Users";
import { Button } from "@/components/ui/button";
import { addUser } from "@/api/Users";
import { showError, showSuccess } from "@/utils/toastUtils";
import { AddUserFormProps } from "@/types/props/UserProps";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const roleOptions = [
  { value: ROLES.ADMIN, label: "Admin" },
  { value: ROLES.DEVELOPER, label: "Developer" },
  { value: ROLES.MEMBER, label: "Member" },
];

const AddUserForm: React.FC<AddUserFormProps> = ({ onSave, setIsOpen }) => {
  const { user } = useUser();
  const [orgList, setOrgList] = useState<{ value: string; label: string }[]>(
    [],
  );
  const [loading, setLoading] = useState<boolean>(false);

  const handleSaveUser = async (users: Users[]) => {
    setLoading(true);
    try {
      const userToInvite = users.shift();
      const response = await addUser(userToInvite);
      if (response.status === 200) {
        // Track user invitation event
        identifyUserFromObject(user);
        captureEvent("user_invited", {
          invited_user_email_hash: hashString(userToInvite?.email || ""),
          inviter_id_hash: hashString(user?.clientId || ""),
          organization_id_hash: hashString(userToInvite?.organization_id || ""),
          role: userToInvite?.role || "",
          invitation_type: "organization",
          description: "User invites another user to organization",
        });

        showSuccess(`${response?.data?.message}`);
        onSave();
      }
    } catch (error: any) {
      showError(`${error?.response?.data?.message}`);
      console.error("response-- in add user", error);
    } finally {
      setLoading(false);
    }
  };

  const {
    control,
    register,
    handleSubmit,
    setValue,
    getValues,
    formState: { errors },
  } = useForm<AddUsersFormValues>({
    resolver: zodResolver(AddUsersFormSchema),
    defaultValues: {
      users: [
        {
          first_name: "",
          last_name: "",
          email: "",
          role: "",
          organization_id: "",
        },
      ],
    },
  });

  const { fields, append, remove } = useFieldArray({
    control,
    name: "users",
  });

  useEffect(() => {
    const fetchOrgs = async () => {
      const response = await getOrganization();
      const formatted = response?.data?.data?.items?.map((org: any) => ({
        label: org.name,
        value: org.id,
      }));
      setOrgList(formatted || []);
    };
    fetchOrgs();
  }, []);

  const onSubmit = (data: AddUsersFormValues) => {
    handleSaveUser(data.users);
  };

  return (
    <form className="space-y-4 pt-4" onSubmit={handleSubmit(onSubmit)}>
      {fields.map((field, index) => (
        <div
          key={field.id}
          className="grid grid-cols-3 gap-4 items-center px-4"
        >
          <div>
            <label className="block text-sm font-medium">First Name</label>
            <Input {...register(`users.${index}.first_name`)} />
            <p className="text-xs text-red-600">
              {errors.users?.[index]?.first_name?.message}
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium">Last Name</label>
            <Input {...register(`users.${index}.last_name`)} />
            <p className="text-xs text-red-600">
              {errors.users?.[index]?.last_name?.message}
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium">Email</label>
            <Input type="email" {...register(`users.${index}.email`)} />
            <p className="text-xs text-red-600">
              {errors.users?.[index]?.email?.message}
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium">Organization</label>

            <Controller
              control={control}
              name={`users.${index}.organization_id`}
              rules={{ required: "Organization is required" }}
              render={({ field, fieldState }) => (
                <>
                  <Select
                    {...field}
                    options={orgList}
                    styles={userSelectCustomStyles}
                    value={orgList.find((opt) => opt.value === field.value)}
                    onChange={(selected) =>
                      field.onChange(selected?.value || "")
                    }
                  />
                  {fieldState.error && (
                    <p className="text-xs text-red-600">
                      {fieldState.error.message}
                    </p>
                  )}
                </>
              )}
            />
          </div>

          <div className="flex col-span-1 items-end space-x-2">
            <div className="w-full">
              <label className="block text-sm font-medium">Role</label>
              <Controller
                control={control}
                name={`users.${index}.role`}
                rules={{ required: "Role is required" }}
                render={({ field, fieldState }) => (
                  <>
                    <Select
                      {...field}
                      options={roleOptions}
                      styles={userSelectCustomStyles}
                      value={roleOptions.find(
                        (opt) => opt.value === field.value,
                      )}
                      onChange={(selected) =>
                        field.onChange(selected?.value || "")
                      }
                    />
                    {fieldState.error && (
                      <p className="text-xs text-red-600">
                        {fieldState.error.message}
                      </p>
                    )}
                  </>
                )}
              />
            </div>
            {fields.length > 1 && (
              <button
                type="button"
                onClick={() => remove(index)}
                className="mb-3 text-red-600"
              >
                <CircleMinus size={20} />
              </button>
            )}
          </div>
        </div>
      ))}

      <div
        className="p-4 pt-0 font-medium text-xs text-blue-10 flex justify-start items-center cursor-pointer"
        onClick={() =>
          append({
            first_name: "",
            last_name: "",
            email: "",
            role: "",
            organization_id: "",
          })
        }
      >
        <Plus size={12} className="me-1" />
        Add More
      </div>

      <div className="flex justify-end space-x-2 p-4 border-t">
        <button
          type="button"
          onClick={() => setIsOpen(false)}
          className="px-4 py-2 border rounded-md"
        >
          Cancel
        </button>
        <Button
          isLoading={loading}
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded-md"
        >
          Send Invite
        </Button>
      </div>
    </form>
  );
};

export default AddUserForm;
