import React, { useEffect, useState } from "react";
import { Input } from "@/components/ui/input"; // Adjust import path as needed
import { createDestination, getDestinationStatus } from "@/api/destination";
import { useSearchParams } from "next/navigation"; // Import useSearchParams hook
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useForm, Controller } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { showError, showSuccess } from "@/utils/toastUtils";
import { Button } from "../ui/button";

// Zod validation schema
const formSchema = z
  .object({
    name: z.string().min(1, "Name is required"),
    description: z.string().min(1, "Description is required"),
    is_active: z.boolean(),
    pg_vector: z
      .object({
        host: z.string().min(1, "Host is required"),
        // port: z.number().positive("Port must be a positive number").int(),
        database_name: z.string().min(1, "Database Name is required"),
        username: z.string().min(1, "Username is required"),
        password: z.string().min(1, "Password is required"),
        table_name: z.string().min(1, "Table name is required"),
      })
      .optional(),
    databricks: z
      .object({
        workspace_url: z.string().min(1, "Workspace URL is required"),
        token: z.string().min(1, "Token is required"),
        cluster_id: z.string().min(1, "Cluster id is required"),
        database_name: z.string().min(1, "Database Name is required"),
        table_name: z.string().min(1, "Table Name is required"),
      })
      .optional(),
  })
  .refine(
    (data) => {
      // Ensure that only one of pg_vector or databricks is provided
      const pgVectorProvided = data.pg_vector !== undefined;
      const databricksProvided = data.databricks !== undefined;

      return pgVectorProvided !== databricksProvided; // True if exactly one is provided
    },
    {
      message: "You must provide either pg_vector or databricks, but not both.",
      path: ["pg_vector", "databricks"], // Set the path for the error
    },
  );

// Component
const CreateDestinationForm = ({ onSuccess }) => {
  const searchParams = useSearchParams();
  const [organizationId, setOrganizationId] = useState(null);
  const [testConnectionEnable, setTestConnectionEnable] = useState(false);
  const [destinationId, setDestinationId] = useState(null);
  const [apiLoading, setApiLoading] = useState(false);
  // Fetch the organization_id from the URL search params
  useEffect(() => {
    const id = searchParams.get("id");
    setOrganizationId(id);
  }, [searchParams]);

  const [selectedVector, setSelectedVector] = useState("");

  const {
    control,
    handleSubmit,
    formState: { errors, isSubmitting },
    setValue,
    getValues,
    watch,
  } = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      name: "",
      description: "",
      is_active: false,
      pg_vector: {
        host: "",
        port: 0,
        database_name: "",
        username: "",
        password: "",
        table_name: "",
      },
      databricks: {
        workspace_url: "",
        token: "",
        cluster_id: "",
        database_name: "",
        table_name: "",
      },
    },
  });
  const watchSelectedVector = watch("pg_vector") || watch("databricks");

  useEffect(() => {
    if (selectedVector === "pg-vector") {
      setValue("pg_vector", {
        host: "",
        port: 0,
        database_name: "",
        username: "",
        password: "",
      });
      setValue("databricks", undefined); // Clear databricks if pg_vector is selected
    } else if (selectedVector === "databricks") {
      setValue("databricks", {
        workspace_url: "",
        token: "",
        cluster_id: "",
        database_name: "",
        table_name: "",
      });
      setValue("pg_vector", undefined); // Clear pg_vector if databricks is selected
    }
  }, [selectedVector, setValue]);

  const getStatus = async () => {
    setApiLoading(true);

    try {
      const connectionStatus = await getDestinationStatus(
        organizationId,
        destinationId,
      );
      connectionStatus && setApiLoading(false);
      if (connectionStatus?.data?.data?.status === "success") {
        showSuccess(`${connectionStatus?.data?.message}`);

        // close modal
        onSuccess();
        // Optionally reset form
        setValue("name", "");
        setValue("description", "");
        setValue("is_active", false);
        setValue("pg_vector", {
          host: "",
          port: 0,
          database_name: "",
          username: "",
          password: "",
        });
        setValue("databricks", {
          workspace_url: "",
          token: "",
          cluster_id: "",
          database_name: "",
          table_name: "",
        });
      }
    } catch (error) {
      showError(`${error?.response?.data?.detail}`);
      setTestConnectionEnable(false);
    } finally {
      setApiLoading(false);
    }
  };

  const onSubmit = async (data) => {
    setApiLoading(true);
    try {
      const response = await createDestination(organizationId, data);
      if (response.status === 200) {
        showSuccess(`${response?.data?.message}`);
        setTestConnectionEnable(true);
        setDestinationId(response?.data?.data?.id);
      }
    } catch (err) {
      // console.error("Error creating destination:", err);
      const errMessage =
        err?.response?.data?.detail || "error creating destination";
      showError(errMessage);
    } finally {
      setApiLoading(false);
    }
  };

  return (
    <form
      className="p-4 space-y-4 max-h-[80vh] overflow-y-auto"
      onSubmit={handleSubmit(onSubmit)}
    >
      <div>
        <label
          htmlFor="name"
          className="block mb-2 text-sm font-medium text-gray-900"
        >
          Name
        </label>
        <Controller
          control={control}
          name="name"
          render={({ field }) => (
            <Input
              disabled={testConnectionEnable}
              type="text"
              id="name"
              placeholder="Destination Name"
              {...field}
            />
          )}
        />
        {errors.name && (
          <div className="text-red-600 text-sm mt-2">{errors.name.message}</div>
        )}
      </div>

      <div>
        <label
          htmlFor="description"
          className="block mb-2 text-sm font-medium text-gray-900"
        >
          Description
        </label>
        <Controller
          control={control}
          name="description"
          render={({ field }) => (
            <Input
              disabled={testConnectionEnable}
              type="text"
              id="description"
              placeholder="Description"
              {...field}
            />
          )}
        />
        {errors.description && (
          <div className="text-red-600 text-sm mt-2">
            {errors.description.message}
          </div>
        )}
      </div>

      <div>
        <label
          htmlFor="is_active"
          className="block mb-2 text-sm font-medium text-gray-900"
        >
          Is Active
        </label>
        <button
          disabled={testConnectionEnable}
          type="button"
          className={`px-4 py-2 w-20 rounded text-sm ${
            getValues("is_active")
              ? "bg-green-500 text-white"
              : "bg-gray-300 text-black"
          }`}
          onClick={() => setValue("is_active", !getValues("is_active"))}
        >
          {getValues("is_active") ? "Active" : "Inactive"}
        </button>
      </div>

      <div>
        <label
          htmlFor="vector"
          className="block mb-2 text-sm font-medium text-gray-900"
        >
          Select Vector
        </label>
        <Select onValueChange={setSelectedVector}>
          <SelectTrigger>
            <SelectValue placeholder="Select a Vector" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectItem value="pg-vector">PG Vector</SelectItem>
              <SelectItem value="databricks">
                Databricks Vector Search
              </SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>

      {/* Render PG Vector fields conditionally */}
      {selectedVector === "pg-vector" && (
        <div className="space-y-4">
          <div>
            <label
              htmlFor="pg_vector.host"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Host
            </label>
            <Controller
              control={control}
              name="pg_vector.host"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="text"
                  id="pg_vector.host"
                  placeholder="Host"
                  {...field}
                />
              )}
            />
            {errors.pg_vector?.host && (
              <div className="text-red-600 text-sm mt-2">
                {errors.pg_vector.host.message}
              </div>
            )}
          </div>

          <div>
            <label
              htmlFor="pg_vector.port"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Port
            </label>
            <Controller
              control={control}
              name="pg_vector.port"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="number"
                  id="pg_vector.port"
                  placeholder="Port"
                  {...field}
                />
              )}
            />
            {errors.pg_vector?.port && (
              <div className="text-red-600 text-sm mt-2">
                {errors.pg_vector.port.message}
              </div>
            )}
          </div>

          <div>
            <label
              htmlFor="pg_vector.database_name"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Database Name
            </label>
            <Controller
              control={control}
              name="pg_vector.database_name"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="text"
                  id="pg_vector.database_name"
                  placeholder="Database Name"
                  {...field}
                />
              )}
            />
            {errors.pg_vector?.database_name && (
              <div className="text-red-600 text-sm mt-2">
                {errors.pg_vector.database_name.message}
              </div>
            )}
          </div>

          <div>
            <label
              htmlFor="pg_vector.username"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Username
            </label>
            <Controller
              control={control}
              name="pg_vector.username"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="text"
                  id="pg_vector.username"
                  placeholder="Username"
                  {...field}
                />
              )}
            />
            {errors.pg_vector?.username && (
              <div className="text-red-600 text-sm mt-2">
                {errors.pg_vector.username.message}
              </div>
            )}
          </div>

          <div>
            <label
              htmlFor="pg_vector.password"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Password
            </label>
            <Controller
              control={control}
              name="pg_vector.password"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="password"
                  id="pg_vector.password"
                  placeholder="Password"
                  {...field}
                />
              )}
            />
            {errors.pg_vector?.password && (
              <div className="text-red-600 text-sm mt-2">
                {errors.pg_vector.password.message}
              </div>
            )}
          </div>
          <div>
            <label
              htmlFor="pg_vector.table_name"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Table Name
            </label>
            <Controller
              control={control}
              name="pg_vector.table_name"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="text"
                  id="pg_vector.table_name"
                  placeholder="Table name"
                  {...field}
                />
              )}
            />
            {errors.pg_vector?.table_name && (
              <div className="text-red-600 text-sm mt-2">
                {errors.pg_vector.table_name.message}
              </div>
            )}
          </div>
        </div>
      )}
      {/* Render databricks Vector fields conditionally */}
      {selectedVector === "databricks" && (
        <div className="space-y-4">
          <div>
            <label
              htmlFor="databricks.workspace_url"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Workspace URL
            </label>
            <Controller
              control={control}
              name="databricks.workspace_url"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="text"
                  id="databricks.workspace_url"
                  placeholder="Workspace URL"
                  {...field}
                />
              )}
            />
            {errors.databricks?.workspace_url && (
              <div className="text-red-600 text-sm mt-2">
                {errors.databricks.workspace_url.message}
              </div>
            )}
          </div>

          <div>
            <label
              htmlFor="databricks.token"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Token
            </label>
            <Controller
              control={control}
              name="databricks.token"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="text"
                  id="databricks.token"
                  placeholder="Token"
                  {...field}
                />
              )}
            />
            {errors.databricks?.token && (
              <div className="text-red-600 text-sm mt-2">
                {errors.databricks.token.message}
              </div>
            )}
          </div>

          <div>
            <label
              htmlFor="databricks.cluster_id"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Cluster ID
            </label>
            <Controller
              control={control}
              name="databricks.cluster_id"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="text"
                  id="databricks.cluster_id"
                  placeholder="Cluster ID"
                  {...field}
                />
              )}
            />
            {errors.databricks?.cluster_id && (
              <div className="text-red-600 text-sm mt-2">
                {errors.databricks.cluster_id.message}
              </div>
            )}
          </div>

          <div>
            <label
              htmlFor="databricks.database_name"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Database Name
            </label>
            <Controller
              control={control}
              name="databricks.database_name"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="text"
                  id="databricks.database_name"
                  placeholder="Database Name"
                  {...field}
                />
              )}
            />
            {errors.databricks?.database_name && (
              <div className="text-red-600 text-sm mt-2">
                {errors.databricks.database_name.message}
              </div>
            )}
          </div>

          <div>
            <label
              htmlFor="databricks.table_name"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Table Name
            </label>
            <Controller
              control={control}
              name="databricks.table_name"
              render={({ field }) => (
                <Input
                  disabled={testConnectionEnable}
                  type="text"
                  id="databricks.table_name"
                  placeholder="Table Name"
                  {...field}
                />
              )}
            />
            {errors.databricks?.table_name && (
              <div className="text-red-600 text-sm mt-2">
                {errors.databricks.table_name.message}
              </div>
            )}
          </div>
        </div>
      )}
      <div className="w-full">
        {testConnectionEnable ? (
          <Button
            type="button"
            className="bg-blue-500 text-white px-4 py-2 rounded text-sm align-right"
            isLoading={apiLoading}
            onClick={getStatus}
          >
            Test connection
          </Button>
        ) : (
          <Button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded text-sm align-right"
            isLoading={apiLoading}
            // Removed sensitive form schema logging for security
          >
            Create
          </Button>
        )}
      </div>
    </form>
  );
};

export default CreateDestinationForm;
