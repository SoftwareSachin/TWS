"use client";

import React, { useState } from "react";
import { useForm, useFieldArray } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import Image from "next/image";
import { retryTrainSqlDataset, trainSqlDataset } from "@/api/dataset";
import { showError, showSuccess } from "@/utils/toastUtils";
import { useRouter } from "next/navigation";

const formSchema = z.object({
  documentation: z.string().min(1, "Documentation is required"),
  queries: z.array(
    z.object({
      exampleQuery: z.string().min(1, "Example query is required"),
      sql: z.string().min(1, "SQL query is required"),
    }),
  ),
});

export const SqlTrainingForm = ({
  setIsOpen,
  dataSetId,
  workspaceId,
  datasetId,
  onClose,
  setNewDataAdded,
  newDataAdded,
  submitButton = "Start Training",
  initialTrainingData = [],
}) => {
  const getDefaultValues = () => {
    const { documentation = "", question_sql_pairs } =
      initialTrainingData || {};
    return {
      documentation,
      queries: question_sql_pairs?.map(({ question = "", sql = "" }) => ({
        exampleQuery: question,
        sql,
      })) || [{ exampleQuery: "", sql: "" }],
    };
  };

  const {
    register,
    control,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: getDefaultValues(),
  });

  const { fields, append, remove } = useFieldArray({
    control,
    name: "queries",
  });

  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const onSubmit = async (data) => {
    try {
      setLoading(true);
      console.log("Submitted Data:", data);
      const body = {
        documentation: data.documentation,
        question_sql_pairs: data.queries.map((pair) => {
          return { question: pair.exampleQuery, sql: pair.sql };
        }),
      };

      let response;

      // Conditionally call different APIs based on button text
      if (submitButton === "Start Training") {
        response = await trainSqlDataset({
          workspaceId,
          dataSetId,
          body,
        });
      } else if (submitButton === "Retry Training") {
        response = await retryTrainSqlDataset({
          workspaceId,
          dataSetId,
          body,
        });
      } else {
        // Default fallback to trainSqlDataset
        response = await trainSqlDataset({
          workspaceId,
          dataSetId,
          body,
        });
      }

      if (response?.status === 200) {
        showSuccess(`${response?.data?.message}`);
        setNewDataAdded(!newDataAdded);
        onClose();
        router.push(`/workspace/${workspaceId}/datasets/${dataSetId}`);
      }
    } catch (err) {
      console.error("API Error:", err);
      showError("Failed to train on selected datasource");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full px-4">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        {/* Documentation Field */}
        <div>
          <label className="block text-sm font-medium">Add Documentation</label>
          <textarea
            {...register("documentation")}
            className="w-full border rounded-lg p-2"
            placeholder="Add the details of the schema of the database, including information on columns."
          />
          {errors?.documentation && (
            <p className="text-red-500 text-xs">
              {errors?.documentation.message}
            </p>
          )}
        </div>

        {/* Dynamic Example Queries */}
        <div className={"max-h-[60vh] overflow-y-auto"}>
          {fields.map((field, index) => (
            <div
              key={field.id}
              className="border flex text-sm w-full items-center gap-2 p-3 rounded-lg"
            >
              <div className={"flex flex-1 flex-col"}>
                <label className="block text-sm font-medium">
                  Example Query
                </label>
                <textarea
                  {...register(`queries.${index}.exampleQuery`)}
                  className="w-full border rounded-lg p-2 h-20 text-sm"
                  placeholder="What is the month on month sales of all products in New York?"
                />
                {errors?.queries?.[index]?.exampleQuery && (
                  <p className="text-red-500 text-xs">
                    {errors?.queries[index]?.exampleQuery?.message}
                  </p>
                )}
              </div>

              <div className={"flex flex-1 flex-col"}>
                <label className="block text-sm font-medium">SQL</label>
                <textarea
                  {...register(`queries.${index}.sql`)}
                  className="w-full border rounded-lg p-2 h-20 text-sm"
                  placeholder="SELECT date, sum(value) FROM sales WHERE city = 'New York'
GROUP BY date_trunc('month', date)"
                />
                {errors?.queries?.[index]?.sql && (
                  <p className="text-red-500 text-xs">
                    {errors?.queries[index]?.sql?.message}
                  </p>
                )}
              </div>

              {/* Remove Query Button */}
              {fields.length > 1 && (
                <Image
                  className={"cursor-pointer"}
                  onClick={() => remove(index)}
                  src={"/assets/icons/remove.svg"}
                  alt={"Remove"}
                  height={"24"}
                  width={"24"}
                />
              )}
            </div>
          ))}
        </div>

        {/* Add More Queries */}
        <button
          type="button"
          onClick={() => append({ exampleQuery: "", sql: "" })}
          className="text-blue-500 text-sm"
        >
          + Add more
        </button>

        {/* Buttons */}
        <div className="flex justify-end space-x-3 my-4">
          <button
            type="button"
            className="border px-4 py-2 rounded text-sm"
            onClick={() => setIsOpen(false)}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg"
          >
            {loading ? "Submitting..." : submitButton}
          </button>
        </div>
      </form>
    </div>
  );
};
