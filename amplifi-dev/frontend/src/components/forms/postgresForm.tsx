import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { sourceConnector } from "@/api/Workspace/workspace";
import { useRouter } from "next/navigation";
import { PostgresFormProps } from "@/types/props/SourceConnectorProps";
import { PGSourceConnector, PostgresFormSchema } from "@/types/SourceConnector";
import { editSource } from "@/api/Workspace/WorkSpaceFiles";
import { showError, showSuccess } from "@/utils/toastUtils";
import { constants } from "@/lib/constants";

export const PostgresForm: React.FC<PostgresFormProps> = ({
  workSpaceId,
  source,
  sourceType,
}) => {
  const router = useRouter();
  const [sourceId, setSourceId] = useState(null);
  const [loading, setLoading] = useState(false);
  const form = useForm<PGSourceConnector>({
    resolver: zodResolver(PostgresFormSchema),
    defaultValues: {
      database_name: source?.database_name || "",
      host: source?.host || "",
      port: source?.port || "5432",
      username: "",
      password: "",
      ssl_mode: "disabled",
    },
  });

  useEffect(() => {
    if (sourceType === constants.SOURCE_TYPE.MYSQL) {
      form.register("ssl_mode");
    } else {
      form.unregister("ssl_mode");
    }
  }, [sourceType]);

  const onSubmit = async (data: Partial<PGSourceConnector>) => {
    setLoading(true);
    const payload = {
      id: workSpaceId,
      body: { source_type: sourceType, ...data },
    };
    try {
      const response = source
        ? await editSource(workSpaceId, source.id, payload.body)
        : await sourceConnector(payload);

      if (response.status === 200) {
        showSuccess(`${response?.data?.message}`);
        setSourceId(response?.data?.data?.id);
        router.push(`/workspace/${workSpaceId}/files/0`); // Navigate to the workspace page
      }
    } catch (error: any) {
      showError(`${error.response?.data?.detail}`);
    } finally {
      setLoading(false);
    }
  };
  return (
    <>
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(onSubmit)}
          className="space-y-4 flex flex-wrap"
        >
          <div className="w-full px-4">
            <FormField
              control={form.control}
              name="database_name"
              render={({ field }: any) => (
                <FormItem>
                  <FormLabel>Database Name</FormLabel>
                  <FormControl>
                    <Input placeholder="Enter the database name" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <div className="w-1/2 px-4">
            <FormField
              control={form.control}
              name="host"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Host Name</FormLabel>
                  <FormControl>
                    <Input placeholder="Host name" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <div className="w-1/2 px-4">
            <FormField
              control={form.control}
              name="port"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Port</FormLabel>
                  <FormControl>
                    <Input
                      type={"number"}
                      placeholder="Port Number"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <div className="w-1/2 px-4">
            <FormField
              control={form.control}
              name="username"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Username</FormLabel>
                  <FormControl>
                    <Input placeholder="Enter User name" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <div className="w-1/2 px-4">
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Password</FormLabel>
                  <FormControl>
                    <div className={"relative"}>
                      <Input
                        type={"password"}
                        placeholder="Enter password"
                        {...field}
                      />
                    </div>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          {sourceType === constants.SOURCE_TYPE.MYSQL && (
            <div className="w-full px-4">
              <FormField
                control={form.control}
                name="ssl_mode"
                render={({ field }) => (
                  <FormItem>
                    <div className="flex items-center space-x-2">
                      <FormControl>
                        <input
                          type="checkbox"
                          id="ssl_mode"
                          checked={field.value === "required"}
                          onChange={(e) =>
                            field.onChange(
                              e.target.checked ? "required" : "disabled",
                            )
                          }
                          className="h-4 w-4 rounded border border-gray-300 bg-white checked:bg-blue-500 checked:border-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                        />
                      </FormControl>
                      <FormLabel
                        htmlFor="ssl_mode"
                        className="text-sm font-medium cursor-pointer"
                      >
                        Add SSL Mode
                      </FormLabel>
                    </div>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          )}
          <div className="border-t flex justify-end gap-4 p-4">
            {/*<Button type="button">Test Connection</Button>*/}
            <Button
              isLoading={loading}
              type="submit"
              className="bg-blue-500 text-white px-4 py-2 rounded text-sm"
            >
              Next
            </Button>
          </div>
        </form>
      </Form>
    </>
  );
};
