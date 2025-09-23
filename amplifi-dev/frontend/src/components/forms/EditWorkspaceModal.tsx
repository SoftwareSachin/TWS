"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogOverlay,
} from "@/components/ui/dialog";
import { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { useForm } from "react-hook-form";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  WORKSPACE_NAME_MAX_LENGTH,
  DECSRIPTION_MAX_LENGTH,
  ALLOWED_NAME_REGEX,
  RESERVED_WORDS,
} from "@/lib/file-constants";

const formSchema = z.object({
  name: z
    .string()
    .min(1, "Please fill the workspace name")
    .max(WORKSPACE_NAME_MAX_LENGTH, "Name must be 25 characters or less")
    .refine((val) => ALLOWED_NAME_REGEX.test(val), {
      message:
        "Name must start with a letter and use only letters, numbers, _ or -",
    })
    .refine((val) => !RESERVED_WORDS.includes(val.toLowerCase()), {
      message: "Name cannot be a reserved word (e.g., select, workspace)",
    }),
  description: z
    .string()
    .max(DECSRIPTION_MAX_LENGTH, "Description must be 100 characters or less")
    .regex(/^[\w\s.,!?'"()-]*$/, "No excessive special characters")
    .optional()
    .nullable(),
});

type FormData = z.infer<typeof formSchema>;
interface EditWorkspaceModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (updatedData: {
    name: string;
    description?: string;
    is_active: boolean;
  }) => void;
  initialData: { name: string; description?: string };
}

export const EditWorkspaceModal: React.FC<EditWorkspaceModalProps> = ({
  open,
  onClose,
  onSubmit,
  initialData,
}) => {
  const {
    register,
    handleSubmit,
    reset,
    watch,
    formState: { errors, isSubmitting },
  } = useForm<FormData>({
    resolver: zodResolver(formSchema),
    defaultValues: initialData,
  });

  useEffect(() => {
    if (open) {
      reset(initialData);
    }
  }, [open, initialData, reset]);

  const onFormSubmit = async (data: FormData) => {
    onSubmit({
      ...data,
      description: data.description ?? undefined,
      is_active: true,
    });
    onClose();
  };

  const nameValue = watch("name") || "";
  const descriptionValue = watch("description") || "";

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogOverlay className="DialogOverlay z-50 flex items-center justify-center bg-black-20 opacity-80" />
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Workspace</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit(onFormSubmit)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium">Workspace Name</label>
            <Input
              {...register("name")}
              maxLength={WORKSPACE_NAME_MAX_LENGTH}
              placeholder="Enter your workspace name"
            />
            <p className="text-sm text-red-500 text-gray-500">
              {errors.name?.message}
            </p>
            <p className="text-sm text-right text-gray-500 mt-1">
              {nameValue.length}/{WORKSPACE_NAME_MAX_LENGTH} characters
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium">
              Description (Optional)
            </label>
            <Textarea
              {...register("description")}
              maxLength={DECSRIPTION_MAX_LENGTH}
              placeholder="Description"
            />
            <p className="text-sm text-red-500 text-gray-500">
              {errors.description?.message}
            </p>
            <p className="text-sm text-right text-gray-500 mt-1">
              {descriptionValue.length}/{DECSRIPTION_MAX_LENGTH} characters
            </p>
          </div>
          <Button
            className="w-full bg-blue-500 hover:bg-blue-600 text-white"
            isLoading={isSubmitting}
            type="submit"
          >
            Save
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
};
