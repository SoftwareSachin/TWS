/* The above code is a React component called `MultiStepForm` that implements a multi-step form
functionality. It allows users to create a new workspace in a step-by-step manner. Here is a
breakdown of the key functionalities: */
"use client";

import React, { useState, useEffect } from "react";
import FileUploadComponent from "./uploadfile";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Button } from "../ui/button";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../ui/form";
import { useSearchParams } from "next/navigation";
import { createWorkspace } from "@/api/Workspace/workspace";
import { showError, showSuccess } from "@/utils/toastUtils";
import { useUser } from "@/context_api/userContext";
import {
  WORKSPACE_NAME_MAX_LENGTH,
  DECSRIPTION_MAX_LENGTH,
  ALLOWED_NAME_REGEX,
  RESERVED_WORDS,
} from "@/lib/file-constants";

// Define the form schema
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
    .optional(),
});

const MultiStepForm = () => {
  const [step, setStep] = useState(1);
  const [workSpaceId, setWorkSpaceId] = useState("");
  const [loading, setLoading] = useState(false);
  const searchParams = useSearchParams();
  const search = searchParams.get("id");
  const sourceId = searchParams.get("sourceId");
  const { user } = useUser();

  const orgId = user?.clientId;

  useEffect(() => {
    if (search) {
      setStep(2);
      setWorkSpaceId(search);
    }
  }, []);

  const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      name: "",
      description: "",
    },
  });

  const handleNext = async (data) => {
    setLoading(true);
    try {
      const payload = {
        id: orgId,
        body: {
          ...data,
          is_active: true,
          description: data.description === "" ? null : data.description,
        },
      };
      const response = await createWorkspace(payload);
      if (response.status === 200) {
        setLoading(false);
        showSuccess(`${response.data?.message}`);
        setWorkSpaceId(response.data?.data?.id);
        if (step < 2) {
          setStep(step + 1);
        }
      }
    } catch (error) {
      setLoading(false);
      showError(`${error.response.data.detail}`);
    }
  };

  const steps = [
    { number: 1, label: "Invoice Details" },
    { number: 2, label: "Payment Information" },
  ];

  return (
    <div>
      {/* Step Indicator */}
      <ul className="steps-indicator flex justify-center items-center gap-16 mb-8 relative">
        {steps.map((s, index) => (
          <li key={s.number} className="text-center relative">
            <div
              className={`w-8 h-8 flex items-center justify-center rounded-full mb-2 text-sm ${
                step > s.number
                  ? "border border-blue-500 text-blue-500"
                  : step === s.number
                    ? "bg-blue-600 text-white border border-gray-300"
                    : "border border-gray-400 text-gray-500"
              }`}
            >
              {step > s.number ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              ) : (
                s.number
              )}
            </div>
            {index < steps.length - 1 && (
              <div className="absolute top-1/2 left-[129%] transform -translate-y-1/2 h-[1px] bg-gray-300 w-12" />
            )}
          </li>
        ))}
      </ul>

      {/* Step Forms */}
      {step === 1 && (
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit((data) => {
              handleNext(data);
            })}
          >
            <h3 className="mb-14 text-4xl font-medium text-gray-900">
              Create a new workspace
            </h3>
            <div className="grid gap-4 mb-10 sm:grid-cols-1">
              {/* Workspace Name */}
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => {
                  const charCount = field.value?.length || 0;

                  const handleChange = (e) => {
                    const value = e.target.value.slice(
                      0,
                      WORKSPACE_NAME_MAX_LENGTH,
                    );
                    field.onChange(value);
                  };

                  return (
                    <FormItem>
                      <FormLabel>Workspace Name</FormLabel>
                      <FormControl>
                        <>
                          <Input
                            placeholder="Enter your workspace name"
                            value={field.value}
                            onChange={handleChange}
                          />
                          <div className="text-sm text-right text-gray-500 mt-1">
                            {charCount}/{WORKSPACE_NAME_MAX_LENGTH} characters
                          </div>
                        </>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  );
                }}
              />

              {/* Description */}
              <FormField
                control={form.control}
                name="description"
                render={({ field }) => {
                  const charCount = field.value?.length || 0;

                  const handleChange = (e) => {
                    const value = e.target.value.slice(
                      0,
                      DECSRIPTION_MAX_LENGTH,
                    );
                    field.onChange(value);
                  };

                  return (
                    <FormItem>
                      <FormLabel>Description (Optional)</FormLabel>
                      <FormControl>
                        <>
                          <Textarea
                            id="description"
                            placeholder="Description"
                            className="h-24"
                            value={field.value}
                            onChange={handleChange}
                          />
                          <div className="text-sm text-right text-gray-500 mt-1">
                            {charCount}/{DECSRIPTION_MAX_LENGTH} characters
                          </div>
                        </>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  );
                }}
              />

              <div>
                <Button
                  type="submit"
                  className="bg-blue-500 text-white px-4 py-2 rounded w-full"
                  isLoading={loading}
                >
                  Submit
                </Button>
              </div>
            </div>
          </form>
        </Form>
      )}

      {step === 2 && (
        <Form {...form}>
          <form>
            <h3 className="mb-10 text-4xl font-medium text-gray-900">
              Upload files to connect to a data source
            </h3>
            <FileUploadComponent
              WorkSpaceId={workSpaceId}
              sourceId={sourceId}
              oId={orgId}
            />
          </form>
        </Form>
      )}
    </div>
  );
};

export default MultiStepForm;
// import React, { useState } from "react";
// import FileUploadComponent from "./uploadfile";
// import { Input } from "../ui/input";
// import { Textarea } from "../ui/textarea";
// import { Button } from "../ui/button";
// import { z } from "zod";
// import { useForm } from "react-hook-form";
// import { zodResolver } from "@hookform/resolvers/zod";
// import {
//   Form,
//   FormControl,
//   FormDescription,
//   FormField,
//   FormItem,
//   FormLabel,
//   FormMessage,
// } from "../ui/form";
// const formSchema = z.object({
//   name: z.string().min(2, { message: "Please fill the workspace name" }),
//   password: z
//     .string()
//     .min(1, { message: "Password must be at least 10 characters." }),
// });

// const MultiStepForm = () => {
//   const [step, setStep] = useState(1);
//   const form = useForm({
//     resolver: zodResolver(formSchema),
//     defaultValues: {
//       name: "",
//       password: "",
//     },
//   });
//   const handleNext = (e) => {
//     console.log(e, "checkstep", step);

//     e.preventDefault();
//     if (step < 2) {
//       setStep(step + 1);
//     }
//   };

//   const steps = [
//     { number: 1, label: "Invoice Details" },
//     { number: 2, label: "Payment Information" },
//   ];
//   async function onSubmit(values) {
//     console.log(values, "valuse");
//   }
//   return (
//     <div>
//       {/* Step Indicator */}
//       <ul className="steps-indicator flex justify-center items-center gap-16 mb-8 relative">
//         {steps.map((s, index) => (
//           <li key={s.number} className="text-center relative">
//             {/* Step Circle */}
//             <div
//               className={`w-8 h-8 flex items-center justify-center rounded-full mb-2 text-sm ${
//                 step > s.number
//                   ? "border border-blue-500 text-blue-500"
//                   : step === s.number
//                   ? "bg-blue-600 text-white border border-gray-300"
//                   : "border border-gray-400 text-gray-500"
//               }`}
//             >
//               {step > s.number ? (
//                 <svg
//                   xmlns="http://www.w3.org/2000/svg"
//                   className="h-5 w-5"
//                   fill="none"
//                   viewBox="0 0 24 24"
//                   stroke="currentColor"
//                 >
//                   <path
//                     strokeLinecap="round"
//                     strokeLinejoin="round"
//                     strokeWidth={2}
//                     d="M5 13l4 4L19 7"
//                   />
//                 </svg>
//               ) : (
//                 s.number
//               )}
//             </div>

//             {/* Connecting Line */}
//             {index < steps.length - 1 && (
//               <div className="absolute top-1/2 left-[129%] transform -translate-y-1/2 h-[1px] bg-gray-300 w-12" />
//             )}
//           </li>
//         ))}
//       </ul>

//       {/* Step Forms */}
//       {step === 1 && (
//         <Form {...form}>
//           <form onSubmit={form.handleSubmit(onSubmit)}>
//             <h3 className="mb-14 text-4xl font-medium text-gray-900">
//               Create a new workspace
//             </h3>
//             <div className="grid gap-4 mb-10 sm:grid-cols-1">
//               <div>
//                 {/* <label
//                 htmlFor="workspace-name"
//                 className="block mb-2 text-sm font-medium text-gray-900"
//               >
//                 Workspace Name
//               </label>
//               <Input
//                 type="text"
//                 id="workspace-name"
//                 placeholder="Enter your workspace name"
//                 required
//               />
//                */}
//                 <FormField
//                   control={form.control}
//                   name="name"
//                   render={({ field }) => (
//                     <FormItem>
//                       <FormLabel>Workspace Name</FormLabel>
//                       <FormControl>
//                         <Input
//                           placeholder="Enter your workspace name"
//                           {...field}
//                         />
//                       </FormControl>
//                       <FormMessage />
//                     </FormItem>
//                   )}
//                 />
//               </div>
//               <div>
//                 <label
//                   htmlFor="description"
//                   className="block mb-2 text-sm font-medium text-gray-900"
//                 >
//                   Description (Optional)
//                 </label>
//                 <Textarea
//                   id="description"
//                   placeholder="Description"
//                   className="h-24"
//                   required={false} // Optional field, so no need to mark as required
//                 />
//               </div>
//               <div>
//                 <Button className="bg-blue-500 text-white px-4 py-2 rounded w-full">
//                   Submit
//                 </Button>
//               </div>
//             </div>
//           </form>
//         </Form>
//       )}

//       {step === 2 && (
//         <form onSubmit={handleNext}>
//           <h3 className="mb-10 text-4xl font-medium text-gray-900">
//             Upload files to connect to a data source
//           </h3>
//           <FileUploadComponent />
//         </form>
//       )}
//     </div>
//   );
// };

// export default MultiStepForm;
