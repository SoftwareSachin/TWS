// this is the root file for / root which is currently not in use
import { redirect } from "next/navigation";
import { headers } from "next/headers"; // Import headers from next/headers

export async function generateMetadata() {
  // Use next/headers to access the request headers
  const requestHeaders = await headers();
  const hostname = requestHeaders.get("host"); // Extract the 'host' header
  if (process.env.NEXT_PUBLIC_CHAT_HOST_NAME && process.env.NEXT_PUBLIC_CHAT_HOST_NAME.includes(hostname)) {
    // If the hostname matches, redirect to the chatapp page
    redirect(`/chatapp`);
  } else {
    // Otherwise, redirect to workspace
    redirect(`/workspace`);
  }

  // Return null, as the redirect happens before rendering
  return null;
}

export default function Home() {
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]"></div>
  );
}
