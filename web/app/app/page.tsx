"use client";

/**
 * Landing inside the app shell.
 *
 * Shows the dropzone for uploading a new dataset, plus a "Recent" section
 * if there's existing data. Once upload completes we navigate to the
 * dataset detail page so the user sees the preview before chatting.
 */

import { useRouter } from "next/navigation";
import { UploadDropzone } from "@/components/upload-dropzone";

export default function AppHome() {
  const router = useRouter();

  return (
    <main className="flex flex-1 items-center justify-center px-6">
      <UploadDropzone
        onUploaded={(ds) => router.push(`/app/datasets/${ds.id}`)}
      />
    </main>
  );
}
