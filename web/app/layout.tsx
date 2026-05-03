import type { Metadata } from "next";
import { Heebo, Geist_Mono } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";
import "./globals.css";
import "highlight.js/styles/github.css";
import { Toaster } from "@/components/ui/sonner";

const heebo = Heebo({
  variable: "--font-sans",
  subsets: ["latin", "hebrew"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Data Analyst Chatbot",
  description: "Chat with your data — upload a CSV and ask Claude.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ClerkProvider>
      <html
        lang="en"
        className={`${heebo.variable} ${geistMono.variable} h-full antialiased`}
      >
        <body className="min-h-full flex flex-col" suppressHydrationWarning>
          {children}
          <Toaster richColors />
        </body>
      </html>
    </ClerkProvider>
  );
}
