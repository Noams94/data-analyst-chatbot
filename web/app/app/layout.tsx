import { UserButton } from "@clerk/nextjs";
import Link from "next/link";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-1 flex-col">
      <header className="flex items-center justify-between border-b px-6 py-3">
        <Link href="/app" className="font-semibold text-lg">
          Data Analyst
        </Link>
        <UserButton />
      </header>
      <div className="flex flex-1">{children}</div>
    </div>
  );
}
