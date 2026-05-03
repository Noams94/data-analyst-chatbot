import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <main className="flex flex-1 items-center justify-center px-6">
      <div className="flex flex-col items-center gap-8 text-center max-w-2xl py-16">
        <h1 className="text-5xl font-bold tracking-tight">
          Chat with your data.
        </h1>
        <p className="text-lg text-muted-foreground">
          Upload a CSV and ask Claude. Real numbers from real analysis —
          no hallucinated stats.
        </p>
        <div className="flex gap-4">
          <Button size="lg">
            <Link href="/app">Open the app</Link>
          </Button>
          <Button variant="outline" size="lg">
            <Link href="/sign-in">Sign in</Link>
          </Button>
        </div>
      </div>
    </main>
  );
}
