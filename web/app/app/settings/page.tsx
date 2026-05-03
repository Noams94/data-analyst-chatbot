"use client";

import { useEffect, useState } from "react";
import { Check, Eye, EyeOff, Loader2, Save } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { useApi, type UserSettings } from "@/lib/api";

const ANTHROPIC_MODELS = [
  "claude-opus-4-7",
  "claude-sonnet-4-6",
  "claude-haiku-4-5",
  "claude-opus-4-5",
  "claude-3-5-sonnet-20241022",
  "claude-3-haiku-20240307",
];

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium">{label}</label>
      {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
      {children}
    </div>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mt-6 mb-3">
      {children}
    </h2>
  );
}

export default function SettingsPage() {
  const api = useApi();
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Local form state
  const [provider, setProvider] = useState<"anthropic" | "ollama">("anthropic");
  const [anthropicModel, setAnthropicModel] = useState("claude-sonnet-4-6");
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [ollamaModel, setOllamaModel] = useState("llama3.2");
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");

  useEffect(() => {
    api.getSettings()
      .then((s) => {
        setSettings(s);
        setProvider(s.provider);
        setAnthropicModel(s.anthropicModel);
        setOllamaModel(s.ollamaModel);
        setOllamaUrl(s.ollamaBaseUrl);
      })
      .catch(() => toast.error("Failed to load settings"))
      .finally(() => setLoading(false));
  }, [api]);

  const handleSave = async () => {
    setSaving(true);
    try {
      const updates: Record<string, string> = {
        provider,
        anthropicModel,
        ollamaModel,
        ollamaBaseUrl: ollamaUrl,
      };
      // Only send the key if the user typed something (not the masked placeholder)
      if (apiKeyInput && !apiKeyInput.includes("***")) {
        updates.anthropicApiKey = apiKeyInput;
      }
      const updated = await api.patchSettings(updates);
      setSettings(updated);
      setApiKeyInput(""); // clear after save
      setSaved(true);
      toast.success("Settings saved");
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      toast.error((e as Error).message || "Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center flex-1 text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading…
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-8 max-w-2xl mx-auto w-full">
      <h1 className="text-2xl font-semibold mb-1">Settings</h1>
      <p className="text-sm text-muted-foreground mb-6">
        Configure AI provider, model, and credentials.
      </p>

      <Card className="p-6 space-y-5">
        {/* Provider selector */}
        <SectionHeading>Provider</SectionHeading>
        <Field label="AI Provider" hint="Which backend to use for chat and analysis.">
          <div className="flex gap-3">
            {(["anthropic", "ollama"] as const).map((p) => (
              <button
                key={p}
                onClick={() => setProvider(p)}
                className={`flex-1 rounded-md border py-2 text-sm font-medium transition-colors ${
                  provider === p
                    ? "border-primary bg-primary/5 text-foreground"
                    : "border-muted text-muted-foreground hover:border-foreground/30"
                }`}
              >
                {p === "anthropic" ? "Anthropic (Claude)" : "Ollama (local)"}
              </button>
            ))}
          </div>
        </Field>

        {/* Anthropic section */}
        {provider === "anthropic" ? (
          <>
            <SectionHeading>Anthropic</SectionHeading>
            <Field
              label="API Key"
              hint={
                settings?.hasAnthropicKey
                  ? `Currently set (${settings.anthropicApiKey}). Leave blank to keep unchanged.`
                  : "Enter your Anthropic API key. Falls back to the ANTHROPIC_API_KEY env var."
              }
            >
              <div className="relative">
                <Input
                  type={showKey ? "text" : "password"}
                  placeholder={settings?.hasAnthropicKey ? "Leave blank to keep current key" : "sk-ant-..."}
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                  className="pr-10 font-mono text-sm"
                />
                <button
                  type="button"
                  onClick={() => setShowKey((v) => !v)}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </Field>

            <Field label="Model">
              <div className="flex gap-2 flex-wrap">
                {ANTHROPIC_MODELS.map((m) => (
                  <button
                    key={m}
                    onClick={() => setAnthropicModel(m)}
                    className={`rounded-md border px-3 py-1.5 text-xs font-mono transition-colors ${
                      anthropicModel === m
                        ? "border-primary bg-primary/5 text-foreground"
                        : "border-muted text-muted-foreground hover:border-foreground/30"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
              <Input
                placeholder="Or enter a custom model ID…"
                value={ANTHROPIC_MODELS.includes(anthropicModel) ? "" : anthropicModel}
                onChange={(e) => setAnthropicModel(e.target.value || ANTHROPIC_MODELS[1])}
                className="mt-2 font-mono text-sm"
              />
            </Field>
          </>
        ) : (
          <>
            <SectionHeading>Ollama</SectionHeading>
            <Field label="Base URL" hint="URL where Ollama is running.">
              <Input
                value={ollamaUrl}
                onChange={(e) => setOllamaUrl(e.target.value)}
                placeholder="http://localhost:11434"
                className="font-mono text-sm"
              />
            </Field>
            <Field label="Model" hint="Must be pulled locally via `ollama pull <model>`.">
              <Input
                value={ollamaModel}
                onChange={(e) => setOllamaModel(e.target.value)}
                placeholder="llama3.2"
                className="font-mono text-sm"
              />
            </Field>
          </>
        )}
      </Card>

      <div className="mt-4 flex justify-end">
        <Button onClick={handleSave} disabled={saving} className="gap-2 min-w-24">
          {saving ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : saved ? (
            <Check className="h-4 w-4" />
          ) : (
            <Save className="h-4 w-4" />
          )}
          {saving ? "Saving…" : saved ? "Saved" : "Save"}
        </Button>
      </div>
    </div>
  );
}
