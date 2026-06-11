/** Tiny shared i18n helpers for the bilingual (en/he) UI. */

export type UiLang = "en" | "he";

/** True when the text contains characters from the Hebrew block (U+0590–U+05FF). */
export function detectHebrew(text: string): boolean {
  return /[֐-׿]/.test(text);
}

/** Returns a `t(en, he)` picker bound to the given language. */
export function makeT(lang: UiLang) {
  return (en: string, he: string) => (lang === "he" ? he : en);
}
