// Lets `node --test` import the report route itself (R16-T3): TypeScript and JSX are transpiled
// with the application's own `typescript`, extensionless relative imports are resolved the way
// the bundler resolves them, and `next/headers` — which only works inside a Next.js request — is
// replaced by `next-headers.mjs` beside this file. Nothing else is substituted: React, `next/link`
// and every module under `app/` are the real ones.

import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const NEXT_HEADERS_STUB = new URL("./next-headers.mjs", import.meta.url).href;

async function exists(url) {
  try {
    return (await stat(fileURLToPath(url))).isFile();
  } catch {
    return false;
  }
}

export async function resolve(specifier, context, nextResolve) {
  if (specifier === "next/headers") {
    return { url: NEXT_HEADERS_STUB, shortCircuit: true };
  }

  const fromSource = context.parentURL?.match(/\.tsx?$/);
  const relative = specifier.startsWith("./") || specifier.startsWith("../");

  if (fromSource && relative && !/\.[cm]?[jt]sx?$/.test(specifier)) {
    for (const extension of [".ts", ".tsx"]) {
      const candidate = new URL(specifier + extension, context.parentURL);
      if (await exists(candidate)) {
        return { url: candidate.href, shortCircuit: true };
      }
    }
  }

  try {
    return await nextResolve(specifier, context);
  } catch (error) {
    // `next` publishes its entry points (`next/link`, `next/navigation`) as files without an
    // `exports` map, which the bundler resolves and node's ESM resolver does not.
    if (error?.code === "ERR_MODULE_NOT_FOUND" && specifier.startsWith("next/")) {
      return nextResolve(`${specifier}.js`, context);
    }
    throw error;
  }
}

export async function load(url, context, nextLoad) {
  if (!/\.tsx?$/.test(url) || url.includes("/node_modules/")) {
    return nextLoad(url, context);
  }

  const source = await readFile(fileURLToPath(url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    fileName: fileURLToPath(url),
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
    },
  });

  return { format: "module", source: outputText, shortCircuit: true };
}

