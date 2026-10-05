// Every English sentence the forensic report can print (R16-T3), gathered from where it is
// written rather than from a render: a render shows the branches its fixture reaches, and this is
// meant to find the sentence nobody thought to render.
//
// Two sources. The report route is read with the application's own `typescript` parser, and
// every string that reaches `<Copy>` — its text, its `text` attribute, the `title` and `subtitle`
// of a section, the `label` of a field — is collected. The shared vocabulary in `app/analysis.ts`
// is imported and walked: its tables directly, and its rationales through `riskRationale` for
// every ruleset and rule this build knows. Requires `./tsx-loader.mjs` to be registered.

import { readFileSync } from "node:fs";

import ts from "typescript";

const REPORT = new URL("../../app/app/report/[id]/page.tsx", import.meta.url);
const ANALYSIS = new URL("../../app/analysis.ts", import.meta.url);

// The attributes whose string is drawn through `<Copy>` by the component that receives it.
const COPY_ATTRIBUTES = {
  Copy: ["text"],
  Section: ["title", "subtitle"],
  Field: ["label"],
  Contribution: ["detector"],
};

// The functions and tables in the report whose returned strings it draws through `<Copy>`.
const COPY_DECLARATIONS = [
  "NO_DECISION",
  "CONTRIBUTION_LABELS",
  "contributionDetector",
  "storedStatusNote",
];

const ENTITIES = { "&apos;": "'", "&quot;": '"', "&amp;": "&", "&lt;": "<", "&gt;": ">" };

/** JSX text as React receives it: lines trimmed, blank lines dropped, joined by one space. */
function jsxText(raw) {
  return raw
    .split("\n")
    .map((line, index, lines) => {
      let value = line.replace(/\t/g, " ");
      if (index > 0) value = value.trimStart();
      if (index < lines.length - 1) value = value.trimEnd();
      return value;
    })
    .filter((line) => line.length > 0)
    .join(" ")
    .replace(/&[a-z]+;/g, (entity) => ENTITIES[entity] ?? entity);
}

/** Every string literal under `node`, as the sentence it spells. */
function literals(node, into) {
  const visit = (child) => {
    if (ts.isConditionalExpression(child)) {
      // The branches are what is drawn; the condition is a comparison.
      visit(child.whenTrue);
      visit(child.whenFalse);
      return;
    }
    if (ts.isBinaryExpression(child) && child.operatorToken.kind !== ts.SyntaxKind.PlusToken) {
      // `status === "SUCCESS"`, `a ?? "fallback"`: only the fallback side can be drawn.
      if (child.operatorToken.kind === ts.SyntaxKind.QuestionQuestionToken) visit(child.right);
      return;
    }
    if (ts.isStringLiteral(child) || ts.isNoSubstitutionTemplateLiteral(child)) {
      into.add(child.text);
    } else if (ts.isTemplateExpression(child)) {
      // `No ${what} signal …` — resolved below from the values `what` is given.
      into.add(child.getText().slice(1, -1));
    }
    ts.forEachChild(child, visit);
  };
  visit(node);
}

function tagName(node) {
  return node.tagName.getText();
}

export function reportPageCopy() {
  const source = ts.createSourceFile(
    "page.tsx",
    readFileSync(REPORT, "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const found = new Set();
  const noSignalWhat = new Set();

  const visit = (node) => {
    if (ts.isJsxElement(node) && tagName(node.openingElement) === "Copy") {
      for (const child of node.children) {
        if (ts.isJsxText(child)) {
          const text = jsxText(child.text);
          if (text) found.add(text);
        } else if (ts.isJsxExpression(child) && child.expression) {
          literals(child.expression, found);
        }
      }
    }

    if (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) {
      const name = tagName(node);
      for (const attribute of node.attributes.properties) {
        if (!ts.isJsxAttribute(attribute) || attribute.initializer === undefined) continue;
        const key = attribute.name.getText();
        if (COPY_ATTRIBUTES[name]?.includes(key)) {
          literals(attribute.initializer, found);
        }
        if (name === "NoSignal" && key === "what") {
          literals(attribute.initializer, noSignalWhat);
        }
      }
    }

    if (
      (ts.isVariableDeclaration(node) || ts.isFunctionDeclaration(node)) &&
      COPY_DECLARATIONS.includes(node.name?.getText())
    ) {
      // Only what these return or hold is copy; a `case` label or a comparison is not.
      const visitValues = (child) => {
        if (ts.isVariableDeclaration(child) && child === node && child.initializer && ts.isStringLiteral(child.initializer)) {
          found.add(child.initializer.text);
        } else if (ts.isReturnStatement(child) && child.expression) {
          literals(child.expression, found);
        } else if (ts.isPropertyAssignment(child)) {
          literals(child.initializer, found);
        } else {
          ts.forEachChild(child, visitValues);
        }
      };
      visitValues(node);
    }

    ts.forEachChild(node, visit);
  };
  visit(source);

  // The empty-panel sentence is written once with its detector as a slot; it is printed once
  // per detector, so each of those sentences is what the dictionary has to hold.
  const sentences = new Set();
  for (const text of found) {
    if (text.includes("${what}")) {
      for (const what of noSignalWhat) sentences.add(text.replace("${what}", what));
    } else if (!text.includes("${")) {
      sentences.add(text);
    }
  }
  return sentences;
}

const RULESETS = ["p7-v1.0.0", "r4-v2.0.0", "r5-v3.0.0", "r7-v4.0.0"];
const RULE_IDS = ["R100", "R101", "R102", "R103", "R200", "R201", "R010", "R012"];

export async function reportVocabularyCopy() {
  const analysis = await import(ANALYSIS.href);
  const found = new Set();
  const add = (value) => {
    if (typeof value === "string") found.add(value);
    else if (value && typeof value === "object") Object.values(value).forEach(add);
  };

  add(analysis.PROVENANCE_WORDING);
  add(analysis.V5_VERDICT_WORDING);
  add(analysis.RISK_LABELS);
  add(analysis.UNSUPPORTED);
  add(analysis.RISK_CONDITION_LABELS);
  add(analysis.RISK_CONDITION_DETAILS);
  add(analysis.RISK_CONDITION_UNINTERPRETABLE);
  add(analysis.DEEP_EVIDENCE_WORDING);
  add(analysis.DEEP_EVIDENCE_UNINTERPRETABLE);
  add(analysis.ENRICHMENT_SNAPSHOT_NOTICE);
  add(analysis.COMPONENT_STATE_LABELS);
  add(analysis.COMPONENT_STATE_READING_LABELS);
  add(analysis.COMPONENT_STATE_UNINTERPRETABLE);
  add(analysis.SIGNAL_ABSENT_TERMINAL);
  add(analysis.SIGNAL_ABSENT_STILL_TO_ANSWER);

  for (const ruleset of RULESETS) {
    for (const rule of RULE_IDS) {
      const rationale = analysis.riskRationale(ruleset, rule);
      if (rationale !== null) {
        add(rationale.summary);
        add(rationale.coverage);
        for (const detector of ["syntheticVideo", "faceManipulation", "mouthDynamics"]) {
          add(rationale[detector].detail);
        }
      }
    }
  }

  // The functions that answer a known code with a sentence and anything else with the code.
  for (const reason of [
    "no_reading",
    "detector_did_not_report",
    "uncalibrated_deployment",
    "unreadable_figures",
    "threshold_unresolved",
  ]) {
    add(analysis.unavailableReasonText(reason));
  }
  for (const [role, decisional] of [
    ["decisive", true],
    ["considered", true],
    ["considered", false],
    ["considered", null],
  ]) {
    add(analysis.contributionRoleText(role, decisional));
  }
  for (const signal_type of ["lip_forensics", "face_forgery", "active_speaker", "audio_authenticity"]) {
    add(analysis.componentDetectorName({ provider: "p", signal_type, state: "completed" }));
  }

  // The acquisition sentences, with the host still a slot.
  for (const acquisition_method of ["upload", "url", null]) {
    for (const source_host of [null, "host.example"]) {
      for (const was_assembled of [false, true]) {
        const analysisFacts = { acquisition_method, source_host, was_assembled };
        add(analysis.acquisitionSentence(analysisFacts).text);
        add(analysis.credentialsAbsentSentence(analysisFacts).text);
      }
    }
  }

  return found;
}
