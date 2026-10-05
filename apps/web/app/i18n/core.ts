/**
 * The presentation layer's language, and nothing below it (R16-T1).
 *
 * **Translation happens at the last step before a word is drawn, and nowhere else.** The API
 * speaks one language — its own canonical spellings, `completed`, `FAILED`, `UNREVIEWED` — and
 * every value that crosses the wire, every value written to the database and every value the
 * risk engine reads stays in that spelling. A locale is a property of the reader's browser. It
 * is not in the URL, not in a cookie, not in a request header and not in any payload; switching
 * it changes what is painted and changes nothing that is sent, stored or decided.
 *
 * Two kinds of lookup, and they fail differently on purpose:
 *
 * - `translate()` is for this application's own interface strings — a navigation label, the
 *   selector's name. Those keys are written in this repository, so a missing Turkish entry
 *   falls back to the English one, and only a key nobody defined falls back to itself.
 * - `translateCanonical()` is for a value the API produced. A value this file has no entry for
 *   is returned **exactly as it arrived** — not `Unknown`, not `Bilinmeyen`, not title-cased.
 *   A status the application does not recognise is the case a reader most needs to see as it
 *   is, and a friendly placeholder would hide exactly that.
 *
 * No imports, no React and no `window`: this file is the part that can be tested by
 * `node --test` without a bundler. The browser half — the hook, the selector — is in
 * `./client.tsx`, and it reaches `localStorage` only through `createLocaleStore` below.
 */

export const LOCALES = ["en", "tr"] as const;

export type Locale = (typeof LOCALES)[number];

/** The product's existing language. What the server renders, and what a new reader sees. */
export const DEFAULT_LOCALE: Locale = "en";

/** The one `localStorage` key this feature owns. */
export const LOCALE_STORAGE_KEY = "inspectroot.locale";

export function isLocale(value: unknown): value is Locale {
  return typeof value === "string" && (LOCALES as readonly string[]).includes(value);
}

/* ------------------------------------------------------------------ *
 * Dictionaries
 * ------------------------------------------------------------------ */

/*
 * R16-T1 proved the mechanism on the admin rail, the selector and the job status chips. R16-T2
 * adds the interface around the evidence: the public page, sign-in, the workspace's header,
 * ingest form and queue furniture, the report's screen-only controls and feedback form, and
 * the error boundaries. The report body and every forensic word the dashboard shares with it
 * — risk labels, signal states, the methodology notes — belong to R16-T3 and are not here.
 */
export const MESSAGES = {
  en: {
    "nav.label": "Administration",
    "nav.group.operations": "Operations",
    "nav.group.governance": "Governance",
    "nav.group.access": "Access",
    "nav.accounts": "Accounts",
    "nav.jobs": "Detection jobs",
    "nav.reviewQueue": "Review queue",
    "nav.analytics": "Operational summary",
    "nav.audit": "Audit log",
    "nav.apiKeys": "API keys",
    "nav.backToWorkspace": "Back to workspace",
    "nav.menu": "Menu",
    "nav.close": "Close",
    "nav.closeNavigation": "Close navigation",
    "locale.selector": "Language",
    "locale.en": "English",
    "locale.tr": "Türkçe",
    "auth.signIn": "Sign in",
    "auth.signOut": "Sign out",
    "landing.tagline": "Media forensics, signal by signal.",
    "landing.intro":
      "Submit a file or a URL and read what each detector established on its own — provenance, synthetic video, face manipulation, voice — kept as separate evidence rather than averaged into one number that would mean nothing.",
    "landing.goToWorkspace": "Go to workspace",
    "landing.accounts":
      "Accounts are created by an administrator. Analyses are visible to the account that submitted them.",
    "login.title": "Authenticate",
    "login.intro":
      "Analyses are visible to the account that submitted them. Accounts are created by an administrator.",
    "login.email": "Email",
    "login.password": "Password",
    "login.failed": "Sign in failed. Check your email and password and try again.",
    "header.systemStatus": "System status: ",
    "header.operational": "Operational",
    "header.degraded": "Degraded",
    "header.adminConsole": "Admin console",
    "health.check": "Web → API → DB connectivity check",
    "health.web": "Web",
    "health.api": "API",
    "health.database": "Database",
    "ingest.legend": "Ingest",
    "ingest.heading": "Analyse media",
    "ingest.intro":
      "An MP4 or MOV file, or a link to one. A URL is downloaded by the API first, so the page waits for the download before the analysis is queued; both then go through the same pipeline and appear in the queue below. Live streams cannot be analysed.",
    "ingest.localFile": "Local file",
    "ingest.mediaUrl": "Media URL",
    "ingest.mode": "Analysis mode",
    "ingest.deepAnalysis": "Deep analysis",
    "ingest.quickScan": "Quick scan",
    "ingest.modeNote":
      "Both reach the same verdict from the same detectors. A quick scan does not run the supplementary evidence detectors, which cannot change that verdict.",
    "ingest.urlWins": "If both are filled in, the URL is used.",
    "ingest.submit": "Run analysis",
    "ingest.queued": "Queued for analysis",
    "queue.legend": "Queue",
    "queue.heading": "Recent analyses",
    "queue.captionAdmin": "Every analysis in the system, as an administrator sees it.",
    "queue.captionUser": "The analyses submitted by this account.",
    "queue.emptyTitle": "No analyses yet",
    "queue.emptyBody":
      "Submit a file or a URL above. Each accepted submission becomes a record here.",
    "queue.column.media": "Media",
    "queue.column.status": "Status",
    "queue.column.risk": "Risk",
    "queue.column.submitted": "Submitted",
    "queue.openReport": "— open report",
    "queue.evidence": "Evidence",
    "report.backToDashboard": "← Back to dashboard",
    "report.dashboard": "← Dashboard",
    "report.print": "Print / Save as PDF",
    "report.unavailable": "Report unavailable",
    "feedback.title": "Your feedback",
    "feedback.intro":
      "Do you agree with this result? Your answer is recorded as your feedback only. It does not change the assessment or any evidence in this report, and it is not treated as a verified statement about the media.",
    "feedback.saved": "Thank you — your feedback was saved. The report was not changed.",
    "feedback.loadFailed": "Your earlier feedback could not be loaded.",
    "feedback.assessmentLegend": "Your assessment of this result",
    "feedback.details": "Add details (optional)",
    "feedback.claimQuestion": "What do you believe this media is?",
    "feedback.noClaim": "No claim",
    "feedback.notes": "Notes",
    "feedback.update": "Update feedback",
    "feedback.send": "Send feedback",
    "error.title": "Something went wrong",
    "error.page": "This page could not be rendered. Nothing about any analysis has been changed.",
    "error.application":
      "The application could not be rendered. Nothing about any analysis has been changed.",
    "error.reference": "Reference:",
    "error.retry": "Try again",
  },
  tr: {
    "nav.label": "Yönetim",
    "nav.group.operations": "Operasyon",
    "nav.group.governance": "Yönetişim",
    "nav.group.access": "Erişim",
    "nav.accounts": "Hesaplar",
    "nav.jobs": "Tespit işleri",
    "nav.reviewQueue": "İnceleme kuyruğu",
    "nav.analytics": "Operasyon özeti",
    "nav.audit": "Denetim kaydı",
    "nav.apiKeys": "API anahtarları",
    "nav.backToWorkspace": "Çalışma alanına dön",
    "nav.menu": "Menü",
    "nav.close": "Kapat",
    "nav.closeNavigation": "Gezinmeyi kapat",
    "locale.selector": "Dil",
    "locale.en": "English",
    "locale.tr": "Türkçe",
    "auth.signIn": "Giriş yap",
    "auth.signOut": "Çıkış yap",
    "landing.tagline": "Medya adli analizi, sinyal sinyal.",
    "landing.intro":
      "Bir dosya ya da URL gönderin ve her dedektörün kendi başına neyi tespit ettiğini okuyun — köken bilgisi, sentetik video, yüz manipülasyonu, ses — hiçbir şey ifade etmeyecek tek bir sayıda ortalanmak yerine ayrı kanıtlar olarak tutulur.",
    "landing.goToWorkspace": "Çalışma alanına git",
    "landing.accounts":
      "Hesaplar bir yönetici tarafından oluşturulur. Analizleri yalnızca onları gönderen hesap görebilir.",
    "login.title": "Kimlik doğrulama",
    "login.intro":
      "Analizleri yalnızca onları gönderen hesap görebilir. Hesaplar bir yönetici tarafından oluşturulur.",
    "login.email": "E-posta",
    "login.password": "Parola",
    "login.failed": "Giriş başarısız. E-posta adresinizi ve parolanızı kontrol edip yeniden deneyin.",
    "header.systemStatus": "Sistem durumu: ",
    "header.operational": "Çalışıyor",
    "header.degraded": "Sorunlu",
    "header.adminConsole": "Yönetim konsolu",
    "health.check": "Web → API → Veritabanı bağlantı kontrolü",
    "health.web": "Web",
    "health.api": "API",
    "health.database": "Veritabanı",
    "ingest.legend": "Gönderim",
    "ingest.heading": "Medya analiz et",
    "ingest.intro":
      "Bir MP4 ya da MOV dosyası veya böyle bir dosyanın bağlantısı. URL önce API tarafından indirilir; bu yüzden sayfa, analiz kuyruğa alınmadan önce indirmenin bitmesini bekler. İkisi de ardından aynı işlem hattından geçer ve aşağıdaki kuyrukta görünür. Canlı yayınlar analiz edilemez.",
    "ingest.localFile": "Yerel dosya",
    "ingest.mediaUrl": "Medya URL'si",
    "ingest.mode": "Analiz modu",
    "ingest.deepAnalysis": "Derin analiz",
    "ingest.quickScan": "Hızlı tarama",
    "ingest.modeNote":
      "İkisi de aynı dedektörlerden aynı karara ulaşır. Hızlı tarama, bu kararı değiştiremeyen ek kanıt dedektörlerini çalıştırmaz.",
    "ingest.urlWins": "İkisi de doldurulursa URL kullanılır.",
    "ingest.submit": "Analizi başlat",
    "ingest.queued": "Analiz için kuyruğa alındı",
    "queue.legend": "Kuyruk",
    "queue.heading": "Son analizler",
    "queue.captionAdmin": "Sistemdeki tüm analizler, bir yöneticinin gördüğü şekliyle.",
    "queue.captionUser": "Bu hesabın gönderdiği analizler.",
    "queue.emptyTitle": "Henüz analiz yok",
    "queue.emptyBody":
      "Yukarıdan bir dosya ya da URL gönderin. Kabul edilen her gönderim burada bir kayda dönüşür.",
    "queue.column.media": "Medya",
    "queue.column.status": "Durum",
    "queue.column.risk": "Risk",
    "queue.column.submitted": "Gönderilme",
    "queue.openReport": "— raporu aç",
    "queue.evidence": "Kanıt",
    "report.backToDashboard": "← Panele dön",
    "report.dashboard": "← Panel",
    "report.print": "Yazdır / PDF olarak kaydet",
    "report.unavailable": "Rapor kullanılamıyor",
    "feedback.title": "Geri bildiriminiz",
    "feedback.intro":
      "Bu sonuca katılıyor musunuz? Yanıtınız yalnızca geri bildiriminiz olarak kaydedilir. Bu rapordaki değerlendirmeyi ya da herhangi bir kanıtı değiştirmez ve medya hakkında doğrulanmış bir beyan olarak ele alınmaz.",
    "feedback.saved": "Teşekkürler — geri bildiriminiz kaydedildi. Rapor değiştirilmedi.",
    "feedback.loadFailed": "Önceki geri bildiriminiz yüklenemedi.",
    "feedback.assessmentLegend": "Bu sonuca ilişkin değerlendirmeniz",
    "feedback.details": "Ayrıntı ekle (isteğe bağlı)",
    "feedback.claimQuestion": "Bu medyanın ne olduğunu düşünüyorsunuz?",
    "feedback.noClaim": "İddia yok",
    "feedback.notes": "Notlar",
    "feedback.update": "Geri bildirimi güncelle",
    "feedback.send": "Geri bildirim gönder",
    "error.title": "Bir şeyler ters gitti",
    "error.page": "Bu sayfa görüntülenemedi. Hiçbir analizde değişiklik yapılmadı.",
    "error.application": "Uygulama görüntülenemedi. Hiçbir analizde değişiklik yapılmadı.",
    "error.reference": "Referans:",
    "error.retry": "Tekrar dene",
  },
} as const satisfies Record<Locale, Record<string, string>>;

export type MessageKey = keyof (typeof MESSAGES)["en"];

/*
 * Display names for values the API owns, by domain. The keys are the API's canonical spellings
 * and are matched exactly — `completed` and `COMPLETED` are two different words to this table,
 * because they are two different words on the wire. English is listed too, as the identity, so
 * that a domain's known values are written down in one place; it is never a rewrite.
 */
export const CANONICAL = {
  // `app/db/models.py` — the four values `jobs.status` can hold.
  jobStatus: {
    en: { queued: "queued", processing: "processing", completed: "completed", failed: "failed" },
    tr: { queued: "kuyrukta", processing: "işleniyor", completed: "tamamlandı", failed: "başarısız" },
  },
  // `app/db/models.py` — the three values `analyses.status` can hold.
  analysisStatus: {
    en: { queued: "queued", completed: "completed", failed: "failed" },
    tr: { queued: "kuyrukta", completed: "tamamlandı", failed: "başarısız" },
  },
  // `app/session.ts` — `USER_ROLE_ADMIN` and `USER_ROLE_USER`, spelled as the API spells them.
  userRole: {
    en: { ADMIN: "ADMIN", USER: "USER" },
    tr: { ADMIN: "YÖNETİCİ", USER: "KULLANICI" },
  },
  // The health popover's detail column: the API's own `/health` words (`ok`, `degraded`,
  // `unavailable`) and the three the dashboard writes in when it has no answer to show.
  healthState: {
    en: {
      ok: "ok",
      degraded: "degraded",
      unavailable: "unavailable",
      running: "running",
      unreachable: "unreachable",
      unknown: "unknown",
    },
    tr: {
      ok: "tamam",
      degraded: "sorunlu",
      unavailable: "kullanılamıyor",
      running: "çalışıyor",
      unreachable: "erişilemiyor",
      unknown: "bilinmiyor",
    },
  },
  // `app/user-feedback.ts` — the values a feedback form posts. The form keeps posting these;
  // only the words beside the controls change.
  feedbackAssessment: {
    en: { AGREE: "Agree", DISAGREE: "Disagree", UNSURE: "Unsure" },
    tr: { AGREE: "Katılıyorum", DISAGREE: "Katılmıyorum", UNSURE: "Emin değilim" },
  },
  feedbackClaimedLabel: {
    en: { GENUINE: "Genuine", AI_GENERATED: "AI-generated", FACE_SWAP: "Face swap", OTHER: "Other" },
    tr: { GENUINE: "Gerçek", AI_GENERATED: "Yapay zekâ üretimi", FACE_SWAP: "Yüz değiştirme", OTHER: "Diğer" },
  },
  /*
   * The sentences this application's own route handlers and fetchers write, matched exactly.
   * They travel to the page as a query parameter or a fetch result, in English, and the page is
   * where they are named in the reader's language. The same slot also carries the API's own
   * refusal text and sentences with a status code in them; those match nothing here and are
   * shown as they arrived, which is the point of looking them up through this table rather
   * than through `translate`.
   */
  webMessage: {
    en: {},
    tr: {
      "The submission could not be read.": "Gönderim okunamadı.",
      "Choose a file or paste a URL first.": "Önce bir dosya seçin ya da bir URL yapıştırın.",
      "The upload could not be completed.": "Yükleme tamamlanamadı.",
      "The API could not be reached.": "API'ye ulaşılamadı.",
      "Feedback cannot be given on this analysis.": "Bu analize geri bildirim verilemez.",
      "Choose Agree, Disagree or Unsure before sending.":
        "Göndermeden önce Katılıyorum, Katılmıyorum ya da Emin değilim seçeneğini işaretleyin.",
      "The analysis list is temporarily unavailable.": "Analiz listesi geçici olarak kullanılamıyor.",
      "The analysis list could not be read.": "Analiz listesi okunamadı.",
      "No analysis was found with this id.": "Bu kimlikle bir analiz bulunamadı.",
      "This analysis is temporarily unavailable.": "Bu analiz geçici olarak kullanılamıyor.",
      "This analysis could not be read.": "Bu analiz okunamadı.",
    },
  },
} as const satisfies Record<string, Record<Locale, Record<string, string>>>;

export type CanonicalDomain = keyof typeof CANONICAL;

/* ------------------------------------------------------------------ *
 * Lookups
 * ------------------------------------------------------------------ */

/** An interface string. Falls back to English, then to the key itself. */
export function translate(locale: Locale, key: MessageKey): string {
  const table: Record<string, string> = MESSAGES[locale] ?? MESSAGES[DEFAULT_LOCALE];
  return table[key] ?? (MESSAGES[DEFAULT_LOCALE] as Record<string, string>)[key] ?? key;
}

/**
 * A value the API produced, as the reader's locale names it.
 *
 * **An unrecognised value is returned unchanged, byte for byte.** No `Unknown`, no English
 * fallback that might itself be a guess, no case folding. `Object.hasOwn` rather than `in` or
 * a plain index, so a value spelled like an inherited property (`constructor`, `toString`) is
 * treated as the unknown word it is rather than resolving to a function.
 */
export function translateCanonical(
  locale: Locale,
  domain: CanonicalDomain,
  value: string,
): string {
  const table: Record<string, string> | undefined = CANONICAL[domain]?.[locale];
  if (table !== undefined && Object.hasOwn(table, value)) {
    return table[value];
  }
  return value;
}

/* ------------------------------------------------------------------ *
 * Persistence
 * ------------------------------------------------------------------ */

/** The slice of `Storage` this feature uses, so a test can hand it a plain object. */
export type LocaleStorage = Pick<Storage, "getItem" | "setItem">;

/**
 * The reader's choice, kept in their browser and nowhere else.
 *
 * `storage` is a getter rather than a value because `localStorage` itself can throw on access —
 * a private window, blocked site data — and so can every read and write. Each of those is
 * caught and read as "no stored choice": the worst a broken store can do is leave the reader in
 * English, never break the page.
 *
 * `subscribe` is the `useSyncExternalStore` contract. Listeners hear a change made in this tab
 * through `set`, and `notify` exists for the browser's `storage` event, which is how another
 * tab's choice reaches this one.
 */
export function createLocaleStore(storage: () => LocaleStorage | null) {
  const listeners = new Set<() => void>();

  function get(): Locale {
    try {
      const stored = storage()?.getItem(LOCALE_STORAGE_KEY);
      return isLocale(stored) ? stored : DEFAULT_LOCALE;
    } catch {
      return DEFAULT_LOCALE;
    }
  }

  function notify(): void {
    for (const listener of listeners) {
      listener();
    }
  }

  function set(locale: Locale): void {
    if (!isLocale(locale)) {
      return;
    }
    try {
      storage()?.setItem(LOCALE_STORAGE_KEY, locale);
    } catch {
      // Not persisted. `get` reads the store, so an unwritable store cannot hold a choice and
      // the reader stays in the default language. There is no second place to keep it.
    }
    notify();
  }

  function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }

  return { get, set, subscribe, notify };
}
