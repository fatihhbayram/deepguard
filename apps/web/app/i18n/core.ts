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
 * the error boundaries. The report body's sentences are in `REPORT_COPY` below (R16-T3). The
 * forensic words the dashboard and the admin pages share with the report are translated there
 * only where the report prints them; those two surfaces still print them in English.
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

/*
 * The forensic report's own sentences (R16-T3), keyed by the exact English the report prints.
 *
 * Keyed by the sentence rather than by a name because the English is where these sentences are
 * written and reviewed — in the report, and in the vocabulary `../analysis` shares with the
 * dashboard — and it stays there unchanged: the English report is byte for byte the report it
 * was before this table existed. This is the shape R16-T2 gave `webMessage`, for the same reason.
 *
 * What is translated is presentation copy and nothing else. A slot written `{name}` is a value
 * from the record — a host, a status, a rule id, a figure — and is filled in by the renderer as
 * the record holds it; no slot is ever looked up here, and no entry names a forensic concept
 * the English does not. The product's own name is InspectRoot in both languages.
 *
 * English is not listed: it is the key. A sentence with no entry here is drawn in English, and
 * marked as English, rather than guessed at.
 */
export const REPORT_COPY = {
  tr: {
    "No risk decision":
      "Risk kararı yok",
    "No synthetic-video signal is stored for this analysis.":
      "Bu analiz için kayıtlı bir sentetik video sinyali yok.",
    "No provenance signal is stored for this analysis.":
      "Bu analiz için kayıtlı bir köken bilgisi sinyali yok.",
    "No active-speaker signal is stored for this analysis.":
      "Bu analiz için kayıtlı bir aktif konuşmacı sinyali yok.",
    "No audio-authenticity signal is stored for this analysis.":
      "Bu analiz için kayıtlı bir ses özgünlüğü sinyali yok.",
    "No face-manipulation signal is stored for this analysis.":
      "Bu analiz için kayıtlı bir yüz manipülasyonu sinyali yok.",
    "No mouth-dynamics signal is stored for this analysis.":
      "Bu analiz için kayıtlı bir ağız dinamiği sinyali yok.",
    "Analysed media":
      "Analiz edilen medya",
    "What ffprobe established about the analysed artifact, as the database kept it. This acquisition was assembled by InspectRoot, so it is not a copy of a single published file.":
      "ffprobe'un analiz edilen artefakt hakkında belirlediği bilgiler, veritabanının sakladığı haliyle. Bu edinim InspectRoot tarafından birleştirildi; bu nedenle yayımlanmış tek bir dosyanın kopyası değildir.",
    "What ffprobe established about the forensic original, as the database kept it.":
      "ffprobe'un adli orijinal hakkında belirlediği bilgiler, veritabanının sakladığı haliyle.",
    "Original filename":
      "Orijinal dosya adı",
    "Declared content type":
      "Bildirilen içerik türü",
    "Container (ffprobe)":
      "Kapsayıcı (ffprobe)",
    "Video codec":
      "Video codec'i",
    "Original encoded resolution":
      "Orijinal kodlanmış çözünürlük",
    "Analysed resolution":
      "Analiz edilen çözünürlük",
    "Display rotation":
      "Görüntüleme döndürmesi",
    "{degrees}° clockwise":
      "saat yönünde {degrees}°",
    "Frame rate":
      "Kare hızı",
    "Duration":
      "Süre",
    "Pixel format":
      "Piksel biçimi",
    "Constant frame rate":
      "Sabit kare hızı",
    "yes":
      "evet",
    "no":
      "hayır",
    "Normalized for detection":
      "Tespit için normalleştirildi",
    "Acquisition":
      "Edinim",
    "SHA-256 of the submitted media":
      "Gönderilen medyanın SHA-256 değeri",
    "This is the hash of the media as it reached InspectRoot. It is not a hash or a signature of this report.":
      "Bu, medyanın InspectRoot'a ulaştığı haliyle hash değeridir. Bu raporun hash değeri ya da imzası değildir.",
    "The detectors did not read these bytes. This media required normalization, so every reading on this report was taken on a transcoded derivative of it, and this hash identifies the submitted file rather than the derivative that was scored. The derivative's own content identity is not shown on this report.":
      "Dedektörler bu baytları okumadı. Bu medya normalleştirme gerektirdi; bu nedenle bu rapordaki her okuma, medyanın yeniden kodlanmış bir türevi üzerinde alındı ve bu hash, puanlanan türevi değil gönderilen dosyayı tanımlar. Türevin kendi içerik kimliği bu raporda gösterilmez.",
    "This media required no normalization, so these are also the bytes every detector on this report read.":
      "Bu medya normalleştirme gerektirmedi; dolayısıyla bunlar aynı zamanda bu rapordaki her dedektörün okuduğu baytlardır.",
    "Because this acquisition was assembled from separate video and audio streams, it hashes the artifact InspectRoot built and stored, not a file the source published — the source published no single file to compare it against.":
      "Bu edinim ayrı video ve ses akışlarından birleştirildiği için bu hash, kaynağın yayımladığı bir dosyanın değil, InspectRoot'un oluşturup sakladığı artefaktın hash değeridir — kaynak, karşılaştırılabilecek tek bir dosya yayımlamadı.",
    "It hashes the file served to InspectRoot at acquisition time. Whether that file matches any upstream or publisher-original file is not something this analysis establishes.":
      "Bu hash, edinim anında InspectRoot'a sunulan dosyanın hash değeridir. Bu dosyanın herhangi bir üst kaynaktaki ya da yayıncının orijinal dosyasıyla eşleşip eşleşmediği bu analizin belirlediği bir şey değildir.",
    "Reached its threshold — this is what decided":
      "Eşiğine ulaştı — kararı veren budur",
    "Below its threshold — did not contribute":
      "Eşiğinin altında — katkıda bulunmadı",
    "Did not reach its threshold — did not contribute":
      "Eşiğine ulaşmadı — katkıda bulunmadı",
    "Answered, but unreadable":
      "Yanıt verdi, ancak okunamadı",
    "No usable reading":
      "Kullanılabilir okuma yok",
    "Not read by this ruleset":
      "Bu kural seti tarafından okunmadı",
    "Read as evidence — not eligible to decide under this ruleset":
      "Kanıt olarak okundu — bu kural setinde karar vermeye uygun değil",
    "Not named by the rule — see this detector's own panel below":
      "Kural tarafından adı belirtilmedi — bu dedektörün aşağıdaki kendi paneline bakın",
    "This detector returned no reading — its signal is stored as {status}.":
      "Bu dedektör bir okuma döndürmedi — sinyali {status} olarak kayıtlı.",
    "No signal from this detector is recorded for this analysis.":
      "Bu analiz için bu dedektöre ait kayıtlı bir sinyal yok.",
    "This detector returned a reading; its section below gives the figures and the build that produced them.":
      "Bu dedektör bir okuma döndürdü; aşağıdaki kendi bölümü değerleri ve bunları üreten sürümü verir.",
    "Assessment summary":
      "Değerlendirme özeti",
    "Decision coverage: {usable}/{total} {status}":
      "Karar kapsamı: {usable}/{total} {status}",
    "Recorded risk level":
      "Kayıtlı risk düzeyi",
    "InspectRoot risk classification":
      "InspectRoot risk sınıflandırması",
    "No risk decision is stored for this analysis. That is not the same as {unknown}: nothing classified this analysis, so there is no decision to report — it was analysed before the risk engine existed, or it has not finished.":
      "Bu analiz için kayıtlı bir risk kararı yok. Bu, {unknown} ile aynı şey değildir: bu analizi hiçbir şey sınıflandırmadı, dolayısıyla raporlanacak bir karar yok — analiz risk motoru var olmadan önce yapıldı ya da henüz tamamlanmadı.",
    "Unknown":
      "Bilinmiyor",
    "The stored risk state {level} is not a risk class this build classifies under, so it is reported as unsupported rather than presented as an InspectRoot classification.":
      "Kayıtlı risk durumu {level}, bu sürümün sınıflandırdığı bir risk sınıfı değildir; bu nedenle bir InspectRoot sınıflandırması olarak sunulmak yerine desteklenmeyen olarak raporlanır.",
    "The risk engine ran and a rule fired. Its conclusion is that the evidence does not support a classification — an answer, not a missing one.":
      "Risk motoru çalıştı ve bir kural tetiklendi. Vardığı sonuç, kanıtın bir sınıflandırmayı desteklemediğidir — bu eksik bir yanıt değil, bir yanıttır.",
    "Why this classification":
      "Neden bu sınıflandırma",
    "How each detector contributed":
      "Her dedektörün katkısı",
    "NVIDIA synthetic-video detector":
      "NVIDIA sentetik video dedektörü",
    "EfficientNet-B7 face-manipulation classifier":
      "EfficientNet-B7 yüz manipülasyonu sınıflandırıcısı",
    "LipForensics mouth-dynamics model":
      "LipForensics ağız dinamiği modeli",
    "What this covers":
      "Bunun kapsadıkları",
    "Rule fired":
      "Tetiklenen kural",
    "Ruleset version":
      "Kural seti sürümü",
    "Calibration ID":
      "Kalibrasyon kimliği",
    "This build has no description for rule {rule} under ruleset {ruleset}, so the trace above is shown without one. The decision itself is reproduced exactly as it was stored.":
      "Bu sürümde {rule} kuralı için, {ruleset} kural seti altında bir açıklama yok; bu nedenle yukarıdaki iz açıklamasız gösteriliyor. Kararın kendisi, kaydedildiği haliyle aynen yeniden üretilmiştir.",
    "The assessment above is a deterministic InspectRoot classification based on calibrated forensic evidence. It is not a Fake/Real determination. Each detector was compared only against the threshold measured for it, and those thresholds are points on unrelated scales that cannot be compared with each other; the scores were never averaged, weighted, voted on or combined into a single number. This classification was recorded when the analysis ran and is reproduced here unchanged; it is not recalculated by this report.":
      "Yukarıdaki değerlendirme, kalibre edilmiş adli kanıta dayanan deterministik bir InspectRoot sınıflandırmasıdır. Sahte/Gerçek belirlemesi değildir. Her dedektör yalnızca kendisi için ölçülmüş eşikle karşılaştırıldı ve bu eşikler, birbiriyle karşılaştırılamayan ilişkisiz ölçeklerdeki noktalardır; skorların ortalaması hiç alınmadı, ağırlıklandırılmadı, oylanmadı ya da tek bir sayıda birleştirilmedi. Bu sınıflandırma analiz çalıştığında kaydedildi ve burada değiştirilmeden yeniden üretilir; bu rapor tarafından yeniden hesaplanmaz.",
    "Synthetic-video detector":
      "Sentetik video dedektörü",
    "Face-manipulation classifier":
      "Yüz manipülasyonu sınıflandırıcısı",
    "Mouth-dynamics model":
      "Ağız dinamiği modeli",
    "Provider":
      "Sağlayıcı",
    "Deployment":
      "Dağıtım",
    "Score (as stored)":
      "Skor (kayıtlı haliyle)",
    "Threshold under this ruleset":
      "Bu kural setindeki eşik",
    "This build has no interpretation for that condition, so none is given. No threshold outcome is inferred from it, and it establishes nothing about the media.":
      "Bu sürümün bu koşul için bir yorumu yok; bu nedenle yorum verilmiyor. Ondan herhangi bir eşik sonucu çıkarılmaz ve medya hakkında hiçbir şey belirlemez.",
    "Decision breakdown":
      "Karar dökümü",
    "The stored decision read back under the ruleset it was taken under. Every statement below is the API's; this report compares no score against any threshold and re-derives no part of the classification.":
      "Kayıtlı karar, alındığı kural seti altında geri okunmuştur. Aşağıdaki her ifade API'ye aittir; bu rapor hiçbir skoru hiçbir eşikle karşılaştırmaz ve sınıflandırmanın hiçbir kısmını yeniden türetmez.",
    "Decision as stored":
      "Kaydedildiği haliyle karar",
    "What the rule that fired meant":
      "Tetiklenen kuralın anlamı",
    "The API could not resolve this decision's ruleset version or the calibration it was measured under, so the detailed reading of it is unavailable. The decision itself is reproduced above exactly as it was stored; what is missing is the interpretation, and none is guessed in its place.":
      "API, bu kararın kural seti sürümünü ya da ölçüldüğü kalibrasyonu çözümleyemedi; bu nedenle kararın ayrıntılı okuması kullanılamıyor. Kararın kendisi yukarıda kaydedildiği haliyle aynen yeniden üretilmiştir; eksik olan yorumdur ve yerine hiçbir tahmin konmaz.",
    "How each detector stood in this decision":
      "Bu kararda her dedektörün durumu",
    "The API reported no detector contributions for this decision. That is a statement about what this record can be read to say, not a finding that no detector ran — each detector's own stored evidence is reproduced further down this report.":
      "API bu karar için hiçbir dedektör katkısı bildirmedi. Bu, bu kaydın neyi söylediğinin okunabileceğine dair bir ifadedir; hiçbir dedektörün çalışmadığına dair bir bulgu değildir — her dedektörün kendi kayıtlı kanıtı bu raporun ilerleyen kısmında yeniden üretilmiştir.",
    "None of the conditions above is a statement about the media. A detector that did not reach its threshold, one that could not be read, and one with no reading at all are three different facts about the evidence, and none of them is evidence that the media was not manipulated.":
      "Yukarıdaki koşulların hiçbiri medya hakkında bir ifade değildir. Eşiğine ulaşmayan bir dedektör, okunamayan bir dedektör ve hiç okuması olmayan bir dedektör, kanıt hakkında üç farklı olgudur ve hiçbiri medyanın manipüle edilmediğinin kanıtı değildir.",
    "Scope of this assessment":
      "Bu değerlendirmenin kapsamı",
    "Scope of this risk model":
      "Bu risk modelinin kapsamı",
    "This assessment is validated for generated video and for face swaps, by the two decision detectors it is taken from, each read against a threshold measured for it alone.":
      "Bu değerlendirme, üretilmiş video ve yüz değiştirmeler için, dayandığı iki karar dedektörü tarafından doğrulanmıştır; her biri yalnızca kendisi için ölçülmüş bir eşiğe göre okunur.",
    "Both thresholds were set to almost never flag legitimate footage: neither detector flagged any of the 54 genuine clips in the calibration corpus. That choice is paid for in detection rate. At these operating points the synthetic-video detector flagged 54.6% of generated video and the face classifier flagged 44% of face swaps, so a great deal of manipulated media is correctly not flagged.":
      "Her iki eşik de meşru görüntüleri neredeyse hiç işaretlememek üzere ayarlandı: kalibrasyon derlemindeki 54 gerçek klibin hiçbirini iki dedektör de işaretlemedi. Bu tercihin bedeli tespit oranıyla ödenir. Bu çalışma noktalarında sentetik video dedektörü üretilmiş videoların %54.6'sını, yüz sınıflandırıcısı ise yüz değiştirmelerin %44'ünü işaretledi; dolayısıyla manipüle edilmiş medyanın büyük bir kısmı doğru biçimde işaretlenmez.",
    "The mouth-dynamics model still runs and its score is reported below, but under this ruleset it is evidence only: it cannot reach the assessment above, and a reading it failed to produce does not reduce the decision coverage stated there. R7-T5 replayed the rules that once let it decide over 307 independent genuine recordings and found it responsible for 21 of 22 false HIGH results, so it was withdrawn rather than re-tuned — no study has measured an operating point that would be safe here.":
      "Ağız dinamiği modeli hâlâ çalışır ve skoru aşağıda raporlanır, ancak bu kural setinde yalnızca kanıttır: yukarıdaki değerlendirmeye ulaşamaz ve üretemediği bir okuma orada belirtilen karar kapsamını azaltmaz. R7-T5, bir zamanlar onun karar vermesine izin veren kuralları 307 bağımsız gerçek kayıt üzerinde yeniden oynattı ve 22 yanlış HIGH sonucunun 21'inden onun sorumlu olduğunu buldu; bu nedenle yeniden ayarlanmak yerine geri çekildi — hiçbir çalışma burada güvenli olacak bir çalışma noktası ölçmedi.",
    "The two deciding detectors cover different things and are read independently. Neither staying below its threshold is evidence about the other: in the R4-T1 study they never agreed on a single clip, and each was blind to the manipulation family the other was calibrated for.":
      "Karar veren iki dedektör farklı şeyleri kapsar ve bağımsız olarak okunur. Birinin eşiğinin altında kalması diğeri hakkında kanıt değildir: R4-T1 çalışmasında tek bir klipte bile hiç uyuşmadılar ve her biri, diğerinin kalibre edildiği manipülasyon ailesine karşı kördü.",
    "This risk model is validated for generated video and for face swaps, by two separate detectors with separate thresholds. Absence of HIGH risk does not mean the media is genuine.":
      "Bu risk modeli, üretilmiş video ve yüz değiştirmeler için, ayrı eşiklere sahip iki ayrı dedektör tarafından doğrulanmıştır. HIGH riskin olmaması medyanın gerçek olduğu anlamına gelmez.",
    "Both thresholds were set to almost never flag legitimate footage: neither detector flagged any of the 54 genuine clips in the calibration corpus. That choice is paid for in detection rate. At these operating points the synthetic-video detector flagged 54.6% of generated video and the face classifier flagged 44% of face swaps, so a great deal of manipulated media is correctly not flagged as HIGH.":
      "Her iki eşik de meşru görüntüleri neredeyse hiç işaretlememek üzere ayarlandı: kalibrasyon derlemindeki 54 gerçek klibin hiçbirini iki dedektör de işaretlemedi. Bu tercihin bedeli tespit oranıyla ödenir. Bu çalışma noktalarında sentetik video dedektörü üretilmiş videoların %54.6'sını, yüz sınıflandırıcısı ise yüz değiştirmelerin %44'ünü işaretledi; dolayısıyla manipüle edilmiş medyanın büyük bir kısmı doğru biçimde HIGH olarak işaretlenmez.",
    "The mouth-dynamics model still runs and its score is reported below, but it cannot change this classification. Under the previous ruleset it could, on its own. R7-T5 replayed those rules over 307 independent genuine recordings and found that rule responsible for 21 of 22 false HIGH results, so it was removed rather than re-tuned — no study has measured an operating point that would be safe here. Its score is recorded as independent forensic evidence and nothing more.":
      "Ağız dinamiği modeli hâlâ çalışır ve skoru aşağıda raporlanır, ancak bu sınıflandırmayı değiştiremez. Önceki kural setinde tek başına değiştirebiliyordu. R7-T5 bu kuralları 307 bağımsız gerçek kayıt üzerinde yeniden oynattı ve 22 yanlış HIGH sonucunun 21'inden o kuralın sorumlu olduğunu buldu; bu nedenle yeniden ayarlanmak yerine kaldırıldı — hiçbir çalışma burada güvenli olacak bir çalışma noktası ölçmedi. Skoru bağımsız adli kanıt olarak kaydedilir, fazlası değil.",
    "The two deciding detectors cover different things and are read independently. Neither scoring low is evidence against the other: in the R4-T1 study they never agreed on a single clip, and each was blind to the manipulation family the other was calibrated for.":
      "Karar veren iki dedektör farklı şeyleri kapsar ve bağımsız olarak okunur. Birinin düşük skor alması diğerine karşı kanıt değildir: R4-T1 çalışmasında tek bir klipte bile hiç uyuşmadılar ve her biri, diğerinin kalibre edildiği manipülasyon ailesine karşı kördü.",
    "This risk model is validated for generated video and for face swaps, by three separate detectors with separate thresholds. Absence of HIGH risk does not mean the media is genuine.":
      "Bu risk modeli, üretilmiş video ve yüz değiştirmeler için, ayrı eşiklere sahip üç ayrı dedektör tarafından doğrulanmıştır. HIGH riskin olmaması medyanın gerçek olduğu anlamına gelmez.",
    "Every threshold was set to almost never flag legitimate footage: no detector flagged any genuine clip in the corpus it was calibrated on. That choice is paid for in detection rate. At these operating points the synthetic-video detector flagged 54.6% of generated video and the face classifier flagged 44% of face swaps, so a great deal of manipulated media is correctly not flagged as HIGH.":
      "Her eşik, meşru görüntüleri neredeyse hiç işaretlememek üzere ayarlandı: hiçbir dedektör, kalibre edildiği derlemdeki hiçbir gerçek klibi işaretlemedi. Bu tercihin bedeli tespit oranıyla ödenir. Bu çalışma noktalarında sentetik video dedektörü üretilmiş videoların %54.6'sını, yüz sınıflandırıcısı ise yüz değiştirmelerin %44'ünü işaretledi; dolayısıyla manipüle edilmiş medyanın büyük bir kısmı doğru biçimde HIGH olarak işaretlenmez.",
    "The mouth-dynamics model was calibrated separately, over a smaller corpus: 40 clips of one dataset, where it flagged all 20 face swaps and none of the 20 genuine clips. That dataset lies inside the model's training distribution, so its perfect separation there is a property of that corpus and not a claim about media in general — and 20 genuine clips bound its false-positive rate more loosely than the 54 behind the other two.":
      "Ağız dinamiği modeli ayrıca, daha küçük bir derlem üzerinde kalibre edildi: tek bir veri kümesinden 40 klip; burada 20 yüz değiştirmenin tamamını işaretledi ve 20 gerçek klibin hiçbirini işaretlemedi. Bu veri kümesi modelin eğitim dağılımının içinde yer alır; dolayısıyla oradaki kusursuz ayrımı genel olarak medya hakkında bir iddia değil, o derlemin bir özelliğidir — ve 20 gerçek klip, yanlış pozitif oranını diğer ikisinin arkasındaki 54 klibe göre daha gevşek sınırlar.",
    "The three detectors cover different things and are read independently. None scoring low is evidence against another: in the R4-T1 study the first two never agreed on a single clip, each was blind to the manipulation family the other was calibrated for, and the third asks a question neither of them asks — how a mouth moves over consecutive frames, rather than how a frame looks. It also needs a face tracked through 25 consecutive frames, so it abstains outright on media the other two score without difficulty.":
      "Üç dedektör farklı şeyleri kapsar ve bağımsız olarak okunur. Birinin düşük skor alması bir diğerine karşı kanıt değildir: R4-T1 çalışmasında ilk ikisi tek bir klipte bile hiç uyuşmadı, her biri diğerinin kalibre edildiği manipülasyon ailesine karşı kördü ve üçüncüsü ikisinin de sormadığı bir soru sorar — bir karenin nasıl göründüğünü değil, bir ağzın ardışık kareler boyunca nasıl hareket ettiğini. Ayrıca 25 ardışık kare boyunca izlenen bir yüze ihtiyaç duyar; bu nedenle diğer ikisinin zorlanmadan skorladığı medyada doğrudan çekimser kalır.",
    "The two detectors cover different things and are read independently. Neither one scoring low is evidence against the other: in the calibration study the two never agreed on a single clip, and each was blind to the manipulation family the other was calibrated for.":
      "İki dedektör farklı şeyleri kapsar ve bağımsız olarak okunur. Birinin düşük skor alması diğerine karşı kanıt değildir: kalibrasyon çalışmasında ikisi tek bir klipte bile hiç uyuşmadı ve her biri, diğerinin kalibre edildiği manipülasyon ailesine karşı kördü.",
    "This decision was taken under a single-detector ruleset that is validated for generated video and is not validated for face-swap detection. Absence of HIGH risk does not rule out face manipulation.":
      "Bu karar, üretilmiş video için doğrulanmış ve yüz değiştirme tespiti için doğrulanmamış tek dedektörlü bir kural seti altında alındı. HIGH riskin olmaması yüz manipülasyonunu dışlamaz.",
    "Snapshot.":
      "Anlık görüntü.",
    "Enrichment state":
      "Zenginleştirme durumu",
    "Decision state":
      "Karar durumu",
    "This states what had run when this page was rendered. A copy printed or exported before the supplementary detectors finished records the state above as it stood at that moment; the assessment it accompanies is final either way, and no detector named here can reach it under this ruleset.":
      "Bu, bu sayfa oluşturulduğunda neyin çalışmış olduğunu belirtir. Ek dedektörler bitmeden önce yazdırılan ya da dışa aktarılan bir kopya, yukarıdaki durumu o anki haliyle kaydeder; eşlik ettiği değerlendirme her iki durumda da nihaidir ve burada adı geçen hiçbir dedektör bu kural setinde ona ulaşamaz.",
    "Direct-risk evidence. The figures below are NVIDIA's own output on NVIDIA's own scale.":
      "Doğrudan risk kanıtı. Aşağıdaki değerler NVIDIA'nın kendi çıktısıdır ve NVIDIA'nın kendi ölçeğindedir.",
    "Signal type":
      "Sinyal türü",
    "Status":
      "Durum",
    "Provider version (NVCF function ID)":
      "Sağlayıcı sürümü (NVCF işlev kimliği)",
    "NVIDIA synthetic probability":
      "NVIDIA sentetik olasılığı",
    "NVIDIA aggregate logit":
      "NVIDIA toplu logit değeri",
    "Clips aggregated by NVIDIA":
      "NVIDIA'nın topladığı klipler",
    "The probability is the provider's score for its own detector, shown as returned. It is NVIDIA evidence, not an InspectRoot confidence, and it is not the risk classification above.":
      "Olasılık, sağlayıcının kendi dedektörü için verdiği skordur ve döndürüldüğü haliyle gösterilir. NVIDIA kanıtıdır, bir InspectRoot güven değeri değildir ve yukarıdaki risk sınıflandırması değildir.",
    "Persisted strongest clips":
      "Kayıtlı en güçlü klipler",
    "(highest logit first)":
      "(en yüksek logit önce)",
    "No clip evidence is stored for this signal.":
      "Bu sinyal için kayıtlı klip kanıtı yok.",
    "Frame index":
      "Kare indeksi",
    "Raw logit":
      "Ham logit",
    "The frame index is NVIDIA's own index for the clip's middle frame. These are the strongest clips the provider reported, not a claim that manipulation occurs at those frames.":
      "Kare indeksi, NVIDIA'nın klibin orta karesi için kullandığı kendi indeksidir. Bunlar sağlayıcının bildirdiği en güçlü kliplerdir; manipülasyonun bu karelerde gerçekleştiğine dair bir iddia değildir.",
    "C2PA provenance":
      "C2PA köken bilgisi",
    "What the stored artifact itself claims about its origin. It was assembled by InspectRoot from separate streams, so any credentials the source may have published alongside them are not expected to survive into it.":
      "Saklanan artefaktın kendi kökeni hakkında ne iddia ettiği. Ayrı akışlardan InspectRoot tarafından birleştirildi; bu nedenle kaynağın bu akışlarla birlikte yayımlamış olabileceği kimlik bilgilerinin ona taşınması beklenmez.",
    "What the file itself claims about its origin, read from the forensic original.":
      "Dosyanın kendi kökeni hakkında ne iddia ettiği, adli orijinalden okunmuştur.",
    "C2PA SDK version":
      "C2PA SDK sürümü",
    "Manifest present in the file":
      "Dosyada manifest var",
    "unknown — the reading failed":
      "bilinmiyor — okuma başarısız oldu",
    "Validation state":
      "Doğrulama durumu",
    "Claim generator":
      "Talep (claim) oluşturucu",
    "Signature issuer":
      "İmzayı veren",
    "Remote manifest URL":
      "Uzak manifest URL'si",
    "Any remote manifest URL was recorded and never fetched.":
      "Herhangi bir uzak manifest URL'si kaydedildi ve hiçbir zaman getirilmedi.",
    "These bytes were assembled by InspectRoot from separate video and audio streams, so this reading describes the stored artifact and not a file the source published. Reading it as a statement about the source would be a mistake in either direction.":
      "Bu baytlar ayrı video ve ses akışlarından InspectRoot tarafından birleştirildi; dolayısıyla bu okuma, kaynağın yayımladığı bir dosyayı değil saklanan artefaktı tanımlar. Bunu kaynak hakkında bir ifade olarak okumak, her iki yönde de bir hata olur.",
    "Active speaker (cross-modal speaking evidence)":
      "Aktif konuşmacı (çapraz kipli konuşma kanıtı)",
    "Where a tracked face was observed speaking. This is not a deepfake detector.":
      "İzlenen bir yüzün nerede konuşurken gözlemlendiği. Bu bir deepfake dedektörü değildir.",
    "Provider version":
      "Sağlayıcı sürümü",
    "Speaking segments found":
      "Bulunan konuşma bölümleri",
    "Stored timeline truncated":
      "Kayıtlı zaman çizelgesi kısaltıldı",
    "The detector ran and recorded no speaking segments. That is an observation about this media, not a missing reading.":
      "Dedektör çalıştı ve hiçbir konuşma bölümü kaydetmedi. Bu, eksik bir okuma değil, bu medya hakkında bir gözlemdir.",
    "No speaking timeline is stored for this signal.":
      "Bu sinyal için kayıtlı bir konuşma zaman çizelgesi yok.",
    "Start":
      "Başlangıç",
    "End":
      "Bitiş",
    "Face ID":
      "Yüz kimliği",
    "Diarized speaker":
      "Ayrıştırılmış konuşmacı",
    "no matched voice":
      "eşleşen ses yok",
    "The face ID is the provider's own identifier for a tracked face; the speaker label is the diarized voice matched to it. This timeline says who was speaking when. It makes no claim about whether the media is genuine.":
      "Yüz kimliği, sağlayıcının izlenen bir yüz için kullandığı kendi tanımlayıcısıdır; konuşmacı etiketi, ona eşleştirilen ayrıştırılmış (diarize edilmiş) sestir. Bu zaman çizelgesi kimin ne zaman konuştuğunu söyler. Medyanın gerçek olup olmadığı hakkında hiçbir iddiada bulunmaz.",
    "AASIST audio evidence":
      "AASIST ses kanıtı",
    "Raw model output per preprocessing window. No threshold, no calibration, no classes.":
      "Ön işleme penceresi başına ham model çıktısı. Eşik yok, kalibrasyon yok, sınıf yok.",
    "Checkpoint":
      "Checkpoint",
    "Windows produced":
      "Üretilen pencereler",
    "Windows stored":
      "Kayıtlı pencereler",
    "Stored windows truncated":
      "Kayıtlı pencereler kısaltıldı",
    "The reading succeeded and stored no windows.":
      "Okuma başarılı oldu ve hiçbir pencere kaydetmedi.",
    "No audio evidence windows are stored for this signal.":
      "Bu sinyal için kayıtlı ses kanıtı penceresi yok.",
    "Window":
      "Pencere",
    "Raw logit[0]":
      "Ham logit[0]",
    "Bona fide logit":
      "Bona fide logit",
    "These figures are raw model output. They are {probabilities}, {confidence}, and {decisions}. The checkpoint publishes no threshold, no calibration and no classes, so no classification is derived from them and none should be read into them.":
      "Bu değerler ham model çıktısıdır. Bunlar {probabilities}, {confidence} ve {decisions}. Checkpoint hiçbir eşik, kalibrasyon ya da sınıf yayımlamaz; bu nedenle bunlardan hiçbir sınıflandırma türetilmez ve bunlara hiçbir sınıflandırma yüklenmemelidir.",
    "not probabilities":
      "olasılık değildir",
    "not confidence values":
      "güven değeri değildir",
    "not Fake/Real decisions":
      "Sahte/Gerçek kararı değildir",
    "The time bounds are {windows} — where the audio was cut before being given to the model. They are not model-detected manipulation timestamps, and the model reports no timeline of its own.":
      "Zaman sınırları {windows} — sesin modele verilmeden önce kesildiği yerler. Bunlar modelin tespit ettiği manipülasyon zaman damgaları değildir ve model kendine ait bir zaman çizelgesi bildirmez.",
    "InspectRoot preprocessing windows":
      "InspectRoot ön işleme pencereleridir",
    "EfficientNet-B7 face manipulation detector":
      "EfficientNet-B7 yüz manipülasyonu dedektörü",
    "Calibrated evidence. The score below is the model's own output, banded against a threshold measured for it in R4-T1.":
      "Kalibre edilmiş kanıt. Aşağıdaki skor modelin kendi çıktısıdır ve R4-T1'de onun için ölçülmüş bir eşiğe göre bantlanmıştır.",
    "Independent evidence. The score below is the model's own output and is not part of the risk classification.":
      "Bağımsız kanıt. Aşağıdaki skor modelin kendi çıktısıdır ve risk sınıflandırmasının bir parçası değildir.",
    "Model score":
      "Model skoru",
    "Frames sampled":
      "Örneklenen kareler",
    "Frames decoded":
      "Çözülen kareler",
    "Frames with a detected face":
      "Yüz tespit edilen kareler",
    "This reading did not produce a score. A clip in which no face was found is the ordinary case, and it means the classifier was never asked — it is not a finding that the media is genuine.":
      "Bu okuma bir skor üretmedi. İçinde yüz bulunmayan bir klip olağan durumdur ve sınıflandırıcıya hiç sorulmadığı anlamına gelir — medyanın gerçek olduğuna dair bir bulgu değildir.",
    "The score is the mean of the model's per-frame output over the frames above, shown exactly as the model produced it. It is {notProbability} and {notDecision}.":
      "Skor, modelin yukarıdaki kareler üzerindeki kare başına çıktısının ortalamasıdır ve modelin ürettiği haliyle aynen gösterilir. Bu skor {notProbability} ve {notDecision}.",
    "not a probability that this media is manipulated":
      "bu medyanın manipüle edilmiş olma olasılığı değildir",
    "not a Fake/Real decision":
      "bir Sahte/Gerçek kararı değildir",
    "Under this ruleset the score is compared against {threshold} — the threshold measured for this detector in R4-T1, and for this detector only. It is never averaged or combined with the other detectors' scores; each is a separate question with a separate answer, and the risk classification names which of them decided.":
      "Bu kural setinde skor {threshold} ile karşılaştırılır — R4-T1'de bu dedektör için, yalnızca bu dedektör için ölçülmüş eşik. Diğer dedektörlerin skorlarıyla hiçbir zaman ortalanmaz ya da birleştirilmez; her biri ayrı bir yanıtı olan ayrı bir sorudur ve risk sınıflandırması hangisinin karar verdiğini belirtir.",
    "It is {notCalibrated} under this ruleset, no threshold is applied to it, and it does not contribute to the risk classification above and cannot change it. This signal is recorded as an independent forensic fact only.":
      "Bu kural setinde {notCalibrated}; ona hiçbir eşik uygulanmaz, yukarıdaki risk sınıflandırmasına katkıda bulunmaz ve onu değiştiremez. Bu sinyal yalnızca bağımsız bir adli olgu olarak kaydedilir.",
    "not calibrated":
      "kalibre edilmemiştir",
    "LipForensics mouth-dynamics detector":
      "LipForensics ağız dinamiği dedektörü",
    "Calibrated evidence. The score below is the model's own output, banded against a threshold measured for it in R5-T3.":
      "Kalibre edilmiş kanıt. Aşağıdaki skor modelin kendi çıktısıdır ve R5-T3'te onun için ölçülmüş bir eşiğe göre bantlanmıştır.",
    "The score below is the model's own output, shown as this analysis recorded it.":
      "Aşağıdaki skor modelin kendi çıktısıdır ve bu analizin kaydettiği haliyle gösterilir.",
    "Model":
      "Model",
    "Runs sampled":
      "Örneklenen diziler",
    "Runs decoded":
      "Çözülen diziler",
    "Runs with a tracked face":
      "İzlenen yüz içeren diziler",
    "This reading did not produce a score. A clip in which no run held a trackable face throughout is the ordinary case, and it means the model was never asked — it is not a finding that the media is genuine.":
      "Bu okuma bir skor üretmedi. Hiçbir dizinin baştan sona izlenebilir bir yüz içermediği bir klip olağan durumdur ve modele hiç sorulmadığı anlamına gelir — medyanın gerçek olduğuna dair bir bulgu değildir.",
    "The score is the model's output for the runs above — each a stretch of 25 consecutive frames, scored on how the mouth moves across them — shown exactly as the model produced it. It is {notProbability} and {notDecision}.":
      "Skor, modelin yukarıdaki diziler için çıktısıdır — her biri, ağzın bu kareler boyunca nasıl hareket ettiğine göre skorlanan 25 ardışık karelik bir kesittir — ve modelin ürettiği haliyle aynen gösterilir. Bu skor {notProbability} ve {notDecision}.",
    "Despite the model's name, this is {notLipSync}. The model is given no audio at all: it reads the movement of the mouth in the picture and nothing else, and what it was trained to separate is forged facial motion from genuine facial motion.":
      "Modelin adına rağmen bu, {notLipSync}. Modele hiç ses verilmez: yalnızca görüntüdeki ağız hareketini okur, başka hiçbir şeyi değil; ayırt etmek üzere eğitildiği şey, sahte yüz hareketini gerçek yüz hareketinden ayırmaktır.",
    "not a measure of audio/video lip synchronisation":
      "ses/video dudak senkronizasyonunun bir ölçüsü değildir",
    "Under this ruleset the score is compared against {threshold} — the threshold measured for this model in R5-T3, and for this model only. That figure is lower than the thresholds above and this does not mean it is more easily convinced: the three numbers are points on three unrelated scales and comparing them to each other is not a comparison of anything. It was measured over 40 clips of a single dataset, which is a smaller study than the one behind the two detectors above.":
      "Bu kural setinde skor {threshold} ile karşılaştırılır — R5-T3'te bu model için, yalnızca bu model için ölçülmüş eşik. Bu değer yukarıdaki eşiklerden düşüktür ve bu, modelin daha kolay ikna olduğu anlamına gelmez: üç sayı, üç ilişkisiz ölçekteki noktalardır ve bunları birbiriyle karşılaştırmak hiçbir şeyin karşılaştırması değildir. Tek bir veri kümesinden 40 klip üzerinde ölçüldü; bu, yukarıdaki iki dedektörün arkasındaki çalışmadan daha küçük bir çalışmadır.",
    "An operating point was measured for this model in R5-T3 and under this ruleset it is {notApplied}: the model is read as {evidenceOnly}, and its score cannot reach the assessment above or change it — including when it stands above that threshold. R7-T6 withdrew it from the rules after R7-T5 measured what that operating point did to genuine media. This signal is recorded as an independent forensic fact.":
      "Bu model için R5-T3'te bir çalışma noktası ölçüldü ve bu kural setinde {notApplied}: model {evidenceOnly} olarak okunur ve skoru yukarıdaki değerlendirmeye ulaşamaz ya da onu değiştiremez — o eşiğin üzerinde olduğunda bile. R7-T5 bu çalışma noktasının gerçek medyaya ne yaptığını ölçtükten sonra R7-T6 onu kurallardan çıkardı. Bu sinyal bağımsız bir adli olgu olarak kaydedilir.",
    "not applied":
      "uygulanmaz",
    "evidence only":
      "yalnızca kanıt",
    "This model was within the scope of the ruleset that decided this analysis, and {unstated} — whether it applied the operating point measured in R5-T3 or read the score as evidence only. The score below is shown as recorded, and nothing further is said here about the part it played, because this record does not say.":
      "Bu model, bu analize karar veren kural setinin kapsamındaydı ve {unstated} — R5-T3'te ölçülen çalışma noktasını uygulayıp uygulamadığı ya da skoru yalnızca kanıt olarak okuyup okumadığı. Aşağıdaki skor kaydedildiği haliyle gösterilir ve oynadığı rol hakkında burada başka bir şey söylenmez, çünkü bu kayıt bunu söylemez.",
    "this record does not state what that ruleset could do with it":
      "bu kayıt, o kural setinin onunla ne yapabildiğini belirtmez",
    "It is {notSecondReading}. That model judges the appearance of a face crop; this one judges movement over time. The two figures are on different scales and are never averaged, compared or reconciled — agreement between them would not strengthen a finding, and disagreement does not weaken one. Where both reached their own thresholds the risk classification records that as two independent findings and not as a stronger one.":
      "Bu, {notSecondReading}. O model bir yüz kırpıntısının görünümünü değerlendirir; bu ise zaman içindeki hareketi. İki değer farklı ölçeklerdedir ve hiçbir zaman ortalanmaz, karşılaştırılmaz ya da uzlaştırılmaz — aralarındaki uyum bir bulguyu güçlendirmez, uyumsuzluk da zayıflatmaz. Her ikisi de kendi eşiğine ulaştığında risk sınıflandırması bunu daha güçlü tek bir bulgu olarak değil, iki bağımsız bulgu olarak kaydeder.",
    "not a second reading of the face-manipulation score above":
      "yukarıdaki yüz manipülasyonu skorunun ikinci bir okuması değildir",
    "Forensic Evidence Report":
      "Adli Kanıt Raporu",
    "Analysis ID":
      "Analiz kimliği",
    "Analysis status":
      "Analiz durumu",
    "Analysis timestamp (UTC)":
      "Analiz zaman damgası (UTC)",
    "Size on disk":
      "Diskteki boyut",
    "{bytes} bytes":
      "{bytes} bayt",
    "Authenticity and provenance":
      "Özgünlük ve köken bilgisi",
    "A separate question from the assessment above, answered from separate evidence. Provenance is what the file itself carries about where it came from and who signed for it; the detector evidence further down is what was measured about the picture and the sound. Neither reaches the other: no assessment on this page was moved by the provenance state below, and the provenance state below was not moved by any assessment on this page. Nothing here reports the media as authentic or manipulated, and no provenance state on this page can.":
      "Yukarıdaki değerlendirmeden ayrı bir soru, ayrı kanıtlarla yanıtlanır. Köken bilgisi, dosyanın kendisinin nereden geldiği ve kimin imzaladığı hakkında taşıdığı bilgidir; aşağıdaki dedektör kanıtı ise görüntü ve ses hakkında ölçülenlerdir. Hiçbiri diğerine ulaşmaz: bu sayfadaki hiçbir değerlendirme aşağıdaki köken bilgisi durumundan etkilenmedi ve aşağıdaki köken bilgisi durumu bu sayfadaki hiçbir değerlendirmeden etkilenmedi. Buradaki hiçbir şey medyayı özgün ya da manipüle edilmiş olarak raporlamaz ve bu sayfadaki hiçbir köken bilgisi durumu bunu yapamaz.",
    "Technical forensic evidence":
      "Teknik adli kanıt",
    "The record behind the assessment: the decision as it was stored, the artifact it was taken on, and every detector reading kept for it. None of it is recomputed here.":
      "Değerlendirmenin arkasındaki kayıt: kaydedildiği haliyle karar, kararın alındığı artefakt ve bunun için saklanan her dedektör okuması. Hiçbiri burada yeniden hesaplanmaz.",
    "Independent detector evidence":
      "Bağımsız dedektör kanıtı",
    "Each source is recorded separately and none of them is combined into the other.":
      "Her kaynak ayrı kaydedilir ve hiçbiri diğeriyle birleştirilmez.",
    "Two of them can reach the assessment above — the synthetic-video detector and the face-manipulation classifier — each against a threshold measured for it alone, and never by pooling their scores. The mouth-dynamics model is calibrated and is recorded here as independent evidence; under this ruleset it cannot reach that assessment, and a reading it failed to produce does not reduce the decision coverage stated there. Speaking evidence and audio evidence have no calibrated threshold at all.":
      "Bunlardan ikisi yukarıdaki değerlendirmeye ulaşabilir — sentetik video dedektörü ve yüz manipülasyonu sınıflandırıcısı — her biri yalnızca kendisi için ölçülmüş bir eşiğe göre ve hiçbir zaman skorlarını birleştirerek değil. Ağız dinamiği modeli kalibre edilmiştir ve burada bağımsız kanıt olarak kaydedilir; bu kural setinde o değerlendirmeye ulaşamaz ve üretemediği bir okuma orada belirtilen karar kapsamını azaltmaz. Konuşma kanıtı ve ses kanıtının hiçbir kalibre edilmiş eşiği yoktur.",
    "Two of them can reach the risk classification above — the synthetic-video detector and the face-manipulation classifier — each against a threshold measured for it alone, and never by pooling their scores. The mouth-dynamics model is calibrated and is recorded here as independent evidence, but under this ruleset it cannot change that classification. Neither can speaking evidence or audio evidence, which have no calibrated threshold at all.":
      "Bunlardan ikisi yukarıdaki risk sınıflandırmasına ulaşabilir — sentetik video dedektörü ve yüz manipülasyonu sınıflandırıcısı — her biri yalnızca kendisi için ölçülmüş bir eşiğe göre ve hiçbir zaman skorlarını birleştirerek değil. Ağız dinamiği modeli kalibre edilmiştir ve burada bağımsız kanıt olarak kaydedilir, ancak bu kural setinde o sınıflandırmayı değiştiremez. Hiçbir kalibre edilmiş eşiği olmayan konuşma kanıtı ve ses kanıtı da değiştiremez.",
    "Three of them are calibrated and can reach the risk classification above — the synthetic-video detector, the face-manipulation classifier and the mouth-dynamics model — each against a threshold measured for it alone, and never by pooling their scores. Speaking evidence and audio evidence have no calibrated threshold and cannot change that classification.":
      "Bunlardan üçü kalibre edilmiştir ve yukarıdaki risk sınıflandırmasına ulaşabilir — sentetik video dedektörü, yüz manipülasyonu sınıflandırıcısı ve ağız dinamiği modeli — her biri yalnızca kendisi için ölçülmüş bir eşiğe göre ve hiçbir zaman skorlarını birleştirerek değil. Konuşma kanıtı ve ses kanıtının kalibre edilmiş eşiği yoktur ve o sınıflandırmayı değiştiremez.",
    "Two of them are calibrated and can reach the risk classification above — the synthetic-video detector and the face-manipulation classifier — each against a threshold measured for it alone, and never by pooling their scores. Speaking evidence, mouth-dynamics evidence and audio evidence have no calibrated threshold and cannot change that classification.":
      "Bunlardan ikisi kalibre edilmiştir ve yukarıdaki risk sınıflandırmasına ulaşabilir — sentetik video dedektörü ve yüz manipülasyonu sınıflandırıcısı — her biri yalnızca kendisi için ölçülmüş bir eşiğe göre ve hiçbir zaman skorlarını birleştirerek değil. Konuşma kanıtı, ağız dinamiği kanıtı ve ses kanıtının kalibre edilmiş eşiği yoktur ve o sınıflandırmayı değiştiremez.",
    "Only the synthetic-video detector contributes to the risk classification above; speaking evidence, face-manipulation evidence, mouth-dynamics evidence and audio evidence are recorded as independent forensic facts and cannot change that classification.":
      "Yukarıdaki risk sınıflandırmasına yalnızca sentetik video dedektörü katkıda bulunur; konuşma kanıtı, yüz manipülasyonu kanıtı, ağız dinamiği kanıtı ve ses kanıtı bağımsız adli olgular olarak kaydedilir ve o sınıflandırmayı değiştiremez.",
    "This report is a rendering of forensic evidence persisted by InspectRoot for the analysis named above. It is not cryptographically signed, and reproducing it does not establish that its contents are unaltered. The SHA-256 shown is the hash of the submitted media, not of this report — and, where the media was normalized for detection, not of the derivative the detectors read.":
      "Bu rapor, yukarıda adı geçen analiz için InspectRoot tarafından saklanan adli kanıtın bir sunumudur. Kriptografik olarak imzalanmamıştır ve yeniden üretilmesi içeriğinin değiştirilmediğini kanıtlamaz. Gösterilen SHA-256, bu raporun değil gönderilen medyanın hash değeridir — ve medya tespit için normalleştirildiyse, dedektörlerin okuduğu türevin de değildir.",
    "Nothing in this document makes a binary Fake/Real authenticity determination. Where applicable, the assessment above reports calibrated manipulation evidence under the stated ruleset. Provenance is reported separately and does not determine that assessment.":
      "Bu belgedeki hiçbir şey ikili bir Sahte/Gerçek özgünlük belirlemesi yapmaz. Uygun olduğu durumlarda yukarıdaki değerlendirme, belirtilen kural seti altında kalibre edilmiş manipülasyon kanıtını raporlar. Köken bilgisi ayrı raporlanır ve bu değerlendirmeyi belirlemez.",
    "Provenance could not be evaluated":
      "Köken bilgisi değerlendirilemedi",
    "The provenance reading did not complete, so whether this file carries Content Credentials is unknown.":
      "Köken bilgisi okuması tamamlanmadı; bu nedenle bu dosyanın Content Credentials taşıyıp taşımadığı bilinmiyor.",
    "Provenance could not be evaluated. This is a gap in the reading, not a finding about the media, and it is not evidence of manipulation or of authenticity.":
      "Köken bilgisi değerlendirilemedi. Bu, medya hakkında bir bulgu değil okumadaki bir boşluktur ve ne manipülasyonun ne de özgünlüğün kanıtıdır.",
    "No provenance credentials found":
      "Köken bilgisi kimlik bilgisi bulunamadı",
    "The file was read successfully and carries no Content Credentials, so there is no signed origin record to examine.":
      "Dosya başarıyla okundu ve hiçbir Content Credentials taşımıyor; dolayısıyla incelenecek imzalı bir köken kaydı yok.",
    "The absence of provenance credentials is not evidence of manipulation. Most media on the internet does not carry these credentials.":
      "Köken bilgisi kimlik bilgilerinin yokluğu manipülasyonun kanıtı değildir. İnternetteki medyanın çoğu bu kimlik bilgilerini taşımaz.",
    "Provenance credentials present":
      "Köken bilgisi kimlik bilgileri mevcut",
    "The file was read successfully and carries a provenance manifest, recorded in full in the C2PA evidence below.":
      "Dosya başarıyla okundu ve aşağıdaki C2PA kanıtında eksiksiz kaydedilen bir köken bilgisi manifesti taşıyor.",
    "A provenance manifest/credential is present. This provides origin data but its validation state must be examined separately, and it does not automatically guarantee the media has not been tampered with prior to signing.":
      "Bir köken bilgisi manifesti/kimlik bilgisi mevcut. Bu köken verisi sağlar, ancak doğrulama durumu ayrıca incelenmelidir ve medyanın imzalanmadan önce kurcalanmadığını otomatik olarak garanti etmez.",
    "Manipulation detected":
      "Manipülasyon tespit edildi",
    "One or more calibrated decision detectors reached their operating point.":
      "Kalibre edilmiş karar dedektörlerinden biri ya da birkaçı çalışma noktasına ulaştı.",
    "This identifies calibrated manipulation evidence. It does not establish the original source or provenance of the media.":
      "Bu, kalibre edilmiş manipülasyon kanıtını tanımlar. Medyanın orijinal kaynağını ya da köken bilgisini belirlemez.",
    "No calibrated manipulation signal detected":
      "Kalibre edilmiş manipülasyon sinyali tespit edilmedi",
    "The completed decision detectors produced no threshold-reaching manipulation signal.":
      "Tamamlanan karar dedektörleri eşiğe ulaşan hiçbir manipülasyon sinyali üretmedi.",
    "This does not prove that the media is authentic, genuine, or source-verified.":
      "Bu, medyanın özgün, gerçek ya da kaynağı doğrulanmış olduğunu kanıtlamaz.",
    "Inconclusive":
      "Sonuçsuz",
    "InspectRoot could not complete the calibrated automated assessment because one or more required decision detectors did not produce a usable reading.":
      "InspectRoot, gerekli karar dedektörlerinden biri ya da birkaçı kullanılabilir bir okuma üretmediği için kalibre edilmiş otomatik değerlendirmeyi tamamlayamadı.",
    "This result is neither evidence of manipulation nor evidence of authenticity.":
      "Bu sonuç ne manipülasyonun ne de özgünlüğün kanıtıdır.",
    "High risk":
      "Yüksek risk",
    "Medium risk":
      "Orta risk",
    "Unsupported":
      "Desteklenmiyor",
    "Reached its threshold":
      "Eşiğine ulaştı",
    "Did not reach its threshold":
      "Eşiğine ulaşmadı",
    "Threshold could not be interpreted":
      "Eşik yorumlanamadı",
    "This detector's score reached the operating point measured for it under this ruleset. Both figures are shown as the record holds them.":
      "Bu dedektörün skoru, bu kural setinde onun için ölçülmüş çalışma noktasına ulaştı. Her iki değer de kaydın tuttuğu haliyle gösterilir.",
    "This detector's score did not reach the operating point measured for it. That is not a finding that the media is genuine: a detector reports this for a manipulation family it cannot see just as readily as for unmanipulated media.":
      "Bu dedektörün skoru, onun için ölçülmüş çalışma noktasına ulaşmadı. Bu, medyanın gerçek olduğuna dair bir bulgu değildir: bir dedektör bunu, manipüle edilmemiş medya için olduğu kadar göremediği bir manipülasyon ailesi için de bildirir.",
    "This detector contributed no reading the decision could use, so no score and no threshold were read against each other and none is shown. Silence from a detector is not evidence that the media was not manipulated.":
      "Bu dedektör, kararın kullanabileceği hiçbir okuma sağlamadı; bu nedenle hiçbir skor ve eşik birbirine göre okunmadı ve hiçbiri gösterilmez. Bir dedektörün sessizliği, medyanın manipüle edilmediğinin kanıtı değildir.",
    "The detector's own figures are readable, but the threshold they would have to be read against under this ruleset could not be resolved. No outcome is reported, because reading them against another ruleset's threshold would describe a comparison that was never made.":
      "Dedektörün kendi değerleri okunabilir, ancak bu kural setinde bunların karşılaştırılması gereken eşik çözümlenemedi. Hiçbir sonuç bildirilmez, çünkü bunları başka bir kural setinin eşiğine göre okumak hiç yapılmamış bir karşılaştırmayı tanımlamak olur.",
    "Condition not interpretable by this build":
      "Koşul bu sürüm tarafından yorumlanamıyor",
    "Supplementary evidence not requested":
      "Ek kanıt istenmedi",
    "This analysis was submitted as a quick scan, so its supplementary detectors were not run. That is a scheduling choice and not a failure: the assessment above is complete, was taken from the same detectors under the same ruleset, and would not be changed by the evidence below being gathered.":
      "Bu analiz hızlı tarama olarak gönderildi; bu nedenle ek dedektörleri çalıştırılmadı. Bu bir başarısızlık değil, bir zamanlama tercihidir: yukarıdaki değerlendirme tamamdır, aynı kural seti altında aynı dedektörlerden alınmıştır ve aşağıdaki kanıtın toplanması onu değiştirmezdi.",
    "Supplementary evidence queued":
      "Ek kanıt kuyrukta",
    "The assessment above is final. Supplementary detectors are queued and have not started. Nothing they produce can change the assessment, the rule that was applied, or the decision coverage stated with it.":
      "Yukarıdaki değerlendirme nihaidir. Ek dedektörler kuyruktadır ve başlamadı. Üretecekleri hiçbir şey değerlendirmeyi, uygulanan kuralı ya da onunla birlikte belirtilen karar kapsamını değiştiremez.",
    "Supplementary evidence still running":
      "Ek kanıt hâlâ çalışıyor",
    "The assessment above is final and this report is complete as a decision. Supplementary detectors are still running, so the evidence panels below may be incomplete at this moment. Nothing they produce can change the assessment, the rule that was applied, or the decision coverage stated with it.":
      "Yukarıdaki değerlendirme nihaidir ve bu rapor bir karar olarak tamamdır. Ek dedektörler hâlâ çalışıyor; bu nedenle aşağıdaki kanıt panelleri şu anda eksik olabilir. Üretecekleri hiçbir şey değerlendirmeyi, uygulanan kuralı ya da onunla birlikte belirtilen karar kapsamını değiştiremez.",
    "Supplementary evidence complete":
      "Ek kanıt tamamlandı",
    "Every supplementary detector reached a terminal state and each of them answered. A detector that reported nothing to score — no trackable face, no audio stream — answered the question it was asked, and is counted here as having done so.":
      "Her ek dedektör nihai bir duruma ulaştı ve her biri yanıt verdi. Skorlanacak bir şey bildirmeyen bir dedektör — izlenebilir yüz yok, ses akışı yok — kendisine sorulan soruyu yanıtlamıştır ve burada yanıt vermiş sayılır.",
    "Supplementary evidence incomplete":
      "Ek kanıt eksik",
    "Every supplementary detector reached a terminal state; some answered and some did not. The assessment above is unaffected and unqualified by this — none of these detectors can reach it under this ruleset. Which component did what is listed below.":
      "Her ek dedektör nihai bir duruma ulaştı; bazıları yanıt verdi, bazıları vermedi. Yukarıdaki değerlendirme bundan etkilenmez ve bununla nitelenmez — bu dedektörlerin hiçbiri bu kural setinde ona ulaşamaz. Hangi bileşenin ne yaptığı aşağıda listelenmiştir.",
    "Supplementary evidence unavailable":
      "Ek kanıt kullanılamıyor",
    "Every supplementary detector reached a terminal state and none of them answered. The assessment above still stands, unchanged and unqualified: an analysis with a verdict and no supplementary evidence is a complete decision with a thin report, not a broken analysis.":
      "Her ek dedektör nihai bir duruma ulaştı ve hiçbiri yanıt vermedi. Yukarıdaki değerlendirme değişmeden ve nitelenmeden geçerliliğini korur: bir kararı olan ve ek kanıtı olmayan bir analiz, bozuk bir analiz değil, raporu ince olan tam bir karardır.",
    "Supplementary evidence state not interpretable by this build":
      "Ek kanıt durumu bu sürüm tarafından yorumlanamıyor",
    "The API reported a state this build has no wording for. It is shown below as the record's own word, and nothing is concluded from it.":
      "API, bu sürümün karşılığında bir ifadesi olmayan bir durum bildirdi. Aşağıda kaydın kendi sözcüğü olarak gösterilir ve ondan hiçbir sonuç çıkarılmaz.",
    "This report is a snapshot. Deep analysis enrichment is currently processing, and the detectors below may not have produced a reading yet. The assessment above is already final and is not provisional: what is still running is supplementary evidence, and nothing it produces can change the assessment, the rule that was applied, or the decision coverage stated with it.":
      "Bu rapor bir anlık görüntüdür. Derin analiz zenginleştirmesi şu anda işleniyor ve aşağıdaki dedektörler henüz bir okuma üretmemiş olabilir. Yukarıdaki değerlendirme zaten nihaidir ve geçici değildir: hâlâ çalışan şey ek kanıttır ve ürettiği hiçbir şey değerlendirmeyi, uygulanan kuralı ya da onunla birlikte belirtilen karar kapsamını değiştiremez.",
    "Not requested":
      "İstenmedi",
    "Pending":
      "Beklemede",
    "Processing":
      "İşleniyor",
    "Completed with reading":
      "Okumayla tamamlandı",
    "Completed with abstention":
      "Çekimser kalarak tamamlandı",
    "Failed":
      "Başarısız",
    "Processing — no reading available yet":
      "İşleniyor — henüz okuma yok",
    "Completed — reading not found in this record":
      "Tamamlandı — okuma bu kayıtta bulunamadı",
    "Completed":
      "Tamamlandı",
    "State not interpretable by this build":
      "Durum bu sürüm tarafından yorumlanamıyor",
    "That is not a failed reading — nothing recorded one, so there is no evidence from this source either way.":
      "Bu başarısız bir okuma değildir — hiçbir şey bir okuma kaydetmedi; dolayısıyla bu kaynaktan ne yönde olursa olsun hiçbir kanıt yok.",
    "This detector has not produced a reading yet — it is part of the deep analysis that is still running for this report. That is not a finding that there is no evidence from this source; it means this source has not answered yet.":
      "Bu dedektör henüz bir okuma üretmedi — bu rapor için hâlâ çalışmakta olan derin analizin bir parçasıdır. Bu, bu kaynaktan kanıt olmadığına dair bir bulgu değildir; bu kaynağın henüz yanıt vermediği anlamına gelir.",
    "NVIDIA's synthetic-video detector scored at or above 0.98, the threshold this ruleset banded on.":
      "NVIDIA'nın sentetik video dedektörü, bu kural setinin bantlama yaptığı eşik olan 0.98 veya üzerinde skor aldı.",
    "This ruleset read one detector. It was validated for generated video and was not validated for face-swap detection, so this result does not cover face manipulation either way.":
      "Bu kural seti tek bir dedektör okudu. Üretilmiş video için doğrulandı ve yüz değiştirme tespiti için doğrulanmadı; dolayısıyla bu sonuç, yüz manipülasyonunu hiçbir yönde kapsamaz.",
    "Scored at or above 0.98.":
      "0.98 veya üzerinde skor aldı.",
    "Not read by this ruleset. Under p7-v1.0.0 the face-manipulation score was stored as independent forensic evidence and was not eligible to move the risk level.":
      "Bu kural seti tarafından okunmadı. p7-v1.0.0 altında yüz manipülasyonu skoru bağımsız adli kanıt olarak saklandı ve risk düzeyini değiştirmeye uygun değildi.",
    "Not read by this ruleset. The mouth-dynamics score was stored as independent forensic evidence and had no measured threshold, so it was not eligible to move the risk level.":
      "Bu kural seti tarafından okunmadı. Ağız dinamiği skoru bağımsız adli kanıt olarak saklandı ve ölçülmüş bir eşiği yoktu; bu nedenle risk düzeyini değiştirmeye uygun değildi.",
    "NVIDIA's synthetic-video detector produced a usable reading below 0.98.":
      "NVIDIA'nın sentetik video dedektörü 0.98'in altında kullanılabilir bir okuma üretti.",
    "This ruleset read one detector and was not validated for face-swap detection. Absence of a high-risk finding here does not rule out face manipulation.":
      "Bu kural seti tek bir dedektör okudu ve yüz değiştirme tespiti için doğrulanmadı. Burada yüksek riskli bir bulgunun olmaması yüz manipülasyonunu dışlamaz.",
    "Scored below 0.98.":
      "0.98'in altında skor aldı.",
    "No usable synthetic-video reading was available from the calibrated deployment.":
      "Kalibre edilmiş dağıtımdan kullanılabilir bir sentetik video okuması elde edilemedi.",
    "This is a statement about the evidence, not about the media.":
      "Bu, medya hakkında değil, kanıt hakkında bir ifadedir.",
    "No usable calibrated reading — missing, failed, or produced by a deployment the threshold was never measured against.":
      "Kullanılabilir kalibre edilmiş okuma yok — eksik, başarısız ya da eşiğin hiç ölçülmediği bir dağıtım tarafından üretilmiş.",
    "NVIDIA's synthetic-video detector answered, but its figures could not be read.":
      "NVIDIA'nın sentetik video dedektörü yanıt verdi, ancak değerleri okunamadı.",
    "No usable reading.":
      "Kullanılabilir okuma yok.",
    "NVIDIA's synthetic-video detector reached its calibrated threshold. That finding alone produced this level.":
      "NVIDIA'nın sentetik video dedektörü kalibre edilmiş eşiğine ulaştı. Bu düzeyi tek başına bu bulgu üretti.",
    "The level rests on one detector's evidence. The other detector's low score is not a second opinion — in the calibration study the two detectors never once agreed, and requiring agreement would have detected nothing at all.":
      "Düzey tek bir dedektörün kanıtına dayanır. Diğer dedektörün düşük skoru ikinci bir görüş değildir — kalibrasyon çalışmasında iki dedektör bir kez bile uyuşmadı ve uyuşma şartı koşmak hiçbir şey tespit etmemekle sonuçlanırdı.",
    "Scored at or above 0.9551, the threshold measured for it. This detector separates generated video well and is the one calibrated for it.":
      "Onun için ölçülmüş eşik olan 0.9551 veya üzerinde skor aldı. Bu dedektör üretilmiş videoyu iyi ayırt eder ve onun için kalibre edilmiş olandır.",
    "Scored below 0.9868 and did not contribute. This detector is calibrated for face swaps and performs worse than chance on generated video, so a low score from it is not evidence against the finding above and was not allowed to reduce the level.":
      "0.9868'in altında skor aldı ve katkıda bulunmadı. Bu dedektör yüz değiştirmeler için kalibre edilmiştir ve üretilmiş videoda rastgeleden daha kötü performans gösterir; dolayısıyla ondan gelen düşük skor yukarıdaki bulguya karşı kanıt değildir ve düzeyi düşürmesine izin verilmedi.",
    "The EfficientNet-B7 face-manipulation classifier reached its calibrated threshold. That finding alone produced this level.":
      "EfficientNet-B7 yüz manipülasyonu sınıflandırıcısı kalibre edilmiş eşiğine ulaştı. Bu düzeyi tek başına bu bulgu üretti.",
    "The level rests on one detector's evidence. This is the finding the previous ruleset could not make: face manipulation was not covered by any calibrated rule before R4-T1.":
      "Düzey tek bir dedektörün kanıtına dayanır. Bu, önceki kural setinin yapamadığı bulgudur: R4-T1'den önce yüz manipülasyonu hiçbir kalibre edilmiş kural tarafından kapsanmıyordu.",
    "Scored below 0.9551 and did not contribute. This detector is near-blind to face swaps — it flagged none of the 50 in the calibration corpus — so a low score from it is not evidence against the finding above and was not allowed to reduce the level.":
      "0.9551'in altında skor aldı ve katkıda bulunmadı. Bu dedektör yüz değiştirmelere neredeyse kördür — kalibrasyon derlemindeki 50 yüz değiştirmenin hiçbirini işaretlemedi — dolayısıyla ondan gelen düşük skor yukarıdaki bulguya karşı kanıt değildir ve düzeyi düşürmesine izin verilmedi.",
    "Scored at or above 0.9868, the threshold measured for it. This detector separates face swaps well and is the one calibrated for them.":
      "Onun için ölçülmüş eşik olan 0.9868 veya üzerinde skor aldı. Bu dedektör yüz değiştirmeleri iyi ayırt eder ve onlar için kalibre edilmiş olandır.",
    "Both calibrated detectors independently reached their own thresholds on this media.":
      "Kalibre edilmiş her iki dedektör de bu medyada bağımsız olarak kendi eşiğine ulaştı.",
    "Two independent findings, reached separately. The scores were not combined, averaged or weighted against each other; each was compared only to its own threshold.":
      "Ayrı ayrı ulaşılmış iki bağımsız bulgu. Skorlar birleştirilmedi, ortalanmadı ya da birbirine göre ağırlıklandırılmadı; her biri yalnızca kendi eşiğiyle karşılaştırıldı.",
    "Scored at or above 0.9551, the threshold measured for it.":
      "Onun için ölçülmüş eşik olan 0.9551 veya üzerinde skor aldı.",
    "Scored at or above 0.9868, the threshold measured for it.":
      "Onun için ölçülmüş eşik olan 0.9868 veya üzerinde skor aldı.",
    "Both detectors read this media and neither reached its own threshold.":
      "Her iki dedektör de bu medyayı okudu ve hiçbiri kendi eşiğine ulaşmadı.",
    "Both a generated-video and a face-manipulation question were asked of this media, and neither was answered above its threshold. That is not a finding that the media is genuine: each detector is deliberately set to a point that almost never flags legitimate footage, which means a great deal of manipulated media also falls below it.":
      "Bu medyaya hem üretilmiş video hem de yüz manipülasyonu sorusu soruldu ve hiçbiri eşiğinin üzerinde yanıtlanmadı. Bu, medyanın gerçek olduğuna dair bir bulgu değildir: her dedektör bilinçli olarak meşru görüntüleri neredeyse hiç işaretlemeyen bir noktaya ayarlanmıştır; bu da manipüle edilmiş medyanın büyük bir kısmının da bu noktanın altında kaldığı anlamına gelir.",
    "Produced a usable calibrated reading below 0.9551.":
      "0.9551'in altında kullanılabilir bir kalibre edilmiş okuma üretti.",
    "Produced a usable calibrated reading below 0.9868.":
      "0.9868'in altında kullanılabilir bir kalibre edilmiş okuma üretti.",
    "Only one of the two calibrated detectors produced a usable reading, and it did not reach its threshold.":
      "Kalibre edilmiş iki dedektörden yalnızca biri kullanılabilir bir okuma üretti ve eşiğine ulaşmadı.",
    "This level was reached with half the coverage of a two-detector result: one of the two questions was never answered for this media. It is a weaker basis than it looks, and it is reported separately for exactly that reason.":
      "Bu düzeye, iki dedektörlü bir sonucun kapsamının yarısıyla ulaşıldı: iki sorudan biri bu medya için hiç yanıtlanmadı. Göründüğünden daha zayıf bir dayanaktır ve tam da bu nedenle ayrı raporlanır.",
    "See this detector's own panel below for whether it produced a reading.":
      "Bir okuma üretip üretmediği için bu dedektörün aşağıdaki kendi paneline bakın.",
    "Neither calibrated detector produced a reading that any rule could be applied to.":
      "Kalibre edilmiş dedektörlerin hiçbiri herhangi bir kuralın uygulanabileceği bir okuma üretmedi.",
    "This is a statement about the evidence, not about the media. Nothing here suggests the media is either genuine or manipulated.":
      "Bu, medya hakkında değil, kanıt hakkında bir ifadedir. Buradaki hiçbir şey medyanın gerçek ya da manipüle edilmiş olduğunu düşündürmez.",
    "No usable calibrated reading — missing, failed, or produced by a build the thresholds were never measured against.":
      "Kullanılabilir kalibre edilmiş okuma yok — eksik, başarısız ya da eşiklerin hiç ölçülmediği bir sürüm tarafından üretilmiş.",
    "No usable calibrated reading — missing, failed, abstained because no face was found, or produced by a build the thresholds were never measured against.":
      "Kullanılabilir kalibre edilmiş okuma yok — eksik, başarısız, yüz bulunamadığı için çekimser kalınmış ya da eşiklerin hiç ölçülmediği bir sürüm tarafından üretilmiş.",
    "A calibrated detector answered, but its figures could not be read, so no rule could be applied.":
      "Kalibre edilmiş bir dedektör yanıt verdi, ancak değerleri okunamadı; bu nedenle hiçbir kural uygulanamadı.",
    "No usable reading. See this detector's own panel below.":
      "Kullanılabilir okuma yok. Bu dedektörün aşağıdaki kendi paneline bakın.",
    "The level rests on one detector's evidence. The other two are calibrated for face swaps, which is not what this detector reports on, so their silence is not a second opinion — in the R4-T1 calibration study the synthetic-video and face-manipulation detectors never once agreed, and requiring agreement would have detected nothing at all.":
      "Düzey tek bir dedektörün kanıtına dayanır. Diğer ikisi yüz değiştirmeler için kalibre edilmiştir; bu, bu dedektörün raporladığı şey değildir, dolayısıyla onların sessizliği ikinci bir görüş değildir — R4-T1 kalibrasyon çalışmasında sentetik video ve yüz manipülasyonu dedektörleri bir kez bile uyuşmadı ve uyuşma şartı koşmak hiçbir şey tespit etmemekle sonuçlanırdı.",
    "Below its threshold, or without a usable reading, and did not contribute. Under this ruleset a detector that did not reach its own threshold is never allowed to reduce or veto another detector's finding: see this detector's own panel below for what it reported.":
      "Eşiğinin altında ya da kullanılabilir bir okuma olmadan; katkıda bulunmadı. Bu kural setinde, kendi eşiğine ulaşmayan bir dedektörün başka bir dedektörün bulgusunu düşürmesine ya da veto etmesine hiçbir zaman izin verilmez: ne bildirdiği için bu dedektörün aşağıdaki kendi paneline bakın.",
    "The level rests on one detector's evidence. The mouth-dynamics model was calibrated for face swaps too, but it asks a different question — how a mouth moves over 25 consecutive frames, rather than how a face crop looks — and it abstains outright on media where no run holds a trackable face. Its silence therefore does not contradict this finding and was not allowed to reduce the level.":
      "Düzey tek bir dedektörün kanıtına dayanır. Ağız dinamiği modeli de yüz değiştirmeler için kalibre edildi, ancak farklı bir soru sorar — bir yüz kırpıntısının nasıl göründüğünü değil, bir ağzın 25 ardışık kare boyunca nasıl hareket ettiğini — ve hiçbir dizinin izlenebilir bir yüz içermediği medyada doğrudan çekimser kalır. Bu nedenle sessizliği bu bulguyla çelişmez ve düzeyi düşürmesine izin verilmedi.",
    "Scored at or above 0.9868, the threshold measured for it. This classifier judges the appearance of sampled face crops and is calibrated for face swaps.":
      "Onun için ölçülmüş eşik olan 0.9868 veya üzerinde skor aldı. Bu sınıflandırıcı örneklenen yüz kırpıntılarının görünümünü değerlendirir ve yüz değiştirmeler için kalibre edilmiştir.",
    "More than one calibrated detector independently reached its own threshold on this media.":
      "Kalibre edilmiş birden fazla dedektör bu medyada bağımsız olarak kendi eşiğine ulaştı.",
    "Two or more independent findings, reached separately. The rule records that more than one detector reached its threshold and deliberately does not name which — each detector's own panel below shows its score and its threshold. The scores were not combined, averaged, weighted or voted on, and agreement did not raise the level: there is no band above HIGH, and no measurement says two findings mean more than one.":
      "Ayrı ayrı ulaşılmış iki ya da daha fazla bağımsız bulgu. Kural, birden fazla dedektörün eşiğine ulaştığını kaydeder ve bilinçli olarak hangileri olduğunu belirtmez — her dedektörün aşağıdaki kendi paneli skorunu ve eşiğini gösterir. Skorlar birleştirilmedi, ortalanmadı, ağırlıklandırılmadı ya da oylanmadı ve uyuşma düzeyi yükseltmedi: HIGH'ın üzerinde bir bant yoktur ve hiçbir ölçüm iki bulgunun tek bir bulgudan daha fazlasını ifade ettiğini söylemez.",
    "See this detector's own panel below for its score and the threshold it was compared against.":
      "Skoru ve karşılaştırıldığı eşik için bu dedektörün aşağıdaki kendi paneline bakın.",
    "The LipForensics mouth-dynamics model reached its calibrated threshold. That finding alone produced this level.":
      "LipForensics ağız dinamiği modeli kalibre edilmiş eşiğine ulaştı. Bu düzeyi tek başına bu bulgu üretti.",
    "The level rests on one detector's evidence, and on the newest of the three calibrations. Its operating point was measured over 40 clips of one dataset — smaller than the 159-clip study behind the other two — so the bound on its false-positive rate is looser even though it flagged none of the 20 genuine clips in that corpus. The face classifier scoring below its own threshold is not evidence against this finding: it judges appearance and this model judges motion.":
      "Düzey tek bir dedektörün kanıtına ve üç kalibrasyonun en yenisine dayanır. Çalışma noktası tek bir veri kümesinden 40 klip üzerinde ölçüldü — diğer ikisinin arkasındaki 159 klipli çalışmadan daha küçük — bu nedenle o derlemdeki 20 gerçek klibin hiçbirini işaretlememiş olsa da yanlış pozitif oranının sınırı daha gevşektir. Yüz sınıflandırıcısının kendi eşiğinin altında skor alması bu bulguya karşı kanıt değildir: o görünümü değerlendirir, bu model ise hareketi.",
    "Scored at or above 0.2296, the threshold measured for it in R5-T3. That number is a point on this model's own scale and is not comparable with the other two thresholds; it reads how the mouth moves across runs of 25 consecutive frames, not how any single frame looks.":
      "R5-T3'te onun için ölçülmüş eşik olan 0.2296 veya üzerinde skor aldı. Bu sayı, bu modelin kendi ölçeğindeki bir noktadır ve diğer iki eşikle karşılaştırılamaz; tek bir karenin nasıl göründüğünü değil, ağzın 25 ardışık karelik diziler boyunca nasıl hareket ettiğini okur.",
    "All three detectors read this media and none reached its own threshold.":
      "Üç dedektörün tamamı bu medyayı okudu ve hiçbiri kendi eşiğine ulaşmadı.",
    "A generated-video question, a face-appearance question and a mouth-motion question were all asked of this media, and none was answered above its threshold. That is not a finding that the media is genuine: each detector is deliberately set to a point that almost never flags legitimate footage, which means a great deal of manipulated media also falls below it.":
      "Bu medyaya üretilmiş video sorusu, yüz görünümü sorusu ve ağız hareketi sorusunun tamamı soruldu ve hiçbiri eşiğinin üzerinde yanıtlanmadı. Bu, medyanın gerçek olduğuna dair bir bulgu değildir: her dedektör bilinçli olarak meşru görüntüleri neredeyse hiç işaretlemeyen bir noktaya ayarlanmıştır; bu da manipüle edilmiş medyanın büyük bir kısmının da bu noktanın altında kaldığı anlamına gelir.",
    "Produced a usable calibrated reading below 0.2296.":
      "0.2296'nın altında kullanılabilir bir kalibre edilmiş okuma üretti.",
    "Only some of the three calibrated detectors produced a usable reading, and none of those reached its threshold.":
      "Kalibre edilmiş üç dedektörden yalnızca bazıları kullanılabilir bir okuma üretti ve bunların hiçbiri eşiğine ulaşmadı.",
    "This level was reached with less coverage than a three-detector result: at least one of the three questions was never answered for this media. It is a weaker basis than it looks, and it is reported separately for exactly that reason. The panels below say which detectors reported and which did not.":
      "Bu düzeye, üç dedektörlü bir sonuçtan daha az kapsamla ulaşıldı: üç sorudan en az biri bu medya için hiç yanıtlanmadı. Göründüğünden daha zayıf bir dayanaktır ve tam da bu nedenle ayrı raporlanır. Aşağıdaki paneller hangi dedektörlerin bildirdiğini ve hangilerinin bildirmediğini söyler.",
    "No calibrated detector produced a reading that any rule could be applied to.":
      "Kalibre edilmiş hiçbir dedektör herhangi bir kuralın uygulanabileceği bir okuma üretmedi.",
    "No usable calibrated reading — missing, failed, abstained because no run of 25 frames held a trackable face, or produced by a build the threshold was never measured against.":
      "Kullanılabilir kalibre edilmiş okuma yok — eksik, başarısız, hiçbir 25 karelik dizi izlenebilir bir yüz içermediği için çekimser kalınmış ya da eşiğin hiç ölçülmediği bir sürüm tarafından üretilmiş.",
    "The level rests on one detector's evidence. The face classifier is calibrated for face swaps, which is not what this detector reports on, so its silence is not a second opinion — in the R4-T1 calibration study the two never once agreed, and requiring agreement would have detected nothing at all. Two detectors can produce this classification under r7-v4.0.0: the synthetic-video detector and the face-manipulation classifier, each against a threshold measured for it alone. The mouth-dynamics model is reported beside them as independent evidence and cannot change the level.":
      "Düzey tek bir dedektörün kanıtına dayanır. Yüz sınıflandırıcısı yüz değiştirmeler için kalibre edilmiştir; bu, bu dedektörün raporladığı şey değildir, dolayısıyla onun sessizliği ikinci bir görüş değildir — R4-T1 kalibrasyon çalışmasında ikisi bir kez bile uyuşmadı ve uyuşma şartı koşmak hiçbir şey tespit etmemekle sonuçlanırdı. r7-v4.0.0 altında bu sınıflandırmayı iki dedektör üretebilir: sentetik video dedektörü ve yüz manipülasyonu sınıflandırıcısı; her biri yalnızca kendisi için ölçülmüş bir eşiğe göre. Ağız dinamiği modeli bunların yanında bağımsız kanıt olarak raporlanır ve düzeyi değiştiremez.",
    "Recorded as independent forensic evidence and not eligible to move the risk level under this ruleset. R7-T5 measured this detector's operating point against 307 independent genuine lineages and found it produced too many false HIGH results to decide on its own, so R7-T6 removed it from the rules. Its score — including a score above the threshold measured for it in R5-T3 — took no part in this level. See its own panel below for what it reported.":
      "Bağımsız adli kanıt olarak kaydedilir ve bu kural setinde risk düzeyini değiştirmeye uygun değildir. R7-T5 bu dedektörün çalışma noktasını 307 bağımsız gerçek soy hattına karşı ölçtü ve tek başına karar vermek için çok fazla yanlış HIGH sonucu ürettiğini buldu; bu nedenle R7-T6 onu kurallardan çıkardı. Skoru — R5-T3'te onun için ölçülen eşiğin üzerindeki bir skor dahil — bu düzeyde hiçbir rol oynamadı. Ne bildirdiği için aşağıdaki kendi paneline bakın.",
    "The level rests on one detector's evidence. The synthetic-video detector is near-blind to face swaps, so its score below its own threshold is not a second opinion and was not allowed to reduce the level. Two detectors can produce this classification under r7-v4.0.0: the synthetic-video detector and the face-manipulation classifier, each against a threshold measured for it alone. The mouth-dynamics model is reported beside them as independent evidence and cannot change the level.":
      "Düzey tek bir dedektörün kanıtına dayanır. Sentetik video dedektörü yüz değiştirmelere neredeyse kördür; dolayısıyla kendi eşiğinin altındaki skoru ikinci bir görüş değildir ve düzeyi düşürmesine izin verilmedi. r7-v4.0.0 altında bu sınıflandırmayı iki dedektör üretebilir: sentetik video dedektörü ve yüz manipülasyonu sınıflandırıcısı; her biri yalnızca kendisi için ölçülmüş bir eşiğe göre. Ağız dinamiği modeli bunların yanında bağımsız kanıt olarak raporlanır ve düzeyi değiştiremez.",
    "Both of the calibrated detectors this ruleset decides from independently reached their own thresholds on this media.":
      "Bu kural setinin karar verdiği kalibre edilmiş her iki dedektör de bu medyada bağımsız olarak kendi eşiğine ulaştı.",
    "Two independent findings, reached separately and on unrelated scales. The scores were not combined, averaged, weighted or voted on, and agreement did not raise the level: there is no band above HIGH, and no measurement says two findings mean more than one. Two detectors can produce this classification under r7-v4.0.0: the synthetic-video detector and the face-manipulation classifier, each against a threshold measured for it alone. The mouth-dynamics model is reported beside them as independent evidence and cannot change the level.":
      "Ayrı ayrı ve ilişkisiz ölçeklerde ulaşılmış iki bağımsız bulgu. Skorlar birleştirilmedi, ortalanmadı, ağırlıklandırılmadı ya da oylanmadı ve uyuşma düzeyi yükseltmedi: HIGH'ın üzerinde bir bant yoktur ve hiçbir ölçüm iki bulgunun tek bir bulgudan daha fazlasını ifade ettiğini söylemez. r7-v4.0.0 altında bu sınıflandırmayı iki dedektör üretebilir: sentetik video dedektörü ve yüz manipülasyonu sınıflandırıcısı; her biri yalnızca kendisi için ölçülmüş bir eşiğe göre. Ağız dinamiği modeli bunların yanında bağımsız kanıt olarak raporlanır ve düzeyi değiştiremez.",
    "All three calibrated detectors produced a usable reading, and neither detector this ruleset decides from reached its threshold.":
      "Kalibre edilmiş üç dedektörün tamamı kullanılabilir bir okuma üretti ve bu kural setinin karar verdiği dedektörlerin hiçbiri eşiğine ulaşmadı.",
    "A generated-video question, a face-appearance question and a mouth-motion question were all asked of this media, and neither of the two questions that can produce a level was answered above its threshold. That is not a finding that the media is genuine: both deciding detectors are deliberately set to a point that almost never flags legitimate footage, which means a great deal of manipulated media also falls below them. Two detectors can produce this classification under r7-v4.0.0: the synthetic-video detector and the face-manipulation classifier, each against a threshold measured for it alone. The mouth-dynamics model is reported beside them as independent evidence and cannot change the level.":
      "Bu medyaya üretilmiş video sorusu, yüz görünümü sorusu ve ağız hareketi sorusunun tamamı soruldu ve bir düzey üretebilen iki sorudan hiçbiri eşiğinin üzerinde yanıtlanmadı. Bu, medyanın gerçek olduğuna dair bir bulgu değildir: karar veren her iki dedektör de bilinçli olarak meşru görüntüleri neredeyse hiç işaretlemeyen bir noktaya ayarlanmıştır; bu da manipüle edilmiş medyanın büyük bir kısmının da bu noktaların altında kaldığı anlamına gelir. r7-v4.0.0 altında bu sınıflandırmayı iki dedektör üretebilir: sentetik video dedektörü ve yüz manipülasyonu sınıflandırıcısı; her biri yalnızca kendisi için ölçülmüş bir eşiğe göre. Ağız dinamiği modeli bunların yanında bağımsız kanıt olarak raporlanır ve düzeyi değiştiremez.",
    "Only some of the three calibrated detectors produced a usable reading, and neither detector this ruleset decides from reached its threshold.":
      "Kalibre edilmiş üç dedektörden yalnızca bazıları kullanılabilir bir okuma üretti ve bu kural setinin karar verdiği dedektörlerin hiçbiri eşiğine ulaşmadı.",
    "This level was reached with less coverage than a three-detector result: at least one of the three questions was never answered for this media. It is a weaker basis than it looks, and it is reported separately for exactly that reason. The panels below say which detectors reported and which did not. Two detectors can produce this classification under r7-v4.0.0: the synthetic-video detector and the face-manipulation classifier, each against a threshold measured for it alone. The mouth-dynamics model is reported beside them as independent evidence and cannot change the level.":
      "Bu düzeye, üç dedektörlü bir sonuçtan daha az kapsamla ulaşıldı: üç sorudan en az biri bu medya için hiç yanıtlanmadı. Göründüğünden daha zayıf bir dayanaktır ve tam da bu nedenle ayrı raporlanır. Aşağıdaki paneller hangi dedektörlerin bildirdiğini ve hangilerinin bildirmediğini söyler. r7-v4.0.0 altında bu sınıflandırmayı iki dedektör üretebilir: sentetik video dedektörü ve yüz manipülasyonu sınıflandırıcısı; her biri yalnızca kendisi için ölçülmüş bir eşiğe göre. Ağız dinamiği modeli bunların yanında bağımsız kanıt olarak raporlanır ve düzeyi değiştiremez.",
    "No signal from this detector is stored for this analysis.":
      "Bu analiz için bu dedektöre ait kayıtlı bir sinyal yok.",
    "The detector produced no reading — it failed, timed out, or abstained.":
      "Dedektör bir okuma üretmedi — başarısız oldu, zaman aşımına uğradı ya da çekimser kaldı.",
    "The reading came from a deployment this ruleset's threshold was never measured against, so the decision did not read it.":
      "Okuma, bu kural setinin eşiğinin hiç ölçülmediği bir dağıtımdan geldi; bu nedenle karar onu okumadı.",
    "The detector answered, but with figures the persisted decision could not read.":
      "Dedektör yanıt verdi, ancak kaydedilen kararın okuyamadığı değerlerle.",
    "The threshold measured for this detector under this ruleset could not be resolved.":
      "Bu kural setinde bu dedektör için ölçülmüş eşik çözümlenemedi.",
    "Decisive — a reason for this level":
      "Belirleyici — bu düzeyin bir nedeni",
    "Considered — read by this ruleset":
      "Değerlendirildi — bu kural seti tarafından okundu",
    "Evidence only — this ruleset took no decision from it":
      "Yalnızca kanıt — bu kural seti ondan karar almadı",
    "Face-forgery model":
      "Yüz sahteciliği modeli",
    "Active-speaker detection":
      "Aktif konuşmacı tespiti",
    "Audio-authenticity model":
      "Ses özgünlüğü modeli",
    "The analysed artifact was uploaded by the submitter.":
      "Analiz edilen artefakt gönderen tarafından yüklendi.",
    "No Content Credentials were found in the analysed artifact. This does not establish whether credentials were present in any file it was derived from.":
      "Analiz edilen artefaktta Content Credentials bulunamadı. Bu, türetildiği herhangi bir dosyada kimlik bilgilerinin bulunup bulunmadığını belirlemez.",
    "Media was acquired by URL as a single served file.":
      "Medya URL ile, sunulan tek bir dosya olarak edinildi.",
    "Media was acquired by URL and assembled from multiple served media components.":
      "Medya URL ile edinildi ve sunulan birden çok medya bileşeninden birleştirildi.",
    "Media was acquired from {host} as a single served file.":
      "Medya {host} adresinden, sunulan tek bir dosya olarak edinildi.",
    "No Content Credentials were found in the artifact acquired from {host}. This does not establish whether credentials were present in an upstream or publisher-original file.":
      "{host} adresinden edinilen artefaktta Content Credentials bulunamadı. Bu, üst kaynaktaki ya da yayıncının orijinal dosyasında kimlik bilgilerinin bulunup bulunmadığını belirlemez.",
    "Media was acquired from {host} and assembled from multiple served media components.":
      "Medya {host} adresinden edinildi ve sunulan birden çok medya bileşeninden birleştirildi.",
    "How this artifact was acquired was not recorded for this analysis.":
      "Bu artefaktın nasıl edinildiği bu analiz için kaydedilmedi.",
    "How this artifact was acquired was not recorded for this analysis. It was assembled by InspectRoot from separate video and audio streams.":
      "Bu artefaktın nasıl edinildiği bu analiz için kaydedilmedi. Ayrı video ve ses akışlarından InspectRoot tarafından birleştirildi.",
    // The dashboard (R16-T5): the queue row, its evidence drawer and the methodology notes.
    // Same rule as above: keyed by the English the page prints, record values only as slots.
    "How to interpret these results":
      "Bu sonuçlar nasıl yorumlanmalı",
    "RISK —":
      "RİSK —",
    "ruleset {ruleset}":
      "kural seti {ruleset}",
    "Trace":
      "İz",
    "Rule: {rule}":
      "Kural: {rule}",
    "Ruleset: {ruleset}":
      "Kural seti: {ruleset}",
    "Calibration: {calibration}":
      "Kalibrasyon: {calibration}",
    "Analysis {status}: no risk decision has been taken yet.":
      "Analiz {status}: henüz bir risk kararı alınmadı.",
    "Stored risk state {level} is not a supported InspectRoot risk classification (ruleset {ruleset}).":
      "Kayıtlı risk durumu {level}, desteklenen bir InspectRoot risk sınıflandırması değildir (kural seti {ruleset}).",
    "Stored risk state {level} is not a supported InspectRoot risk classification.":
      "Kayıtlı risk durumu {level}, desteklenen bir InspectRoot risk sınıflandırması değildir.",
    "Declared type":
      "Bildirilen tür",
    "Media (ffprobe)":
      "Medya (ffprobe)",
    "Normalized":
      "Normalleştirildi",
    "NVIDIA SVD":
      "NVIDIA SVD",
    "no signal":
      "sinyal yok",
    "Provider version: {version}":
      "Sağlayıcı sürümü: {version}",
    "Synthetic probability":
      "Sentetik olasılık",
    "NVIDIA score: {score}":
      "NVIDIA skoru: {score}",
    "N/A":
      "Yok",
    "Clips":
      "Klipler",
    "Strongest clips (logit)":
      "En güçlü klipler (logit)",
    "NVIDIA clip logit: {logit}":
      "NVIDIA klip logit değeri: {logit}",
    "frame {frame} · {logit}":
      "kare {frame} · {logit}",
    "Active speaker":
      "Aktif konuşmacı",
    "Active speaker: {status}":
      "Aktif konuşmacı: {status}",
    "Unavailable":
      "Kullanılamıyor",
    "No speaking faces detected":
      "Konuşan yüz tespit edilmedi",
    "{shown} of {total} segments":
      "{shown}/{total} bölüm",
    "{shown} segment":
      "{shown} bölüm",
    "{shown} segments":
      "{shown} bölüm",
    "{start}s–{end}s · Face {face} · {speaker}":
      "{start}s–{end}s · Yüz {face} · {speaker}",
    "Audio":
      "Ses",
    "Audio authenticity: {status}":
      "Ses özgünlüğü: {status}",
    "No audio evidence windows":
      "Ses kanıtı penceresi yok",
    "{shown} of {total} audio windows":
      "{shown}/{total} ses penceresi",
    "{shown} audio window":
      "{shown} ses penceresi",
    "{shown} audio windows":
      "{shown} ses penceresi",
    "Checkpoint: {version}":
      "Checkpoint: {version}",
    "{start}s–{end}s · Raw logit[0]: {logit} · Bona fide logit: {bonaFide}":
      "{start}s–{end}s · Ham logit[0]: {logit} · Bona fide logit: {bonaFide}",
    "Face manipulation":
      "Yüz manipülasyonu",
    "Face manipulation: {status}":
      "Yüz manipülasyonu: {status}",
    "{scored} of {requested} sampled frames":
      "{scored}/{requested} örneklenen kare",
    "Mouth dynamics":
      "Ağız dinamiği",
    "Mouth dynamics: {status}":
      "Ağız dinamiği: {status}",
    "{scored} of {requested} sampled runs":
      "{scored}/{requested} örneklenen dizi",
    "Model: {version}":
      "Model: {version}",
    "Provenance (C2PA)":
      "Köken bilgisi (C2PA)",
    "No provenance":
      "Köken bilgisi yok",
    "Remote provenance (not fetched)":
      "Uzak köken bilgisi (getirilmedi)",
    "Extraction failed":
      "Okuma başarısız oldu",
    "C2PA SDK: {version} · Manifest URL (not fetched): {url}":
      "C2PA SDK: {version} · Manifest URL'si (getirilmedi): {url}",
    "C2PA SDK: {version}":
      "C2PA SDK: {version}",
    "Manifest URL (not fetched): {url}":
      "Manifest URL'si (getirilmedi): {url}",
    "{duration}s · constant frame rate · encoded {encoded}":
      "{duration}s · sabit kare hızı · kodlanmış {encoded}",
    "{duration}s · variable frame rate · encoded {encoded}":
      "{duration}s · değişken kare hızı · kodlanmış {encoded}",
    "{duration}s · {pixFmt} · constant frame rate · encoded {encoded}":
      "{duration}s · {pixFmt} · sabit kare hızı · kodlanmış {encoded}",
    "{duration}s · {pixFmt} · variable frame rate · encoded {encoded}":
      "{duration}s · {pixFmt} · değişken kare hızı · kodlanmış {encoded}",
    "Risk":
      "Risk",
    "Risk is a deterministic InspectRoot classification based on calibrated forensic evidence. It is not a Fake/Real determination. Under ruleset {ruleset} two detectors are read for the decision — one calibrated for generated video, one for face swaps — each against its own measured threshold; the scores are never averaged, combined or voted on, and the rule in the trace names which detector reached its threshold. The mouth-dynamics model still runs and is reported on the report, but under this ruleset it is evidence only: it cannot reach the assessment, and a reading it failed to produce removes no decision coverage. {detected} means at least one of those two reached its operating point. {noSignal} means both produced usable readings and neither did — which does not establish that the media is authentic, genuine or source-verified, since a detector reports a score below its threshold for a manipulation family it is blind to as readily as for unmanipulated media. {inconclusive} means a deciding detector produced no usable reading, so the calibrated assessment could not be completed; it is neither evidence of manipulation nor evidence of authenticity. Analyses decided under an earlier ruleset carry that ruleset's vocabulary instead — {high}, {medium} and {unknown}, where {unknown} means the engine ran and could not classify. Every decision is shown with the ruleset that produced it, since the same word means something different under a different one, and neither vocabulary is ever read through the other's. None of this is the same as {pending}, where no decision has been taken yet, or {absent}, where an analysis finished before there was an engine to take one. {unsupported} means the stored state is not one this build classifies under, so it is reported as unsupported rather than shown as a risk class InspectRoot has no calibrated meaning for.":
      "Risk, kalibre edilmiş adli kanıta dayanan deterministik bir InspectRoot sınıflandırmasıdır. Sahte/Gerçek belirlemesi değildir. {ruleset} kural setinde karar için iki dedektör okunur — biri üretilmiş video, biri yüz değiştirmeler için kalibre edilmiştir — her biri kendi ölçülmüş eşiğine göre; skorların hiçbir zaman ortalaması alınmaz, birleştirilmez ya da oylanmaz ve izdeki kural hangi dedektörün eşiğine ulaştığını belirtir. Ağız dinamiği modeli hâlâ çalışır ve raporda bildirilir, ancak bu kural setinde yalnızca kanıttır: değerlendirmeye ulaşamaz ve üretemediği bir okuma karar kapsamından hiçbir şey eksiltmez. {detected}, bu ikisinden en az birinin çalışma noktasına ulaştığı anlamına gelir. {noSignal}, ikisinin de kullanılabilir okuma ürettiği ve hiçbirinin ulaşmadığı anlamına gelir — bu, medyanın özgün, gerçek ya da kaynağı doğrulanmış olduğunu belirlemez; çünkü bir dedektör, kör olduğu bir manipülasyon ailesi için de manipüle edilmemiş medya için olduğu kadar kolayca eşiğinin altında bir skor bildirir. {inconclusive}, karar veren bir dedektörün kullanılabilir bir okuma üretmediği, dolayısıyla kalibre edilmiş değerlendirmenin tamamlanamadığı anlamına gelir; ne manipülasyonun ne de özgünlüğün kanıtıdır. Daha önceki bir kural seti altında karar verilmiş analizler bunun yerine o kural setinin sözcüklerini taşır — {high}, {medium} ve {unknown}; burada {unknown}, motorun çalıştığı ancak sınıflandıramadığı anlamına gelir. Her karar, onu üreten kural setiyle birlikte gösterilir, çünkü aynı sözcük farklı bir kural setinde farklı bir anlama gelir ve iki sözcük dağarcığından hiçbiri diğerininki üzerinden okunmaz. Bunların hiçbiri, henüz bir karar alınmamış olan {pending} ile ya da bir analizin, karar alacak bir motor var olmadan önce tamamlandığı {absent} ile aynı şey değildir. {unsupported}, kayıtlı durumun bu sürümün sınıflandırdığı bir durum olmadığı anlamına gelir; bu nedenle InspectRoot'un kalibre edilmiş bir anlamı olmayan bir risk sınıfı olarak gösterilmek yerine desteklenmeyen olarak raporlanır.",
    "Synthetic probability is NVIDIA's own score for its synthetic-video detector, shown as returned. It is not a verdict.":
      "Sentetik olasılık, NVIDIA'nın sentetik video dedektörü için verdiği kendi skorudur ve döndürüldüğü haliyle gösterilir. Bir karar değildir.",
    "Provenance is what the file itself carries: C2PA Content Credentials, read from the forensic original and shown in C2PA's own words. Most media carries none, so {none} is the ordinary case and not a finding — and an invalid manifest means the credentials do not verify, not that the media is fake. {remote} means the file named a manifest stored somewhere else; that URL was recorded and deliberately never visited, so nothing is known about what it holds.":
      "Köken bilgisi, dosyanın kendisinin taşıdığı bilgidir: adli orijinalden okunan ve C2PA'nın kendi sözcükleriyle gösterilen C2PA Content Credentials. Medyanın çoğu hiçbirini taşımaz; bu nedenle {none} olağan durumdur ve bir bulgu değildir — geçersiz bir manifest de medyanın sahte olduğu değil, kimlik bilgilerinin doğrulanmadığı anlamına gelir. {remote}, dosyanın başka bir yerde saklanan bir manifesti adlandırdığı anlamına gelir; o URL kaydedildi ve bilinçli olarak hiç ziyaret edilmedi, dolayısıyla içinde ne olduğu hakkında hiçbir şey bilinmiyor.",
    "Media is what ffprobe read out of the original before any detector ran, shown as ffprobe reported it. The container is its demuxer family — one name covers MOV and MP4 alike — and it is not narrowed to a container the stored evidence cannot prove. The declared type beside it is only what the client claimed.":
      "Medya, herhangi bir dedektör çalışmadan önce ffprobe'un orijinalden okuduğu bilgilerdir ve ffprobe'un bildirdiği haliyle gösterilir. Kapsayıcı, onun demuxer ailesidir — tek bir ad hem MOV'u hem MP4'ü kapsar — ve kayıtlı kanıtın kanıtlayamadığı bir kapsayıcıya daraltılmaz. Yanındaki bildirilen tür yalnızca istemcinin iddia ettiği şeydir.",
    "Active speaker is when NVIDIA saw a tracked face speaking, in seconds from the start of the analysed video, with the face it tracked and the diarized voice matched to it. It is a record of what was observed, not a finding: {none} means the detector ran and saw nobody speaking, which is the ordinary case for most footage, and {unavailable} means it did not get to look at all. Neither says the video is fake.":
      "Aktif konuşmacı, NVIDIA'nın izlenen bir yüzü konuşurken gördüğü zamandır; analiz edilen videonun başından itibaren saniye cinsinden, izlediği yüz ve ona eşleştirilen ayrıştırılmış (diarize edilmiş) sesle birlikte verilir. Bu bir bulgu değil, gözlemlenenin kaydıdır: {none}, dedektörün çalıştığı ve kimseyi konuşurken görmediği anlamına gelir; bu, görüntülerin çoğu için olağan durumdur. {unavailable} ise hiç bakma fırsatı bulmadığı anlamına gelir. Hiçbiri videonun sahte olduğunu söylemez.",
    "Audio is the two raw logits a local anti-spoofing checkpoint emitted for each window of audio it was given, shown as emitted. The times are the bounds of those windows — InspectRoot cut the audio into fixed 4.04s pieces because that is all the model accepts — and not stretches the model found anything in. The model publishes no threshold and no calibration, so neither figure is a probability, a confidence or a verdict, and consecutive windows of genuine speech routinely disagree. {none} means the reading ran and stored none, which is not proof the file carries no audio, and {unavailable} means it did not get to run.":
      "Ses, yerel bir sahtecilik karşıtı (anti-spoofing) checkpoint'in kendisine verilen her ses penceresi için ürettiği iki ham logit değeridir ve üretildiği haliyle gösterilir. Süreler bu pencerelerin sınırlarıdır — InspectRoot sesi sabit 4.04s'lik parçalara böldü, çünkü model yalnızca bunu kabul eder — ve modelin bir şey bulduğu kesitler değildir. Model hiçbir eşik ve kalibrasyon yayımlamaz; bu nedenle iki değerden hiçbiri bir olasılık, güven değeri ya da karar değildir ve gerçek konuşmanın ardışık pencereleri sık sık birbiriyle uyuşmaz. {none}, okumanın çalıştığı ve hiç pencere kaydetmediği anlamına gelir; bu, dosyanın ses taşımadığının kanıtı değildir. {unavailable} ise okumanın hiç çalışma fırsatı bulmadığı anlamına gelir.",
    "Face manipulation is the score a local EfficientNet-B7 gave the face it found in evenly sampled frames of the video, averaged over those frames and shown as the model produced it. It is not a probability that this media is manipulated. It is calibrated: under {rulesetTwo}, {rulesetThree}, {rulesetFour} and {rulesetFive} it is one of the deciding detectors, compared only against its own threshold measured in R4-T1, and reaching that threshold on its own is enough to set the risk classification — the trace names the rule when it does. Its score is never averaged or combined with another detector's, and a score below its threshold is not a finding that the media is genuine. Only analyses decided under {rulesetOne} read it as evidence alone. {unavailable} means the reading did not produce a score — most often because no face was found, in which case the model was never asked and nothing was established either way.":
      "Yüz manipülasyonu, yerel bir EfficientNet-B7'nin videonun eşit aralıklarla örneklenen karelerinde bulduğu yüze verdiği skordur; bu kareler üzerinden ortalaması alınır ve modelin ürettiği haliyle gösterilir. Bu medyanın manipüle edilmiş olma olasılığı değildir. Kalibre edilmiştir: {rulesetTwo}, {rulesetThree}, {rulesetFour} ve {rulesetFive} kural setlerinde karar veren dedektörlerden biridir; yalnızca R4-T1'de kendisi için ölçülmüş eşikle karşılaştırılır ve bu eşiğe tek başına ulaşması risk sınıflandırmasını belirlemeye yeter — böyle olduğunda iz, kuralı belirtir. Skoru hiçbir zaman başka bir dedektörün skoruyla ortalanmaz ya da birleştirilmez ve eşiğinin altındaki bir skor, medyanın gerçek olduğuna dair bir bulgu değildir. Yalnızca {rulesetOne} altında karar verilmiş analizler onu tek başına kanıt olarak okur. {unavailable}, okumanın bir skor üretmediği anlamına gelir — çoğunlukla hiç yüz bulunamadığı için; bu durumda modele hiç sorulmadı ve iki yönde de hiçbir şey belirlenmedi.",
    "Mouth dynamics is the score a local LipForensics model gave the movement of the mouth across evenly spaced runs of 25 consecutive frames, shown as the model produced it. It is a forgery reading taken from how a mouth moves, and it is emphatically {notLipSync} — the model is never given the audio at all. It is a different question from the face-manipulation score above — movement over time, not the appearance of a face crop — on a different scale, and the two are never compared or combined. It is not a probability that this media is manipulated. An operating point was measured for it in R5-T3, and under {rulesetThree} it was one of the deciding detectors. R7-T6 withdrew it from the rules after R7-T5 measured what that operating point did to genuine media, so under {rulesetFour} and {rulesetFive} it is evidence only: no threshold is applied to it, and it cannot change the risk classification, including when its score stands above that operating point. {unavailable} means the reading did not produce a score — most often because no run held a trackable face throughout, in which case the model was never asked and nothing was established either way.":
      "Ağız dinamiği, yerel bir LipForensics modelinin 25 ardışık karelik, eşit aralıklı diziler boyunca ağzın hareketine verdiği skordur ve modelin ürettiği haliyle gösterilir. Bir ağzın nasıl hareket ettiğinden alınan bir sahtecilik okumasıdır ve özellikle belirtmek gerekir ki bu skor {notLipSync} — modele ses hiç verilmez. Yukarıdaki yüz manipülasyonu skorundan farklı bir sorudur — bir yüz kırpıntısının görünümü değil, zaman içindeki hareket — farklı bir ölçektedir ve ikisi hiçbir zaman karşılaştırılmaz ya da birleştirilmez. Bu medyanın manipüle edilmiş olma olasılığı değildir. R5-T3'te onun için bir çalışma noktası ölçüldü ve {rulesetThree} altında karar veren dedektörlerden biriydi. R7-T5 bu çalışma noktasının gerçek medyaya ne yaptığını ölçtükten sonra R7-T6 onu kurallardan çıkardı; bu nedenle {rulesetFour} ve {rulesetFive} altında yalnızca kanıttır: ona hiçbir eşik uygulanmaz ve skoru o çalışma noktasının üzerinde olduğunda bile risk sınıflandırmasını değiştiremez. {unavailable}, okumanın bir skor üretmediği anlamına gelir — çoğunlukla hiçbir dizi baştan sona izlenebilir bir yüz içermediği için; bu durumda modele hiç sorulmadı ve iki yönde de hiçbir şey belirlenmedi.",
    "Strongest clips are the highest-scoring of the clips NVIDIA examined, identified by frame index because the detector reports no timestamps. The figure is its raw model logit, not a probability and not comparable with the percentage beside it.":
      "En güçlü klipler, NVIDIA'nın incelediği klipler arasında en yüksek skoru alanlardır; dedektör zaman damgası bildirmediği için kare indeksiyle tanımlanır. Değer, ham model logit değeridir; bir olasılık değildir ve yanındaki yüzdeyle karşılaştırılamaz.",
  },
} as const satisfies Partial<Record<Locale, Record<string, string>>>;

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

/**
 * One sentence of the report in the reader's language, and the language it ended up in.
 *
 * English is the key, so it is returned as it came. Any other language returns its entry, or —
 * for a sentence nobody translated — the English, labelled English so the page and its tests can
 * tell. `Object.hasOwn` for the reason `translateCanonical` uses it.
 */
export function translateCopy(locale: Locale, english: string): { text: string; lang: Locale } {
  if (locale === DEFAULT_LOCALE) {
    return { text: english, lang: DEFAULT_LOCALE };
  }

  const table: Record<string, string> | undefined = (
    REPORT_COPY as Partial<Record<Locale, Record<string, string>>>
  )[locale];
  if (table !== undefined && Object.hasOwn(table, english)) {
    return { text: table[english], lang: locale };
  }
  return { text: english, lang: DEFAULT_LOCALE };
}

/**
 * A sentence cut at its `{name}` slots: the words, and between them the names of the values the
 * renderer puts there. Only `{` + letters + `}` is a slot; any other brace is a word.
 */
export function copySegments(text: string): Array<string | { slot: string }> {
  const segments: Array<string | { slot: string }> = [];
  let last = 0;
  for (const match of text.matchAll(/\{([A-Za-z]+)\}/g)) {
    if (match.index > last) {
      segments.push(text.slice(last, match.index));
    }
    segments.push({ slot: match[1] });
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    segments.push(text.slice(last));
  }
  return segments;
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
