/* AI Content Studio — local UI
   Vanilla JS, no build step. Talks to webapp.py over a tiny JSON API. */

const state = {
  env: { anthropic: false, elevenlabs: false },
  assets: { counts: { video: 0, image: 0, music: 0 }, stock: false },
  kind: "shorts",          // the only pipeline the UI offers
  library: [],
  view: { kind: "welcome", id: null },
  job: null,        // snapshot of the run currently being displayed
  detail: null,     // library detail currently being displayed
  tab: "overview",
  pollTimer: null,
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(kid));
  }
  return node;
};

/* ------------------------------------------------------------------ utils */

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtDuration(sec) {
  sec = Math.round(sec || 0);
  if (!sec) return "";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
           : `${m}:${String(s).padStart(2, "0")}`;
}

function fmtBytes(n) {
  n = n || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${Math.round(n / 1024)} KB`;
  if (n < 1073741824) return `${Math.round(n / 1048576)} MB`;
  return `${(n / 1073741824).toFixed(1)} GB`;
}

function fmtWhen(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
         " " + d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

let toastTimer = null;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 1800);
}

async function copy(text, label = "Copied") {
  try {
    await navigator.clipboard.writeText(text);
    toast(label);
  } catch {
    // Clipboard API needs a secure context; localhost qualifies, but fall
    // back to a hidden textarea just in case.
    const ta = el("textarea", { value: text });
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    toast(label);
  }
}

function copyBtn(getText, label = "Copy") {
  const b = el("button", { className: "btn-sm", textContent: label });
  b.onclick = () => copy(typeof getText === "function" ? getText() : getText);
  return b;
}

/* A filename she can recognise in Photos or Files.
 *
 * Named from the hook rather than the folder, because the hook is what she
 * will remember the clip by. The server derives its own name for plain
 * downloads; this one rides along with the share sheet, which takes the name
 * from the File object instead.
 */
function clipFileName(s) {
  const stem = (s.hook || s.theme || "short")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60) || "short";
  return `${stem}.mp4`;
}

/* Save a clip to the phone (or disk).
 *
 * iOS is the reason this is not just an <a download>. Safari on iPhone
 * frequently ignores the download attribute for video and opens the file in a
 * player instead, which leaves you looking at the clip with no way to keep it.
 * And even when it does save, it goes to Files — not Photos, which is where
 * TikTok and Instagram look when you go to post.
 *
 * The share sheet solves both. navigator.share() with a File opens the native
 * iOS sheet, which offers "Save Video" straight to the camera roll. That is the
 * button she actually wants.
 *
 * Falling back matters as much: desktop browsers mostly cannot share files, so
 * anything without support gets the ordinary download, which is correct there.
 */
function saveBtn(mediaPath, suggestedName) {
  const b = el("button", { className: "btn-sm", textContent: "Save" });
  const url = `/media/${mediaPath}`;

  const plainDownload = () => {
    const a = el("a", { href: `${url}?download` });
    a.setAttribute("download", suggestedName);
    document.body.append(a);
    a.click();
    a.remove();
  };

  b.onclick = async () => {
    // Feature-detect with an actual File: canShare({files}) is the only
    // reliable check, since navigator.share exists on browsers that cannot
    // take files at all.
    const probe = new File([new Blob()], suggestedName, { type: "video/mp4" });
    if (!(navigator.canShare && navigator.canShare({ files: [probe] }))) {
      plainDownload();
      return;
    }

    const original = b.textContent;
    b.disabled = true;
    b.textContent = "Preparing…";
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const file = new File([blob], suggestedName, { type: "video/mp4" });
      await navigator.share({ files: [file] });
      toast("Choose “Save Video” to put it in Photos");
    } catch (err) {
      // AbortError just means she closed the sheet — not a failure, and
      // absolutely not something to show an error for.
      if (err && err.name === "AbortError") return;
      // Anything else (including iOS revoking the user gesture while the
      // clip downloaded) falls back rather than dead-ending.
      plainDownload();
    } finally {
      b.disabled = false;
      b.textContent = original;
    }
  };
  return b;
}

function linkBtn(href, label) {
  const a = el("a", { href, className: "btn-sm", textContent: label });
  a.style.cssText =
    "display:inline-block;text-decoration:none;border:1px solid var(--border);" +
    "padding:4px 9px;border-radius:7px;color:var(--text);background:var(--bg-inset);font-size:11.5px";
  return a;
}

/* Small Markdown subset: enough for the blog post. */
function markdown(src) {
  const lines = esc(src).split("\n");
  const out = [];
  let list = null;   // "ul" | "ol" | null
  let para = [];

  const flushPara = () => {
    if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
  };
  const flushList = () => {
    if (list) { out.push(`</${list}>`); list = null; }
  };
  const inline = (t) =>
    t.replace(/`([^`]+)`/g, "<code>$1</code>")
     .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
     .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
     .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  for (const raw of lines) {
    const line = raw.trimEnd();
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    const quote = /^>\s?(.*)$/.exec(line);

    if (!line.trim()) { flushPara(); flushList(); continue; }

    if (heading) {
      flushPara(); flushList();
      const lvl = Math.min(heading[1].length, 3);
      out.push(`<h${lvl}>${inline(heading[2])}</h${lvl}>`);
    } else if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
      flushPara(); flushList();
      out.push("<hr />");
    } else if (bullet || numbered) {
      flushPara();
      const want = bullet ? "ul" : "ol";
      if (list !== want) { flushList(); out.push(`<${want}>`); list = want; }
      out.push(`<li>${inline((bullet || numbered)[1])}</li>`);
    } else if (quote) {
      flushPara(); flushList();
      out.push(`<blockquote>${inline(quote[1])}</blockquote>`);
    } else {
      flushList();
      para.push(line.trim());
    }
  }
  flushPara(); flushList();
  return out.join("\n");
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON (shouldn't happen) */ }
  if (!res.ok) throw Object.assign(new Error((data && data.error) || res.statusText), { data, status: res.status });
  return data;
}

/* --------------------------------------------------------------- chrome */

function renderEnv() {
  const wrap = $("#env-status");
  wrap.replaceChildren(
    el("span", {
      className: "badge " + (state.env.anthropic ? "ok" : "bad"),
      textContent: state.env.anthropic ? "Anthropic key ✓" : "Anthropic key missing",
      title: state.env.anthropic ? "ANTHROPIC_API_KEY is set" : "Add ANTHROPIC_API_KEY to .env, then restart webapp.py",
    }),
    el("span", {
      className: "badge " + (state.env.elevenlabs ? "ok" : "off"),
      textContent: state.env.elevenlabs ? "ElevenLabs ✓" : "ElevenLabs off",
      title: state.env.elevenlabs
        ? "Voice-over available — off unless you ask for it"
        : "Optional — voice-over step is skipped",
    }),
  );
}

const ASSET_LABELS = {
  motivational: "Shorts", blog: "Blog", thread: "Thread",
  captions: "Captions", carousel: "Carousel", voiceover: "Voice",
  transcript: "Transcript",
};

function renderAssetsNote() {
  const note = $("#assets-note");
  if (!note) return;
  const c = state.assets.counts || {};
  const parts = [`${c.video || 0} b-roll`, `${c.image || 0} images`, `${c.music || 0} tracks`];
  note.textContent = state.assets.stock
    ? `Library: ${parts.join(" · ")} — stock API connected`
    : `Library: ${parts.join(" · ")} — no stock key, using the talk's own footage`;
}

/* Cutting clips is the whole product, so there is no pipeline to choose.
   `state.kind` stays in the payload because the server still routes on it and
   cli.py remains runnable from the command line — the UI just never offers it. */
function setKind(kind) {
  state.kind = kind || "shorts";
  const options = $("#shorts-options");
  if (options) options.hidden = false;
  $("#run-button").textContent = "Cut clips";
}

/* The run chip lives in the header and survives navigation, so a running job
   is still stoppable after you click into the library. */
function renderRunningChip(job) {
  const chip = $("#running-chip");
  if (!chip) return;

  const active = job && job.status === "running";
  chip.hidden = !active;
  if (!active) return;

  const stage = (job.steps || []).find((s) => s.state === "running");
  const label = stage ? stage.label : "Starting…";
  $("#running-label").textContent = `${label} · ${Math.round(job.elapsed)}s`;
  $("#running-label").title = `${job.url}\nClick to view this run`;
  $("#running-label").onclick = () => showJob(state.job || job);

  const stop = $("#stop-button");
  stop.disabled = false;
  stop.textContent = "Stop";
  stop.onclick = () => stopRun(job.id, stop);
}

/* Put the UI back to idle.

   Needed because a job can vanish from under the client: the server holds jobs
   in memory, so restarting webapp.py forgets every one of them while the open
   page carries on polling an id that no longer exists. Without this the chip
   sits on "Running…" forever and Stop can't clear it either, since cancelling
   a job the server has never heard of just 404s. */
function clearRunState(message) {
  clearTimeout(state.pollTimer);
  state.pollMisses = 0;
  state.job = null;
  renderRunningChip(null);
  const run = $("#run-button");
  if (run) run.disabled = false;
  setKind(state.kind);
  if (message) toast(message);
}

async function stopRun(jobId, button) {
  if (button) {
    button.disabled = true;
    button.textContent = "Stopping…";
  }
  try {
    const job = await api(`/api/jobs/${jobId}/cancel`, { method: "POST" });
    state.job = job;
    if (state.view.kind === "job" && state.view.id === jobId) showJob(job);
    renderRunningChip(job);
    toast("Run stopped");
  } catch (e) {
    // Nothing to cancel means the run is already over — which is what the
    // button was asking for, so treat it as success and clear the chip
    // instead of leaving it stuck with an error toast.
    if (e && e.status === 404) {
      clearRunState("That run had already finished");
      return;
    }
    toast(e.message);
    if (button) {
      button.disabled = false;
      button.textContent = "Stop";
    }
  }
}

function renderLibrary() {
  const wrap = $("#library");
  $("#lib-count").textContent = state.library.length;

  if (!state.library.length) {
    wrap.replaceChildren(el("div", { className: "empty" },
      "Nothing generated yet. Paste a YouTube URL above to make your first set."));
    return;
  }

  wrap.replaceChildren(...state.library.map((item) => {
    const node = el("div", {
      className: "lib-item" + (state.view.kind === "video" && state.view.id === item.video_id ? " active" : ""),
    },
      el("div", { className: "t", textContent: item.title || item.video_id }),
      el("div", { className: "m", textContent:
        [item.channel, fmtDuration(item.duration_seconds), fmtWhen(item.modified)].filter(Boolean).join(" · ") }),
      el("div", { className: "chips" },
        Object.entries(ASSET_LABELS)
          .filter(([k]) => item.assets[k])
          .map(([, label]) => el("span", { className: "chip on", textContent: label })),
      ),
      buildRowActions(item),
    );
    node.onclick = () => openVideo(item.video_id);
    return node;
  }));
}

/* The editing brief, folded away until wanted.

   It is long by design — where the clip came from, what is said, what must not
   be trimmed — and printing all of that under every clip would bury the clips
   themselves. Collapsed, it is one line; open, it is the whole brief. */
function briefBlock(short) {
  const wrap = el("details", { className: "brief" },
    el("summary", {}, "Editing brief"),
    el("pre", { className: "text-block", style: "white-space:pre-wrap;margin-top:8px" }, short.brief),
  );
  return wrap;
}


/* Disk actions for one library row.

   Two separate buttons because they aren't the same decision: freeing space
   strips the video and keeps everything Claude was paid to write, while
   deleting removes the talk outright. Both confirm first — this project isn't
   under version control, so neither is recoverable. */
/* Disk actions for one library row.

   Confirmation is done inline on the button rather than with window.confirm().
   The native dialog is suppressed in embedded and automated browsers — it
   returns false without ever showing, so a guarded delete silently did nothing
   and looked like a broken button. A two-step button works everywhere, and it
   also keeps a destructive action from being one stray click away. */
function buildRowActions(item) {
  const row = el("div", { className: "lib-actions" },
    el("span", { className: "size", textContent: fmtBytes(item.size_bytes) }),
  );

  const wipe = el("button", {
    className: "btn-sm danger",
    textContent: "Delete",
    title: "Remove this talk and everything generated from it",
  });

  let armed = null;
  const disarm = () => {
    clearTimeout(armed);
    armed = null;
    wipe.textContent = "Delete";
    wipe.classList.remove("armed");
  };

  wipe.onclick = (e) => {
    e.stopPropagation();
    if (!armed) {
      wipe.textContent = `Delete ${fmtBytes(item.size_bytes)}?`;
      wipe.classList.add("armed");
      // Re-arming shouldn't linger: an armed button left on screen invites a
      // later click that the user has forgotten the meaning of.
      armed = setTimeout(disarm, 4000);
      return;
    }
    disarm();
    removeLibraryItem(item.video_id, wipe);
  };

  row.append(wipe);
  return row;
}

async function removeLibraryItem(videoId, button) {
  button.disabled = true;
  button.textContent = "…";
  try {
    const res = await api(`/api/library/${encodeURIComponent(videoId)}`, { method: "DELETE" });
    toast(`Deleted — freed ${fmtBytes(res.freed_bytes)}`);

    // A deleted talk can't stay open in the main pane.
    if (state.view.kind === "video" && state.view.id === videoId) showWelcome();

    const data = await api("/api/library");
    state.library = data.library || [];
    renderLibrary();
  } catch (err) {
    toast(err.message);
    button.disabled = false;
    button.textContent = "Delete";
  }
}

/* ----------------------------------------------------------------- views */

function showWelcome() {
  state.view = { kind: "welcome", id: null };
  renderLibrary();

  const formats = [
    ["Blog post", "500–800 words, Markdown"],
    ["X / Twitter thread", "8–12 tweets under 280 chars"],
    ["Social captions", "3 variants + hashtags"],
    ["Carousel", "Rendered PNG slides"],
    ["Short-form clip", "Vertical 9:16 MP4"],
    ["Voice-over", "Script + MP3 (optional)"],
    ["Content brief", "The shared source of truth"],
  ];

  $("#content").replaceChildren(el("div", { className: "content-inner welcome" },
    el("h1", {}, "One video in, eight formats out."),
    el("p", {}, "Paste a YouTube URL in the bar above. The pipeline pulls the transcript, " +
                "distills it into a content brief, then writes every format from that brief " +
                "so nothing contradicts anything else."),
    !state.env.anthropic && el("div", { className: "errbox" },
      "ANTHROPIC_API_KEY isn't set. Copy .env.example to .env, add your key, " +
      "then restart webapp.py — runs will fail without it."),
    el("div", { className: "formats" }, formats.map(([name, sub]) =>
      el("div", { className: "format" },
        el("b", {}, name), el("span", {}, sub)))),
  ));
}

function stepIcon(stateName) {
  if (stateName === "done") return el("span", { textContent: "✓" });
  if (stateName === "warn") return el("span", { textContent: "⚠" });
  if (stateName === "error") return el("span", { textContent: "✗" });
  if (stateName === "skipped") return el("span", { textContent: "–" });
  if (stateName === "running") return el("span", { className: "spinner" });
  return el("span", { textContent: "○", style: "color:var(--text-faint)" });
}

function showJob(job) {
  state.view = { kind: "job", id: job.id };
  state.job = job;
  renderLibrary();

  const running = job.status === "running";
  const statusText = {
    running: "Running…", done: "Finished", error: "Failed", cancelled: "Cancelled",
  }[job.status] || job.status;

  const header = el("div", { className: "status-line" },
    el("strong", { style: "color:var(--text)" }, statusText),
    el("span", {}, `${Math.round(job.elapsed)}s elapsed`),
    el("span", { style: "font-family:var(--mono);font-size:12px" }, job.url),
  );

  if (running) {
    const cancel = el("button", { className: "btn-sm", textContent: "Stop run" });
    cancel.onclick = () => stopRun(job.id, cancel);
    header.append(cancel);
  }
  if (job.status === "done" && job.video_id) {
    const open = el("button", { className: "btn-sm", textContent: "Open results →" });
    open.onclick = () => openVideo(job.video_id);
    header.append(open);
  }

  const steps = el("div", { className: "card" },
    el("div", { className: "card-head" }, el("h3", {}, "Pipeline")),
    el("div", { className: "steps" }, job.steps.map((s) =>
      el("div", { className: "step " + s.state },
        el("div", { className: "icon" }, stepIcon(s.state)),
        el("div", {},
          el("div", { className: "label" }, s.label,
            // The clip stage downloads and re-encodes video with long silent
            // stretches — show a running clock so it doesn't read as frozen.
            s.state === "running" && running
              ? el("span", { style: "color:var(--text-faint);font-weight:400" },
                  ` · ${Math.round(job.step_elapsed)}s`)
              : null),
          s.detail && el("div", { className: "detail" }, s.detail)),
      ))),
  );

  const log = el("div", { className: "log", id: "log-box" }, job.log.join("\n"));

  $("#content").replaceChildren(el("div", { className: "content-inner" },
    el("h1", { className: "title" }, "Generating"),
    header,
    job.error && el("div", { className: "errbox" }, job.error),
    steps,
    el("div", { className: "card" },
      el("div", { className: "card-head" },
        el("h3", {}, "Log"),
        el("div", { className: "card-actions" }, copyBtn(() => job.log.join("\n")))),
      log),
  ));

  log.scrollTop = log.scrollHeight;
}

async function openVideo(videoId, tab) {
  state.view = { kind: "video", id: videoId };
  // null => pick the first tab that actually has content once we know what
  // this folder holds. An explicit tab is always honoured.
  state.tab = tab || null;
  renderLibrary();
  $("#content").replaceChildren(el("div", { className: "content-inner empty" }, "Loading…"));
  try {
    state.detail = await api(`/api/library/${encodeURIComponent(videoId)}`);
  } catch (e) {
    $("#content").replaceChildren(el("div", { className: "content-inner errbox" }, e.message));
    return;
  }
  renderVideo();
}

function renderVideo() {
  const d = state.detail;
  if (!d) return;

  const motivational = d.motivational || { shorts: [], carousel: [] };
  const tabs = [
    { id: "shorts", label: `Shorts${motivational.shorts.length ? ` (${motivational.shorts.length})` : ""}`,
      on: !!motivational.shorts.length },
    { id: "overview", label: "Overview", on: !!d.brief },
    { id: "blog", label: "Blog", on: !!d.blog },
    { id: "thread", label: "Thread", on: !!(d.thread && d.thread.tweets) },
    { id: "captions", label: "Captions", on: !!(d.captions && d.captions.captions) },
    { id: "carousel", label: "Carousel",
      on: !!(motivational.carousel || d.carousel.images.length) },
    { id: "voiceover", label: "Voice-over", on: !!(d.voiceover.script || d.voiceover.audio) },
    { id: "transcript", label: "Transcript", on: !!d.transcript.word_count },
  ];

  // Only show tabs that actually have something behind them.
  //
  // Blog, Thread, Captions and Voice-over come from cli.py, which the UI no
  // longer offers, so on a clips-only run they could never fill — an empty tab
  // reads as a broken feature rather than an unused one. They still appear for
  // older videos that do have that content, so nothing already generated
  // becomes unreachable.
  const shown = tabs.filter((t) => t.on);
  const visible = shown.length ? shown : [tabs[0]];

  if (!state.tab || !visible.some((t) => t.id === state.tab)) {
    state.tab = visible[0].id;
  }

  const tabBar = el("div", { className: "tabs" }, visible.map((t) => {
    const b = el("button", { className: "tab" + (t.id === state.tab ? " active" : "") }, t.label);
    b.onclick = () => { state.tab = t.id; renderVideo(); };
    return b;
  }));

  $("#content").replaceChildren(el("div", { className: "content-inner" },
    el("h1", { className: "title" }, d.title || d.video_id),
    el("div", { className: "meta" },
      d.channel && el("span", {}, d.channel),
      d.duration_seconds ? el("span", {}, fmtDuration(d.duration_seconds)) : null,
      el("a", { href: d.url, target: "_blank", rel: "noopener" }, "watch on YouTube ↗"),
      el("span", { style: "font-family:var(--mono);font-size:11.5px;color:var(--text-faint)" }, d.video_id),
    ),
    tabBar,
    renderTab(d, state.tab),
  ));
}

function card(title, actions, ...body) {
  return el("div", { className: "card" },
    el("div", { className: "card-head" },
      el("h3", {}, title),
      el("div", { className: "card-actions" }, actions || [])),
    ...body);
}

function renderTab(d, tab) {
  const media = (name) => `/media/${encodeURIComponent(d.video_id)}/${name}`;

  if (tab === "shorts") {
    const m = d.motivational || { shorts: [], carousel: [] };
    if (!m.shorts.length) {
      return notGenerated("short clips — run this video through \"Short clips\"");
    }
    return el("div", {},
      card(`${m.shorts.length} short${m.shorts.length > 1 ? "s" : ""}`,
        [copyBtn(() => m.shorts.map((s) =>
          `${s.hook}\n"${s.quote}"\n${s.theme} · ${s.duration_seconds}s`).join("\n\n"), "Copy hooks")],
        el("div", { className: "shorts-grid" }, m.shorts.map((s) =>
          el("div", { className: "short-card" },
            el("video", { src: `/media/${s.media}`, controls: true, preload: "metadata", playsInline: true }),
            el("div", { className: "body" },
              el("div", { className: "hook" }, s.hook || "—"),
              s.quote && el("div", { className: "quote" }, `“${s.quote}”`),
              el("div", { className: "row" },
                s.theme && el("span", { className: "theme-tag" }, s.theme),
                el("span", { className: "chip" },
                  `${Math.round(s.duration_seconds || 0)}s`),
                s.start_seconds != null && el("span", {
                  className: "chip",
                  title: "where this moment starts in the original talk",
                }, `from ${fmtDuration(s.start_seconds)}`),
                s.style && el("span", { className: "chip" }, s.style),
                copyBtn(() => s.quote || s.hook, "Copy"),
                saveBtn(s.media, clipFileName(s))),
              s.reason && el("div", { className: "detail", style: "margin-top:8px;color:var(--text-faint);font-size:11.5px;line-height:1.5" }, s.reason),
              s.brief ? briefBlock(s) : null,
            ))))),
      m.carousel ? card("Carousel copy", [copyBtn(() => m.carousel, "Copy all")],
        el("pre", { className: "text-block", style: "white-space:pre-wrap" }, m.carousel)) : null,
    );
  }

  if (tab === "overview") {
    const b = d.brief;
    if (!b) return notGenerated("content brief");
    return el("div", {},
      card("Summary", [copyBtn(() => b.summary)], el("div", { className: "text-block" }, b.summary)),
      card("At a glance", null, el("dl", { className: "kv" },
        el("dt", {}, "Audience"), el("dd", {}, b.target_audience || "—"),
        el("dt", {}, "Tone"), el("dd", {}, b.tone || "—"),
        el("dt", {}, "Call to action"), el("dd", {}, b.call_to_action || "—"),
        el("dt", {}, "Topics"),
        el("dd", {}, el("div", { className: "tag-row" },
          (b.topics || []).map((t) => el("span", { className: "tag" }, t)))),
      )),
      card("Title suggestions", null, el("div", {},
        (b.title_suggestions || []).map((t) =>
          el("div", { style: "display:flex;gap:8px;align-items:baseline;margin-bottom:6px" },
            el("span", { style: "flex:1" }, t), copyBtn(t))))),
      card("Hooks", null, el("div", {},
        (b.hooks || []).map((h) =>
          el("div", { style: "display:flex;gap:8px;align-items:baseline;margin-bottom:6px" },
            el("span", { style: "flex:1" }, h), copyBtn(h))))),
      card(`Key points (${(b.key_points || []).length})`, null,
        el("div", {}, (b.key_points || []).map((p) =>
          el("div", { className: "point" },
            el("div", { className: "p" }, p.point),
            p.supporting_quote && el("div", { className: "q" }, `“${p.supporting_quote}”`))))),
      (b.notable_quotes || []).length ? card("Notable quotes", null,
        el("div", {}, b.notable_quotes.map((q) =>
          el("div", { className: "point" }, el("div", { className: "q" }, `“${q}”`))))) : null,
    );
  }

  if (tab === "blog") {
    if (!d.blog) return notGenerated("blog post");
    const body = el("div", { className: "prose" });
    body.innerHTML = markdown(d.blog);
    return card("blog_post.md",
      [copyBtn(() => d.blog, "Copy Markdown"), linkBtn(media("blog_post.md") + "?download", "Download")],
      body);
  }

  if (tab === "thread") {
    const tweets = (d.thread && d.thread.tweets) || [];
    if (!tweets.length) return notGenerated("thread");
    return el("div", {},
      card(`${tweets.length} tweets`,
        [copyBtn(() => tweets.map((t, i) => `${i + 1}/${tweets.length}\n${t}`).join("\n\n"), "Copy all"),
         linkBtn(media("twitter_thread.txt") + "?download", "Download .txt")],
        el("div", {}, tweets.map((t, i) =>
          el("div", { className: "tweet" },
            el("div", { className: "tweet-head" },
              el("span", {}, `${i + 1} / ${tweets.length}`),
              el("span", {},
                el("span", { className: t.length > 280 ? "count-over" : "" }, `${t.length}/280`),
                " ",
                copyBtn(t))),
            el("div", { className: "tweet-body" }, t))))),
    );
  }

  if (tab === "captions") {
    const caps = (d.captions && d.captions.captions) || [];
    if (!caps.length) return notGenerated("captions");
    return el("div", {}, caps.map((c) => {
      const tags = (c.hashtags || []).map((h) => "#" + String(h).replace(/^#/, "")).join(" ");
      return card(c.style || "caption",
        [copyBtn(() => `${c.text}\n\n${tags}`, "Copy with tags"), copyBtn(() => c.text, "Copy text")],
        el("div", { className: "text-block" }, c.text),
        tags && el("div", { className: "tag-row", style: "margin-top:10px" },
          (c.hashtags || []).map((h) => el("span", { className: "tag" }, "#" + String(h).replace(/^#/, "")))));
    }));
  }

  if (tab === "carousel") {
    // Words, not pictures: the cards get designed by hand afterwards, so what
    // is useful here is copy to paste rather than an image that gets replaced.
    const mc = (d.motivational || {}).carousel;
    if (mc) {
      return card("Carousel copy", [copyBtn(() => mc, "Copy all")],
        el("pre", { className: "text-block", style: "white-space:pre-wrap" }, mc));
    }
    if (!d.carousel.images.length) return notGenerated("carousel");
    const slides = d.carousel.slides || [];
    return card(`${d.carousel.images.length} slides`,
      [copyBtn(() => slides.map((s, i) =>
        `Slide ${i + 1}: ${s.headline}${s.subtext ? "\n" + s.subtext : ""}`).join("\n\n"), "Copy text")],
      el("div", { className: "slides" }, d.carousel.images.map((name, i) => {
        const a = el("a", { href: `/media/${encodeURIComponent(d.video_id)}/carousel/${name}`, target: "_blank", rel: "noopener" },
          el("img", { src: `/media/${encodeURIComponent(d.video_id)}/carousel/${name}`, alt: `Slide ${i + 1}`, loading: "lazy" }));
        a.style.display = "block";
        return el("div", { className: "slide" }, a,
          el("div", { className: "cap" }, slides[i] ? slides[i].headline : name));
      })));
  }

  if (tab === "voiceover") {
    if (!d.voiceover.script && !d.voiceover.audio) return notGenerated("voice-over");
    return el("div", {},
      d.voiceover.audio ? card("voiceover.mp3",
        [linkBtn(media("voiceover.mp3") + "?download", "Download")],
        el("audio", { src: media("voiceover.mp3"), controls: true })) : null,
      d.voiceover.script ? card("Script",
        [copyBtn(() => d.voiceover.script)],
        el("div", { className: "text-block" }, d.voiceover.script)) : null,
    );
  }

  if (tab === "transcript") {
    const t = d.transcript;
    if (!t.word_count) return notGenerated("transcript");
    return card(`${t.word_count.toLocaleString()} words · ${t.language || "?"} · ${t.segment_count} segments`,
      [copyBtn(() => t.text)],
      el("div", { className: "text-block", style: "max-height:60vh;overflow:auto;color:var(--text-dim)" }, t.text));
  }

  return el("div", { className: "empty" }, "Nothing here.");
}

function notGenerated(what) {
  return el("div", { className: "card" },
    el("div", { className: "empty", style: "padding:4px" },
      `No ${what} in this folder — it either wasn't generated or that step failed. ` +
      `Re-run the video to try again.`));
}

/* ------------------------------------------------------------- job polling */

function pollJob(jobId) {
  clearTimeout(state.pollTimer);
  const tick = async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch (err) {
      // 404 is final: the server has no such job and never will again, so
      // stop rather than retrying an id that cannot come back.
      if (err && err.status === 404) {
        clearRunState("That run is no longer on the server");
        return;
      }
      // Anything else may be the server briefly restarting, so retry — but
      // give up eventually instead of polling a dead endpoint all day.
      state.pollMisses = (state.pollMisses || 0) + 1;
      if (state.pollMisses > 15) {
        clearRunState("Lost contact with the run");
        return;
      }
      state.pollTimer = setTimeout(tick, 2000);
      return;
    }
    state.pollMisses = 0;
    state.job = job;
    renderRunningChip(job);
    if (state.view.kind === "job" && state.view.id === jobId) showJob(job);
    if (job.status === "running") {
      state.pollTimer = setTimeout(tick, 1200);
    } else {
      $("#run-button").disabled = false;
      setKind(state.kind);
      renderRunningChip(job);
      await refreshLibrary();
      try {
        state.assets = await api("/api/assets");
        renderAssetsNote();
      } catch { /* leave the previous counts up */ }
      if (job.status === "done") {
        toast(job.kind === "shorts" ? "Shorts ready" : "Done — all formats generated");
        if (job.video_id) openVideo(job.video_id, job.kind === "shorts" ? "shorts" : undefined);
      }
    }
  };
  tick();
}

async function refreshLibrary() {
  try {
    const data = await api("/api/library");
    state.library = data.library;
    renderLibrary();
  } catch { /* server restarting, ignore */ }
}

/* -------------------------------------------------------------- bootstrap */

$("#run-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#url-input");
  const url = input.value.trim();
  if (!url) return;

  const btn = $("#run-button");
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const payload = { url, kind: state.kind };
    if (state.kind === "shorts") {
      // "auto" is sent as absence — the pipeline then takes its count from
      // however many replay peaks the talk actually has.
      const chosen = $("#opt-count").value;
      if (chosen !== "auto") payload.count = Number(chosen);
      payload.style = $("#opt-style").value;
      payload.carousel = $("#opt-carousel").checked;
    }
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    input.value = "";
    btn.textContent = "Running…";
    showJob(job);
    pollJob(job.id);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = label;
    toast(err.message);
    if (err.data && err.data.job_id) { showJob(await api(`/api/jobs/${err.data.job_id}`)); pollJob(err.data.job_id); }
  }
});

(async function init() {
  try {
    const data = await api("/api/state");
    state.env = data.env;
    state.assets = data.assets || state.assets;
    state.library = data.library;
    renderEnv();
    renderAssetsNote();
    renderLibrary();
    setKind(state.kind);


    const running = data.jobs.find((j) => j.status === "running");
    if (running) {
      $("#run-button").disabled = true;
      $("#run-button").textContent = "Running…";
      showJob(await api(`/api/jobs/${running.id}`));
      pollJob(running.id);
    } else if (state.library.length) {
      openVideo(state.library[0].video_id);
    } else {
      showWelcome();
    }
  } catch (e) {
    $("#content").replaceChildren(el("div", { className: "content-inner errbox" },
      "Couldn't reach the local server: " + e.message));
  }
})();
