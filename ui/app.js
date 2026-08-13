/* AI Content Studio — local UI
   Vanilla JS, no build step. Talks to webapp.py over a tiny JSON API. */

const state = {
  env: { anthropic: false, elevenlabs: false },
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

function linkBtn(href, label) {
  const a = el("a", { href, className: "btn-sm", textContent: label });
  a.style.cssText =
    "display:inline-block;text-decoration:none;border:1px solid var(--border);" +
    "padding:4px 9px;border-radius:7px;color:var(--text);background:var(--bg-inset);font-size:11.5px";
  return a;
}

/* Small Markdown subset: enough for the blog post and LinkedIn output. */
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
      title: state.env.elevenlabs ? "Voice-over will be generated" : "Optional — voice-over step is skipped",
    }),
  );
}

const ASSET_LABELS = {
  blog: "Blog", thread: "Thread", linkedin: "LinkedIn", captions: "Captions",
  carousel: "Carousel", clip: "Clip", voiceover: "Voice", transcript: "Transcript",
};

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
    );
    node.onclick = () => openVideo(item.video_id);
    return node;
  }));
}

/* ----------------------------------------------------------------- views */

function showWelcome() {
  state.view = { kind: "welcome", id: null };
  renderLibrary();

  const formats = [
    ["Blog post", "500–800 words, Markdown"],
    ["X / Twitter thread", "8–12 tweets under 280 chars"],
    ["LinkedIn post", "150–250 words, hook first"],
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
    const cancel = el("button", { className: "btn-sm", textContent: "Cancel" });
    cancel.onclick = async () => {
      cancel.disabled = true;
      try { showJob(await api(`/api/jobs/${job.id}/cancel`, { method: "POST" })); }
      catch (e) { toast(e.message); }
    };
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

  const tabs = [
    { id: "overview", label: "Overview", on: !!d.brief },
    { id: "blog", label: "Blog", on: !!d.blog },
    { id: "thread", label: "Thread", on: !!(d.thread && d.thread.tweets) },
    { id: "linkedin", label: "LinkedIn", on: !!d.linkedin.post },
    { id: "captions", label: "Captions", on: !!(d.captions && d.captions.captions) },
    { id: "carousel", label: "Carousel", on: !!d.carousel.images.length },
    { id: "clip", label: "Clip", on: !!d.clip.video },
    { id: "voiceover", label: "Voice-over", on: !!(d.voiceover.script || d.voiceover.audio) },
    { id: "transcript", label: "Transcript", on: !!d.transcript.word_count },
  ];

  // Tabs without content stay clickable — they explain what's missing —
  // but a freshly opened video lands on the first tab that has something.
  if (!state.tab || !tabs.some((t) => t.id === state.tab)) {
    state.tab = (tabs.find((t) => t.on) || tabs[0]).id;
  }

  const tabBar = el("div", { className: "tabs" }, tabs.map((t) => {
    const b = el("button", { className: "tab" + (t.id === state.tab ? " active" : "") }, t.label);
    if (!t.on) b.append(el("span", { className: "dot", title: "not generated" }));
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

  if (tab === "linkedin") {
    if (!d.linkedin.post) return notGenerated("LinkedIn post");
    const body = el("div", { className: "prose" });
    body.innerHTML = markdown(d.linkedin.post);
    return el("div", {},
      d.linkedin.claims.length ? el("div", { className: "warnbox" },
        el("h4", {}, "⚠ Verify before publishing"),
        el("ul", {}, d.linkedin.claims.map((c) => el("li", {}, c)))) : null,
      card(`linkedin_post.md · ${d.linkedin.post.split(/\s+/).filter(Boolean).length} words`,
        [copyBtn(() => d.linkedin.post)], body),
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

  if (tab === "clip") {
    if (!d.clip.video) return notGenerated("short-form clip");
    const info = d.clip.info || {};
    return el("div", {},
      card("short_form_clip.mp4",
        [linkBtn(media("short_form_clip.mp4") + "?download", "Download")],
        el("video", { className: "clip", src: media("short_form_clip.mp4"), controls: true, preload: "metadata" })),
      info.reason ? card("Why this clip", null,
        el("dl", { className: "kv" },
          el("dt", {}, "Window"),
          el("dd", {}, `${fmtDuration(info.start_seconds)} → ${fmtDuration(info.end_seconds)} (${Math.round(info.duration_seconds || 0)}s)`),
          el("dt", {}, "Reason"), el("dd", {}, info.reason))) : null,
    );
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
    } catch {
      state.pollTimer = setTimeout(tick, 2000);
      return;
    }
    if (state.view.kind === "job" && state.view.id === jobId) showJob(job);
    if (job.status === "running") {
      state.pollTimer = setTimeout(tick, 1200);
    } else {
      $("#run-button").disabled = false;
      $("#run-button").textContent = "Generate";
      await refreshLibrary();
      if (job.status === "done") toast("Done — all formats generated");
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
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    input.value = "";
    btn.textContent = "Running…";
    showJob(job);
    pollJob(job.id);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "Generate";
    toast(err.message);
    if (err.data && err.data.job_id) { showJob(await api(`/api/jobs/${err.data.job_id}`)); pollJob(err.data.job_id); }
  }
});

(async function init() {
  try {
    const data = await api("/api/state");
    state.env = data.env;
    state.library = data.library;
    renderEnv();
    renderLibrary();

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
