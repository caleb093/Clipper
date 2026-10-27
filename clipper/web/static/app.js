"use strict";

const app = document.getElementById("app");
const STEPS = [
  { key: "download", label: "Get video" },
  { key: "transcribe", label: "Transcribe" },
  { key: "pick", label: "Pick clips" },
  { key: "review", label: "Review" },
  { key: "render", label: "Render" },
];
const OPTIONS_KEY = "clipper.options";
const TRIM_STEP = 0.5; // seconds per click of − / + in the review screen
const MIN_CLIP = 3; // seconds; matches the server's limit
let status = null;
let pollTimer = null;

// ---------------------------------------------------------------- helpers

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function fmtTime(seconds) {
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  const mm = h ? String(m).padStart(2, "0") : m;
  return `${h ? h + ":" : ""}${mm}:${String(r).padStart(2, "0")}`;
}

function timeAgo(epoch) {
  const mins = Math.round((Date.now() / 1000 - epoch) / 60);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? "yesterday" : `${days} days ago`;
}

async function api(path, { method = "GET", body } = {}) {
  const init = { method, headers: {} };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  const type = res.headers.get("content-type") || "";
  const data = type.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = data && data.detail;
    const message = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || data || res.statusText;
    throw new Error(message);
  }
  return data;
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), 2200);
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    toast("Couldn't copy automatically. Select the text and press Ctrl+C.");
    return false;
  }
  if (button) {
    const original = button.textContent;
    button.textContent = "Copied ✓";
    button.classList.add("copied");
    setTimeout(() => {
      button.textContent = original;
      button.classList.remove("copied");
    }, 1600);
  }
  return true;
}

function loadOptions() {
  try {
    return JSON.parse(localStorage.getItem(OPTIONS_KEY)) || {};
  } catch {
    return {};
  }
}

function saveOptions(options) {
  try {
    localStorage.setItem(OPTIONS_KEY, JSON.stringify(options));
  } catch {
    /* private window: options just aren't remembered */
  }
}

// The user's own Anthropic key lives only in their browser and is sent with each job.
const KEY_STORAGE = "clipper.anthropicKey";

function loadKey() {
  try {
    return localStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

function storeKey(key) {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
    return true;
  } catch {
    return false;
  }
}

function apiUsable() {
  return Boolean(status && (status.api.available || loadKey()));
}

function updatePill() {
  const pill = $("#picker-pill");
  if (!status) return;
  pill.hidden = false;
  pill.textContent = apiUsable() ? "Clip picking: Claude API available" : "Clip picking: Claude app (free)";
  pill.title = status.api.error || "";
}

// ----------------------------------------------------------------- router

function route() {
  clearInterval(pollTimer);
  pollTimer = null;
  const [, view, id] = location.hash.split("/");
  window.scrollTo(0, 0);
  if (view === "job" && id) return showJob(id);
  if (view === "project" && id) return showProject(decodeURIComponent(id));
  return showHome();
}

window.addEventListener("hashchange", route);
route();

// ------------------------------------------------------------------- home

async function showHome() {
  app.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    status = await api("/api/status");
  } catch (e) {
    app.innerHTML = `<div class="card notice error-card"><h2>Can't reach Clipper</h2><p>${esc(e.message)}</p></div>`;
    return;
  }
  updatePill();
  const o = { ...status.defaults, ...loadOptions() };
  const job = status.active_job;
  const userKey = loadKey();
  const apiOk = apiUsable();
  const picker = o.picker === "api" && apiOk ? "api" : apiOk && !o.picker ? "api" : "chat";

  app.innerHTML = `
    ${job ? `
      <div class="banner">
        <div><b>A video is in progress:</b> ${esc(job.title)}</div>
        <a class="btn primary small" href="#/job/${job.id}">Continue</a>
      </div>` : ""}
    <section class="card hero">
      <h1>Turn a long video into shorts</h1>
      <p class="muted">Paste a YouTube link or the path of a video on this computer. Clipper transcribes it,
        finds the moments strong enough to post, reframes them to 9:16 and adds captions.</p>
      <div class="type-choice" role="radiogroup" aria-label="What kind of video is this?">
        <label class="type-option">
          <input type="radio" name="content_type" value="explainer" ${o.content_type !== "podcast" ? "checked" : ""}>
          <b>Explainer</b><small>A voice-over with images or B-roll. Frames the interesting part of each shot.</small>
        </label>
        <label class="type-option">
          <input type="radio" name="content_type" value="podcast" ${o.content_type === "podcast" ? "checked" : ""}>
          <b>Podcast</b><small>People talking on camera. Follows whoever is speaking, split screen for back-and-forth.</small>
        </label>
      </div>
      <form id="start-form" class="start-row">
        <input id="source" class="input big" autocomplete="off" spellcheck="false"
          placeholder="https://www.youtube.com/watch?v=…   or   C:\\Videos\\my-video.mp4" aria-label="YouTube link or video file path">
        <button class="btn primary big" type="submit" ${job ? "disabled" : ""}>Make clips</button>
      </form>
      <label id="drop" class="dropzone">
        <input type="file" id="file" accept="video/*" hidden>
        <span id="drop-text"><b>Drop a video file here</b> or click to choose one</span>
        <div class="bar small" id="upload-bar" hidden><div class="fill"></div></div>
      </label>
      <p class="error" id="start-error"></p>

      <details class="options">
        <summary>Options</summary>
        <div class="options-grid">
          <label>Clip length
            <span class="inline-inputs">
              <input type="number" id="min_seconds" min="5" max="180" value="${o.min_seconds}"> to
              <input type="number" id="max_seconds" min="10" max="600" value="${o.max_seconds}"> seconds
            </span>
          </label>
          <label>Most clips
            <input type="number" id="max_clips" min="1" max="50" placeholder="No limit" value="${o.max_clips ?? ""}">
            <small>Empty: every strong moment, however many there are.</small>
          </label>
          <label>Framing
            <select id="layout">
              <option value="auto">Auto: crop to the subject, full frame for text and charts</option>
              <option value="crop">Always crop to 9:16</option>
              <option value="fit">Always full frame on a blurred background</option>
            </select>
          </label>
          <label>Transcription
            <select id="whisper_model">
              <option value="base">Fast (base)</option>
              <option value="small">Balanced (small)</option>
              <option value="medium">Most accurate, slow (medium)</option>
            </select>
          </label>
          <label>Hook text on screen
            <span class="inline-inputs"><input type="number" id="hook_seconds" min="0" max="30" step="0.5" value="${o.hook_seconds}"> seconds</span>
            <small>0 hides it.</small>
          </label>
          <div class="radio-row">
            <label class="check"><input type="checkbox" id="captions" ${o.captions ? "checked" : ""}> Word-by-word captions</label>
            <label class="check"><input type="checkbox" id="uppercase" ${o.uppercase ? "checked" : ""}> Captions in capitals</label>
          </div>
          <div class="radio-row" role="radiogroup" aria-label="Pick clips with">
            <b style="font-size:13px">Pick clips with</b>
            <label><input type="radio" name="picker" value="chat" ${picker === "chat" ? "checked" : ""}>
              <span>Claude app <small>(free with your subscription, copy and paste)</small></span></label>
            <label><input type="radio" name="picker" value="api" ${picker === "api" ? "checked" : ""} ${apiOk ? "" : "disabled"}>
              <span>Claude API <small>(automatic, no copy and paste; ${apiOk ? "about $0.25 per 30-minute video" : "add an API key below to turn this on"})</small></span></label>
          </div>
          <div class="key-box">
            <b>Your Anthropic API key <small>(optional)</small></b>
            ${userKey ? `
              <div class="key-saved">
                <span>Saved in this browser: <code>sk-ant-…${esc(userKey.slice(-4))}</code></span>
                <button type="button" class="linkish" id="remove-key">Remove</button>
              </div>` : `
              <div class="key-form">
                <input type="password" id="api-key" class="input" placeholder="sk-ant-…" autocomplete="off" spellcheck="false" aria-label="Your Anthropic API key">
                <button type="button" class="btn small" id="save-key">Check and save</button>
              </div>`}
            <p class="error" id="key-error">${status.api.error && !userKey ? esc(status.api.error) : ""}</p>
            <small>Kept only in this browser and sent with each video you process; never stored on the server.
              Billed to your Anthropic account (prepaid credits from
              <a href="https://platform.claude.com/settings/keys" target="_blank" rel="noopener">platform.claude.com</a>).
              A Claude Pro subscription doesn't include API access.</small>
          </div>
        </div>
      </details>
    </section>

    <div class="section-title"><h2>Recent</h2></div>
    <div id="recent" class="recent-grid"><p class="empty">Loading…</p></div>`;

  $("#layout").value = o.layout;
  $("#whisper_model").value = ["base", "small", "medium"].includes(o.whisper_model) ? o.whisper_model : "small";

  $("#start-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const source = $("#source").value.trim();
    if (!source) {
      $("#start-error").textContent = "Paste a YouTube link or a video file path first.";
      return;
    }
    startJob(source);
  });

  const drop = $("#drop");
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    if (e.dataTransfer.files[0]) uploadAndStart(e.dataTransfer.files[0]);
  });
  $("#file").addEventListener("change", (e) => e.target.files[0] && uploadAndStart(e.target.files[0]));

  setUpKeyForm();
  loadRecent();
}

function setUpKeyForm() {
  const reopen = async (message) => {
    await showHome();
    $("details.options").open = true;
    toast(message);
  };
  const remove = $("#remove-key");
  if (remove) {
    remove.addEventListener("click", () => {
      storeKey("");
      saveOptions({ ...readOptions(), picker: "chat" });
      reopen("API key removed from this browser.");
    });
    return;
  }
  $("#save-key").addEventListener("click", async (e) => {
    const button = e.currentTarget;
    const key = $("#api-key").value.trim();
    $("#key-error").textContent = "";
    if (!key) {
      $("#key-error").textContent = "Paste your key first.";
      return;
    }
    button.disabled = true;
    button.textContent = "Checking…";
    try {
      await api("/api/check-key", { method: "POST", body: { key } });
      if (!storeKey(key)) throw new Error("This browser won't let Clipper save the key (private window?).");
      saveOptions({ ...readOptions(), picker: "api" });
      reopen("Key saved in this browser. Clips will be picked automatically.");
    } catch (err) {
      $("#key-error").textContent = err.message;
      button.disabled = false;
      button.textContent = "Check and save";
    }
  });
}

function readOptions() {
  const num = (id) => Number($(`#${id}`).value);
  const maxClips = $("#max_clips").value.trim();
  return {
    content_type: $("input[name=content_type]:checked").value,
    min_seconds: num("min_seconds"),
    max_seconds: num("max_seconds"),
    max_clips: maxClips ? Number(maxClips) : null,
    layout: $("#layout").value,
    whisper_model: $("#whisper_model").value,
    hook_seconds: num("hook_seconds"),
    captions: $("#captions").checked,
    uppercase: $("#uppercase").checked,
    picker: $("input[name=picker]:checked").value,
  };
}

async function startJob(source) {
  const options = readOptions();
  saveOptions(options);
  $("#start-error").textContent = "";
  try {
    const apiKey = options.picker === "api" ? loadKey() || null : null;
    const { id } = await api("/api/jobs", { method: "POST", body: { source, options, api_key: apiKey } });
    location.hash = `#/job/${id}`;
  } catch (e) {
    $("#start-error").textContent = e.message;
  }
}

function uploadAndStart(file) {
  const bar = $("#upload-bar"), text = $("#drop-text");
  bar.hidden = false;
  text.innerHTML = `Copying <b>${esc(file.name)}</b> into Clipper…`;
  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $(".fill", bar).style.width = `${(100 * e.loaded) / e.total}%`;
  };
  xhr.onload = () => {
    if (xhr.status === 200) {
      const { path } = JSON.parse(xhr.responseText);
      $("#source").value = path;
      startJob(path);
    } else {
      text.innerHTML = `<b>Upload failed.</b> Try pasting the file's path instead.`;
      bar.hidden = true;
    }
  };
  xhr.onerror = xhr.onload;
  xhr.send(form);
}

async function loadRecent() {
  const box = $("#recent");
  try {
    const projects = await api("/api/projects");
    if (!projects.length) {
      box.innerHTML = `<p class="empty">Your finished clips will show up here.</p>`;
      return;
    }
    box.innerHTML = projects
      .map(
        (p) => `
        <a class="recent-card" href="#/project/${encodeURIComponent(p.slug)}">
          <img src="${p.thumb}" alt="" loading="lazy">
          <div><b>${esc(p.title)}</b><small>${p.clips} clip${p.clips === 1 ? "" : "s"} · ${timeAgo(p.updated)}</small></div>
        </a>`
      )
      .join("");
  } catch {
    box.innerHTML = `<p class="empty">Couldn't load recent projects.</p>`;
  }
}

// -------------------------------------------------------------------- job

async function showJob(id) {
  app.innerHTML = `<p class="muted">Loading…</p>`;
  let shownStatus = null;

  const tick = async () => {
    let job;
    try {
      job = await api(`/api/jobs/${id}`);
    } catch (e) {
      clearInterval(pollTimer);
      app.innerHTML = `
        <div class="card notice error-card"><h2>This job isn't available</h2><p>${esc(e.message)}</p>
        <a class="btn" href="#/">Back to start</a></div>`;
      return;
    }
    if (job.status !== shownStatus) {
      shownStatus = job.status;
      buildJob(job);
    }
    updateJob(job);
    if (job.status === "done" && !job.message && job.slug) {
      clearInterval(pollTimer);
      location.hash = `#/project/${encodeURIComponent(job.slug)}`;
    } else if (["done", "error", "cancelled"].includes(job.status)) {
      clearInterval(pollTimer);
    }
  };
  await tick();
  if (!["done", "error", "cancelled"].includes(shownStatus)) pollTimer = setInterval(tick, 1000);
}

function buildJob(job) {
  app.innerHTML = `
    <a class="back" href="#/">← Start</a>
    <h1 class="job-title">${esc(job.title)}</h1>
    <ol class="stepper">${STEPS.map((s) => `<li data-step="${s.key}"><span class="dot"></span>${s.label}</li>`).join("")}</ol>
    <div id="panel"></div>`;
  const panel = $("#panel");
  if (job.status === "running") panel.innerHTML = progressPanel(job);
  else if (job.status === "waiting_reply") buildChatPanel(job, panel);
  else if (job.status === "review") buildReviewPanel(job, panel);
  else if (job.status === "error") {
    panel.innerHTML = `
      <div class="card notice error-card">
        <h2>Something went wrong</h2>
        <p>${esc(job.error)}</p>
        <a class="btn" href="#/">Start again</a>
      </div>${logDetails(job)}`;
  } else if (job.status === "cancelled") {
    panel.innerHTML = `<div class="card notice"><h2>Cancelled</h2><a class="btn" href="#/">Start again</a></div>`;
  } else if (job.status === "done" && job.message) {
    panel.innerHTML = `<div class="card notice"><h2>No clips this time</h2><p>${esc(job.message)}</p>
      <a class="btn" href="#/">Try another video</a></div>`;
  } else {
    panel.innerHTML = `<div class="card notice"><h2>Done</h2><p>Opening your clips…</p></div>`;
  }
  const cancel = $("#cancel");
  if (cancel) {
    cancel.addEventListener("click", async () => {
      cancel.disabled = true;
      cancel.textContent = "Cancelling…";
      await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }).catch(() => {});
    });
  }
}

function updateJob(job) {
  const current = STEPS.findIndex((s) => s.key === job.stage);
  const finished = job.status === "done";
  $$(".stepper li").forEach((li, i) => {
    li.classList.toggle("done", finished || i < current);
    li.classList.toggle("current", !finished && i === current && job.status !== "error" && job.status !== "cancelled");
    li.classList.toggle("paused", job.status === "waiting_reply" || job.status === "review");
  });
  if (job.status !== "running") return;
  $("#stage-label").textContent = job.stage_label;
  const bar = $("#bar"), pct = $("#pct");
  if (job.progress === null || job.progress === undefined) {
    bar.classList.add("indeterminate");
    pct.textContent = "";
  } else {
    bar.classList.remove("indeterminate");
    $(".fill", bar).style.width = `${(100 * job.progress).toFixed(1)}%`;
    pct.textContent = `${Math.floor(100 * job.progress)}%`;
  }
  $("#detail").textContent = job.detail || "";
  const log = $("#log-text");
  if (log) log.textContent = job.log.join("\n");
}

function progressPanel(job) {
  return `
    <div class="card">
      <div class="progress-head"><h2 id="stage-label">${esc(job.stage_label)}</h2><span class="pct" id="pct"></span></div>
      <div class="bar indeterminate" id="bar"><div class="fill"></div></div>
      <p class="detail" id="detail"></p>
      ${logDetails(job)}
      <div class="row-actions"><button class="btn ghost small" id="cancel">Cancel</button>
        <small>You can leave this page; the work carries on. Come back from the start page.</small></div>
    </div>`;
}

function logDetails(job) {
  return `<details class="log"><summary>Details</summary><pre id="log-text">${esc(job.log.join("\n"))}</pre></details>`;
}

async function buildChatPanel(job, panel) {
  const kb = Math.max(1, Math.round(job.prompt_chars / 1000));
  panel.innerHTML = `
    <div class="card">
      <h2>Pick the clips with Claude</h2>
      <p class="muted">Claude reads the transcript and picks the moments strong enough to post.
        This uses your Claude subscription, so there's no API cost.</p>
      <ol class="chat-steps">
        <li><div class="step-body">
          <div><button class="btn primary" id="copy-prompt" disabled>Copy prompt</button></div>
          <small>About ${kb}k characters: instructions plus the numbered transcript.</small>
        </div></li>
        <li><div class="step-body">
          <div><a class="btn" href="https://claude.ai/new" target="_blank" rel="noopener">Open Claude ↗</a></div>
          <small>Paste the prompt into a new chat (Ctrl+V) and send it. A long prompt may show up as an attachment; that's fine.</small>
        </div></li>
        <li><div class="step-body">
          <span>When Claude has answered, click <b>Copy</b> under its reply and paste it here:</span>
          <textarea id="reply" placeholder="Paste the reply here" spellcheck="false"></textarea>
          <p class="error" id="reply-error"></p>
          <div class="row-actions" style="margin-top:0">
            <button class="btn primary" id="send-reply">Use these clips</button>
            <small>A reply from another AI chat works too, as long as it answered this prompt.</small>
          </div>
        </div></li>
      </ol>
      <div class="row-actions"><button class="btn ghost small" id="cancel">Cancel</button></div>
    </div>`;

  let prompt = "";
  try {
    prompt = await api(`/api/jobs/${job.id}/prompt`);
    $("#copy-prompt").disabled = false;
  } catch (e) {
    $("#reply-error").textContent = `Couldn't load the prompt: ${e.message}`;
  }
  $("#copy-prompt").addEventListener("click", (e) => copyText(prompt, e.currentTarget));
  $("#send-reply").addEventListener("click", async (e) => {
    const button = e.currentTarget;
    button.disabled = true;
    $("#reply-error").textContent = "";
    try {
      await api(`/api/jobs/${job.id}/reply`, { method: "POST", body: { text: $("#reply").value } });
      button.textContent = "Reading the clips…";
    } catch (err) {
      $("#reply-error").textContent = err.message;
      button.disabled = false;
    }
  });
}

function buildReviewPanel(job, panel) {
  const clips = job.clips;
  panel.innerHTML = `
    <div class="card">
      <h2>${clips.length} strong moment${clips.length === 1 ? "" : "s"} found</h2>
      <p class="muted">Play a moment to check it, untick any you don't want, trim or extend it, and tweak the text.
        Nothing is rendered until you click the button.</p>
      <div class="review-list">
        ${clips.map((c, i) => `
          <article class="clip-row" data-index="${i}" data-start="${c.start}" data-end="${c.end}">
            <label class="pick" title="Include this clip"><input type="checkbox" checked aria-label="Include clip ${i + 1}"></label>
            <div class="preview">
              <video src="/api/jobs/${job.id}/source#t=${c.start}" preload="metadata" controls></video>
              <div class="trim" aria-label="Clip timing">
                ${["start", "end"].map((edge) => `
                  <div class="trim-row">
                    <span class="trim-label">${edge === "start" ? "Start" : "End"}</span>
                    <button type="button" class="btn small step" data-edge="${edge}" data-delta="-${TRIM_STEP}"
                      title="${edge === "start" ? "Start earlier" : "End earlier"} (${TRIM_STEP}s)" aria-label="${edge} ${TRIM_STEP} seconds earlier">−</button>
                    <output class="trim-time" data-show="${edge}"></output>
                    <button type="button" class="btn small step" data-edge="${edge}" data-delta="${TRIM_STEP}"
                      title="${edge === "start" ? "Start later" : "End later"} (${TRIM_STEP}s)" aria-label="${edge} ${TRIM_STEP} seconds later">+</button>
                  </div>`).join("")}
                <div class="trim-foot">
                  <button type="button" class="linkish" data-action="ending">▶ Hear the ending</button>
                  <button type="button" class="linkish" data-action="reset" hidden>Undo timing changes</button>
                </div>
              </div>
            </div>
            <div class="clip-fields">
              <div class="clip-meta">
                ${c.score ? `<span class="score">${c.score}</span>` : ""}
                <span data-show="length"></span>
              </div>
              <label>Title <input class="input" data-field="title" value="${esc(c.title)}"></label>
              <div class="two-col">
                <label>On-screen hook <input class="input" data-field="hook_text" value="${esc(c.hook_text)}"></label>
                <label>Hashtags <input class="input" data-field="hashtags" value="${esc(c.hashtags.map((t) => "#" + t).join(" "))}"></label>
              </div>
              <label>Caption <textarea data-field="description" rows="2">${esc(c.description)}</textarea></label>
              <details class="why"><summary>Why this clip</summary>
                ${c.reason ? `<p>${esc(c.reason)}</p>` : ""}<p>“${esc(c.transcript)}”</p>
              </details>
            </div>
          </article>`).join("")}
      </div>
      <div class="sticky-actions">
        <button class="btn primary" id="render"></button>
        <small id="render-note"></small>
        <p class="error" id="render-error"></p>
        <span style="flex:1"></span>
        <button class="btn ghost small" id="cancel">Cancel</button>
      </div>
    </div>`;

  const refresh = () => {
    const n = $$(".clip-row input[type=checkbox]:checked").length;
    $("#render").textContent = n ? `Render ${n} clip${n === 1 ? "" : "s"}` : "Select at least one clip";
    $("#render").disabled = n === 0;
    $("#render-note").textContent = n ? `Roughly ${n} to ${n * 2} minutes on this computer.` : "";
  };
  $$(".clip-row").forEach((row) => {
    const box = $("input[type=checkbox]", row);
    box.addEventListener("change", () => {
      row.classList.toggle("off", !box.checked);
      refresh();
    });
    setUpTrim(row, job.video_duration);
  });
  refresh();

  $("#render").addEventListener("click", async (e) => {
    const button = e.currentTarget;
    button.disabled = true;
    const edits = $$(".clip-row").map((row) => {
      const field = (name) => $(`[data-field=${name}]`, row).value;
      return {
        index: Number(row.dataset.index),
        include: $("input[type=checkbox]", row).checked,
        start: Number(row.dataset.start),
        end: Number(row.dataset.end),
        title: field("title"),
        hook_text: field("hook_text"),
        description: field("description"),
        hashtags: field("hashtags"),
      };
    });
    try {
      await api(`/api/jobs/${job.id}/render`, { method: "POST", body: { clips: edits } });
    } catch (err) {
      $("#render-error").textContent = err.message;
      button.disabled = false;
    }
  });
}

/** Start/End nudges for one clip, with a preview that plays exactly the chosen range. */
function setUpTrim(row, videoDuration) {
  const video = $("video", row);
  const original = { start: Number(row.dataset.start), end: Number(row.dataset.end) };
  const get = (edge) => Number(row.dataset[edge]);
  const limit = videoDuration || Infinity;

  const show = () => {
    const start = get("start"), end = get("end");
    $("[data-show=start]", row).textContent = fmtPrecise(start);
    $("[data-show=end]", row).textContent = fmtPrecise(end);
    $("[data-show=length]", row).textContent = `${fmtTime(start)}–${fmtTime(end)} · ${(end - start).toFixed(1)}s`;
    const changed = start !== original.start || end !== original.end;
    $("[data-action=reset]", row).hidden = !changed;
    row.classList.toggle("retimed", changed);
    $$("[data-edge=start][data-delta^='-']", row).forEach((b) => (b.disabled = start <= 0));
    $$("[data-edge=start]:not([data-delta^='-'])", row).forEach((b) => (b.disabled = end - start - TRIM_STEP < MIN_CLIP));
    $$("[data-edge=end][data-delta^='-']", row).forEach((b) => (b.disabled = end - start - TRIM_STEP < MIN_CLIP));
    $$("[data-edge=end]:not([data-delta^='-'])", row).forEach((b) => (b.disabled = end >= limit));
  };

  $$("[data-edge]", row).forEach((button) =>
    button.addEventListener("click", () => {
      const edge = button.dataset.edge;
      let value = Math.round((get(edge) + Number(button.dataset.delta)) * 100) / 100;
      if (edge === "start") value = Math.min(Math.max(0, value), get("end") - MIN_CLIP);
      else value = Math.max(Math.min(limit, value), get("start") + MIN_CLIP);
      row.dataset[edge] = value;
      video.pause();
      video.currentTime = edge === "start" ? value : Math.max(get("start"), value - 0.05); // show the new edge
      show();
    })
  );

  // Keep playback inside the clip: play starts at Start and stops at End.
  video.addEventListener("play", () => {
    if (video.currentTime < get("start") - 0.1 || video.currentTime >= get("end") - 0.1) video.currentTime = get("start");
  });
  video.addEventListener("timeupdate", () => {
    if (video.currentTime >= get("end")) {
      video.pause();
      video.currentTime = get("end");
    }
  });

  $("[data-action=ending]", row).addEventListener("click", () => {
    video.currentTime = Math.max(get("start"), get("end") - 3);
    video.play();
  });
  $("[data-action=reset]", row).addEventListener("click", () => {
    row.dataset.start = original.start;
    row.dataset.end = original.end;
    video.pause();
    video.currentTime = original.start;
    show();
  });
  show();
}

function fmtPrecise(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

// ---------------------------------------------------------------- project

async function showProject(slug) {
  app.innerHTML = `<p class="muted">Loading…</p>`;
  let project;
  try {
    project = await api(`/api/projects/${encodeURIComponent(slug)}`);
  } catch (e) {
    app.innerHTML = `<div class="card notice error-card"><h2>Project not found</h2><p>${esc(e.message)}</p>
      <a class="btn" href="#/">Back to start</a></div>`;
    return;
  }
  const clips = project.clips;
  app.innerHTML = `
    <a class="back" href="#/">← Start</a>
    <div class="project-head">
      <div>
        <h1 class="job-title">${esc(project.title)}</h1>
        <p class="muted">${clips.length} clip${clips.length === 1 ? "" : "s"} ready to post</p>
      </div>
      <button class="btn" id="open-folder">Open folder</button>
    </div>
    <div class="results">
      ${clips.map((c, i) => `
        <article class="card result">
          <video src="${c.url}" poster="${c.thumb}" controls preload="none" playsinline></video>
          <div class="result-body">
            <div class="clip-meta">
              ${c.score ? `<span class="score">${c.score}</span>` : ""}<span>${Math.round(c.duration)}s</span>
              <a class="dl" href="${c.url}" download="${esc(c.file)}">Download</a>
            </div>
            <h3>${esc(c.title)}</h3>
            <div class="result-actions">
              <button class="btn small" data-copy="caption" data-i="${i}" title="Description and hashtags for TikTok or Reels">Copy caption</button>
              <button class="btn small" data-copy="youtube_title" data-i="${i}" title="Title with #shorts for YouTube">Copy YT title</button>
            </div>
            <details><summary>Post text</summary><pre>${esc(c.caption)}</pre></details>
          </div>
        </article>`).join("")}
    </div>`;

  $$("[data-copy]").forEach((button) =>
    button.addEventListener("click", () => copyText(clips[button.dataset.i][button.dataset.copy], button))
  );
  $("#open-folder").addEventListener("click", () =>
    api(`/api/projects/${encodeURIComponent(slug)}/open`, { method: "POST" }).catch((e) => toast(e.message))
  );
}
