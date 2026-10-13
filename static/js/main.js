/* =========================================================================
   main.js — VisionLab
   - Drag & drop + file picker handling (client-side validation)
   - Async POST /predict via Fetch API
   - Dynamic gallery rendering from GET /history
   All user/server-supplied strings are inserted with textContent (XSS-safe).
   ========================================================================= */

(() => {
  "use strict";

  // ---------------------------- Constants --------------------------------
  const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"];
  const MAX_BYTES = 8 * 1024 * 1024; // matches Flask's MAX_CONTENT_LENGTH
  const ENDPOINT_PREDICT = "/predict";
  const ENDPOINT_HISTORY = "/history";

  // ---------------------------- Elements ---------------------------------
  const form = document.getElementById("predict-form");
  const dropzone = document.getElementById("dropzone");
  const dropzoneEmpty = document.getElementById("dropzone-empty");
  const fileInput = document.getElementById("file-input");
  const preview = document.getElementById("preview");
  const clearBtn = document.getElementById("clear-btn");
  const analyzeBtn = document.getElementById("analyze-btn");
  const statusEl = document.getElementById("upload-status");

  const resultCard = document.getElementById("result-card");
  const resultImage = document.getElementById("result-image");
  const resultLabel = document.getElementById("result-label");
  const resultMeterFill = document.getElementById("result-meter-fill");
  const resultConfidence = document.getElementById("result-confidence");
  const resultFile = document.getElementById("result-file");

  const gallery = document.getElementById("gallery");
  const galleryEmpty = document.getElementById("gallery-empty");
  const galleryCount = document.getElementById("gallery-count");
  const refreshBtn = document.getElementById("refresh-btn");
  const toast = document.getElementById("toast");

  let selectedFile = null;
  let previewUrl = null;
  let toastTimer = null;

  // ------------------------- Small utilities -----------------------------
  const extOf = (name) => (name.includes(".") ? name.split(".").pop().toLowerCase() : "");

  const isValidType = (file) =>
    ALLOWED_EXTENSIONS.includes(extOf(file.name)) || (file.type || "").startsWith("image/");

  function setStatus(message, kind = "") {
    statusEl.textContent = message;
    statusEl.className = `status ${kind ? `is-${kind}` : ""}`.trim();
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.add("hidden"), 4200);
  }

  function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading || !selectedFile;
    analyzeBtn.querySelector(".spinner").classList.toggle("hidden", !isLoading);
    analyzeBtn.querySelector(".btn-label").textContent = isLoading ? "Analyzing…" : "Analyze image";
  }

  function formatDate(isoWithZ) {
    const date = new Date(isoWithZ);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(date);
  }

  // ------------------------- File selection ------------------------------
  function selectFile(file) {
    if (!file) return;

    if (!isValidType(file)) {
      showToast(`"${file.name}" is not an allowed type (.jpg, .jpeg, .png, .webp).`);
      setStatus("Unsupported file type.", "error");
      return;
    }
    if (file.size > MAX_BYTES) {
      showToast("File is too large — the limit is 8 MB.");
      setStatus("Image exceeds the 8 MB limit.", "error");
      return;
    }

    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);

    preview.src = previewUrl;
    preview.classList.remove("hidden");
    dropzoneEmpty.classList.add("hidden");
    dropzone.classList.add("has-image");
    clearBtn.classList.remove("hidden");
    analyzeBtn.disabled = false;
    resultCard.classList.add("hidden");
    setStatus(`Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
  }

  function clearSelection() {
    selectedFile = null;
    fileInput.value = "";
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    preview.removeAttribute("src");
    preview.classList.add("hidden");
    dropzoneEmpty.classList.remove("hidden");
    dropzone.classList.remove("has-image");
    clearBtn.classList.add("hidden");
    analyzeBtn.disabled = true;
    resultCard.classList.add("hidden");
    setStatus("");
  }

  // --------------------- Drag & drop event wiring ------------------------
  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.add("is-dragover");
    })
  );

  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.remove("is-dragover");
    })
  );

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    selectFile(file);
  });

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
  clearBtn.addEventListener("click", clearSelection);

  // ------------------------- Predict submission --------------------------
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setStatus("Please choose an image first.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile, selectedFile.name);

    setLoading(true);
    setStatus("Running inference…");

    try {
      const response = await fetch(ENDPOINT_PREDICT, { method: "POST", body: formData });
      let payload = null;
      try {
        payload = await response.json();
      } catch {
        payload = null; // non-JSON error body (e.g. proxy HTML page)
      }

      if (!response.ok || !payload || payload.status !== "success") {
        const message = (payload && payload.message) || `Upload failed (HTTP ${response.status}).`;
        setStatus(message, "error");
        showToast(message);
        return;
      }

      renderResult(payload, selectedFile);
      setStatus("Prediction saved to history.", "ok");
      await loadHistory(); // refresh the gallery with the new row
    } catch (error) {
      console.error(error);
      const message = "Network error — is the server running?";
      setStatus(message, "error");
      showToast(message);
    } finally {
      setLoading(false);
    }
  });

  function renderResult(payload, file) {
    resultImage.src = `${payload.image_url}?t=${Date.now()}`; // bust browser cache
    resultLabel.textContent = payload.prediction;

    const percent = Math.round((payload.confidence || 0) * 100);
    resultConfidence.textContent = `${percent}%`;
    // Reset to 0 first so the CSS width transition animates every time.
    resultMeterFill.style.width = "0%";
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        resultMeterFill.style.width = `${percent}%`;
      });
    });

    resultFile.textContent = `Saved as ${payload.image_url} · original: ${file.name}`;
    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ------------------------------ Gallery --------------------------------
  async function loadHistory() {
    try {
      const response = await fetch(ENDPOINT_HISTORY, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      renderGallery(payload.predictions || []);
    } catch (error) {
      console.error("Could not load history:", error);
      showToast("Could not load prediction history.");
    }
  }

  function renderGallery(items) {
    gallery.querySelectorAll(".card").forEach((node) => node.remove());
    galleryCount.textContent = String(items.length);
    galleryEmpty.classList.toggle("hidden", items.length > 0);

    const fragment = document.createDocumentFragment();

    items.forEach((item, index) => {
      const card = document.createElement("article");
      card.className = "card";
      card.style.animationDelay = `${Math.min(index, 12) * 40}ms`;

      // --- media ---
      const media = document.createElement("div");
      media.className = "card-media";

      const img = document.createElement("img");
      img.src = item.image_url;
      img.alt = item.filename || "Predicted image";
      img.loading = "lazy";
      img.addEventListener("error", () => {
        media.style.background = "var(--surface-2)";
        img.remove();
      });

      const rank = document.createElement("span");
      rank.className = "card-rank";
      rank.textContent = `#${index + 1}`;

      media.append(img, rank);

      // --- body ---
      const body = document.createElement("div");
      body.className = "card-body";

      const label = document.createElement("h3");
      label.className = "card-label";
      label.textContent = item.predicted_label;

      const meta = document.createElement("div");
      meta.className = "card-meta";

      const conf = document.createElement("span");
      conf.className = "card-conf";
      const pct =
        item.confidence_score == null ? null : Math.round(item.confidence_score * 100);
      conf.textContent = pct == null ? "n/a" : `${pct}%`;

      const date = document.createElement("span");
      date.textContent = formatDate(item.created_at);
      date.title = item.created_at || "";

      meta.append(conf, date);

      const file = document.createElement("p");
      file.className = "card-file";
      file.textContent = item.filename;
      file.title = item.filename;

      body.append(label, meta, file);
      card.append(media, body);
      fragment.append(card);
    });

    gallery.append(fragment);
  }

  refreshBtn.addEventListener("click", loadHistory);

  // ------------------------------ Boot -----------------------------------
  loadHistory();
})();
