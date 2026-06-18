const form = document.querySelector("#uploadForm");
const input = document.querySelector("#imageInput");
const fileLabel = document.querySelector("#fileLabel");
const resetButton = document.querySelector("#resetButton");
const statusPanel = document.querySelector("#statusPanel");
const statusText = document.querySelector("#statusText");
const submitButton = document.querySelector(".primary-button");
const metricsButton = document.querySelector("#metricsButton");

const imageIds = {
  input: "#inputImage",
  lesion_mask: "#maskImage",
  segmented_lesion: "#segmentedImage",
  gradcam_overlay: "#gradcamImage",
};

const fields = {
  prediction: document.querySelector("#predictionText"),
  confidence: document.querySelector("#confidenceText"),
  area: document.querySelector("#areaText"),
  size: document.querySelector("#sizeText"),
  chip: document.querySelector("#riskChip"),
  filename: document.querySelector("#filenameText"),
  bars: document.querySelector("#probabilityBars"),
};

function setStatus(message, mode = "ready") {
  statusPanel.classList.remove("loading", "error");
  if (mode !== "ready") {
    statusPanel.classList.add(mode);
  }
  statusText.textContent = message;
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function setImage(selector, src) {
  const image = document.querySelector(selector);
  image.src = src || "";
}

function renderBars(predictions) {
  fields.bars.innerHTML = "";
  predictions.forEach((item) => {
    const row = document.createElement("div");
    row.className = `bar-row ${item.class.toLowerCase()}`;

    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = item.class;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = percent(item.probability);
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = percent(item.probability);

    row.append(label, track, value);
    fields.bars.appendChild(row);
  });
}

function renderResult(data) {
  const prediction = data.prediction;
  const metadata = data.metadata;

  Object.entries(imageIds).forEach(([key, selector]) => {
    setImage(selector, data.images[key]);
  });

  fields.prediction.textContent = prediction.predicted_class;
  fields.confidence.textContent = percent(prediction.confidence);
  fields.area.textContent =
    metadata.lesion_area_percent === null ? "--" : `${metadata.lesion_area_percent.toFixed(2)}%`;
  fields.size.textContent = `${metadata.width} x ${metadata.height}`;
  fields.filename.textContent = metadata.filename || "Uploaded image";

  fields.chip.textContent = prediction.predicted_class;
  fields.chip.classList.remove("benign", "malignant");
  fields.chip.classList.add(prediction.predicted_class.toLowerCase());

  renderBars(prediction.top_predictions);
}

function resetView() {
  input.value = "";
  fileLabel.textContent = "Select dermoscopy image";
  Object.values(imageIds).forEach((selector) => setImage(selector, ""));
  fields.prediction.textContent = "--";
  fields.confidence.textContent = "--";
  fields.area.textContent = "--";
  fields.size.textContent = "--";
  fields.filename.textContent = "No file selected";
  fields.bars.innerHTML = "";
  fields.chip.textContent = "Awaiting image";
  fields.chip.classList.remove("benign", "malignant");
  setStatus("Model ready");
}

input.addEventListener("change", () => {
  const file = input.files[0];
  fileLabel.textContent = file ? file.name : "Select dermoscopy image";
});

resetButton.addEventListener("click", resetView);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = input.files[0];
  if (!file) {
    setStatus("Select an image first", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  submitButton.disabled = true;
  setStatus("Running inference", "loading");

  try {
    const response = await fetch("/predict/visual", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Prediction failed.");
    }

    const data = await response.json();
    renderResult(data);
    setStatus("Analysis complete");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
});


// ═══════════════════════════════════════════════════════════════════
//  MODEL METRICS — Confusion Matrix & Classification Report
// ═══════════════════════════════════════════════════════════════════

const metricsSection = document.querySelector("#metricsSection");
let metricsLoaded = false;

metricsButton.addEventListener("click", async () => {
  if (metricsLoaded) {
    // Toggle visibility
    const isHidden = metricsSection.style.display === "none";
    metricsSection.style.display = isHidden ? "" : "none";
    metricsButton.classList.toggle("active", isHidden);
    return;
  }

  metricsButton.disabled = true;
  metricsButton.textContent = "Evaluating…";
  setStatus("Running evaluation on validation set", "loading");

  try {
    const response = await fetch("/evaluate");
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Evaluation failed.");
    }

    const data = await response.json();
    renderMetrics(data);
    metricsSection.style.display = "";
    metricsLoaded = true;
    metricsButton.classList.add("active");
    setStatus("Evaluation complete");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    metricsButton.disabled = false;
    metricsButton.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
      </svg>
      Model Metrics`;
  }
});


function renderMetrics(data) {
  const { class_names, confusion_matrix, metrics, total_samples } = data;

  // ── Sample count badge ─────────────────────────────────────────
  document.querySelector("#metricsSampleCount").textContent = `${total_samples} samples`;

  // ── Macro metric cards (animated counter) ──────────────────────
  animateValue("metAccuracy", metrics.accuracy);
  animateValue("metPrecision", metrics.macro_precision);
  animateValue("metRecall", metrics.macro_recall);
  animateValue("metF1", metrics.macro_f1);

  // ── Confusion Matrix ──────────────────────────────────────────
  renderConfusionMatrix(class_names, confusion_matrix);

  // ── Per-class table ───────────────────────────────────────────
  renderClassTable(class_names, metrics);
}


function animateValue(elementId, target) {
  const el = document.getElementById(elementId);
  const targetPercent = target * 100;
  const duration = 900;
  const start = performance.now();

  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = eased * targetPercent;
    el.textContent = `${current.toFixed(1)}%`;
    if (progress < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}


function renderConfusionMatrix(classNames, cm) {
  const grid = document.querySelector("#cmGrid");
  const colLabels = document.querySelector("#cmColLabels");
  const rowLabels = document.querySelector("#cmRowLabels");
  const n = classNames.length;

  // Find the max value for colour scaling
  let maxVal = 0;
  cm.forEach((row) => row.forEach((v) => { if (v > maxVal) maxVal = v; }));

  // Column labels (predicted)
  colLabels.innerHTML = "";
  classNames.forEach((name) => {
    const lbl = document.createElement("div");
    lbl.className = "cm-label";
    lbl.textContent = name;
    colLabels.appendChild(lbl);
  });

  // Row labels (actual)
  rowLabels.innerHTML = "";
  classNames.forEach((name) => {
    const lbl = document.createElement("div");
    lbl.className = "cm-label";
    lbl.textContent = name;
    rowLabels.appendChild(lbl);
  });

  // Grid cells
  grid.innerHTML = "";
  grid.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
  grid.style.gridTemplateRows = `repeat(${n}, 1fr)`;

  cm.forEach((row, i) => {
    row.forEach((value, j) => {
      const cell = document.createElement("div");
      cell.className = "cm-cell";

      const intensity = maxVal > 0 ? value / maxVal : 0;
      const isDiagonal = i === j;

      if (isDiagonal) {
        // Correct predictions — teal gradient
        const alpha = 0.15 + intensity * 0.75;
        cell.style.background = `rgba(24, 118, 111, ${alpha})`;
        cell.style.color = intensity > 0.55 ? "#fff" : "#17202a";
      } else {
        // Misclassifications — coral gradient
        const alpha = 0.08 + intensity * 0.72;
        cell.style.background = `rgba(201, 95, 75, ${alpha})`;
        cell.style.color = intensity > 0.55 ? "#fff" : "#17202a";
      }

      const numSpan = document.createElement("span");
      numSpan.className = "cm-value";
      numSpan.textContent = value;

      const pctSpan = document.createElement("span");
      pctSpan.className = "cm-pct";
      const total = cm[i].reduce((a, b) => a + b, 0);
      pctSpan.textContent = total > 0 ? `${((value / total) * 100).toFixed(1)}%` : "0%";

      cell.append(numSpan, pctSpan);
      grid.appendChild(cell);
    });
  });
}


function renderClassTable(classNames, metrics) {
  const tbody = document.querySelector("#classTableBody");
  tbody.innerHTML = "";

  classNames.forEach((name) => {
    const tr = document.createElement("tr");

    const tdName = document.createElement("td");
    tdName.innerHTML = `<span class="class-dot ${name.toLowerCase()}"></span>${name}`;

    const tdPrec = document.createElement("td");
    tdPrec.textContent = (metrics.precision[name] * 100).toFixed(1) + "%";

    const tdRec = document.createElement("td");
    tdRec.textContent = (metrics.recall[name] * 100).toFixed(1) + "%";

    const tdF1 = document.createElement("td");
    tdF1.textContent = (metrics.f1_score[name] * 100).toFixed(1) + "%";

    const tdSpec = document.createElement("td");
    tdSpec.textContent = (metrics.specificity[name] * 100).toFixed(1) + "%";

    const tdSupport = document.createElement("td");
    tdSupport.textContent = metrics.support[name];

    tr.append(tdName, tdPrec, tdRec, tdF1, tdSpec, tdSupport);
    tbody.appendChild(tr);
  });
}
