const form = document.querySelector("#uploadForm");
const input = document.querySelector("#imageInput");
const fileLabel = document.querySelector("#fileLabel");
const resetButton = document.querySelector("#resetButton");
const statusPanel = document.querySelector("#statusPanel");
const statusText = document.querySelector("#statusText");
const submitButton = document.querySelector(".primary-button");

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
