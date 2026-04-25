(function () {
  const UFCEClient = window.UFCEClient || {};
  const pageData = UFCEClient.readJsonScript && UFCEClient.readJsonScript("home-page-data");
  if (!pageData) {
    return;
  }

  const datasets = Array.isArray(pageData.datasets) ? pageData.datasets : [];
  const defaultDatasetKey = pageData.defaultDatasetKey || "";
  const apiPrefix = pageData.apiPrefix || "/api/v1";
  const escapeHtml = UFCEClient.escapeHtml || ((value) => String(value));

  const datasetLookup = Object.fromEntries(datasets.map((dataset) => [dataset.dataset_key, dataset]));

  async function createSession(datasetKey) {
    const normalized = datasetKey || "bank";
    const { response, payload } = await UFCEClient.fetchJson(`${apiPrefix}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_key: normalized }),
    });
    if (!response.ok) {
      window.alert("Failed to create a session.");
      return;
    }
    window.location.href = `/sessions/${payload.session_id}`;
  }

  function renderDatasetDetail(datasetKey) {
    const dataset = datasetLookup[datasetKey] || datasets[0];
    const panel = document.getElementById("dataset-detail-panel");
    if (!dataset || !panel) {
      if (panel) {
        panel.hidden = true;
      }
      return;
    }
    panel.hidden = false;

    document.querySelectorAll("[data-dataset-card]").forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.datasetKey === dataset.dataset_key);
    });

    document.getElementById("dataset-detail-title").textContent = dataset.display_name;
    document.getElementById("dataset-detail-note").textContent = dataset.support_note;

    const statusEl = document.getElementById("dataset-detail-status");
    statusEl.textContent = dataset.availability_status;
    statusEl.className = `status ${dataset.availability_status === "active" ? "status-active" : "status-blocked"}`;

    document.getElementById("dataset-detail-meta").innerHTML = [
      { label: "Outcome label", value: dataset.outcome_label },
      { label: "Desired outcome", value: dataset.desired_outcome },
      { label: "Training logic", value: dataset.training_logic_version },
      { label: "Step provenance", value: dataset.step_provenance },
    ].map((item) => (
      `<div class="meta-card"><span class="meta-label">${escapeHtml(item.label)}</span><strong class="meta-value meta-value-wrap">${escapeHtml(item.value)}</strong></div>`
    )).join("");

    document.getElementById("dataset-detail-changeable").innerHTML = dataset.f2change.map((feature) => (
      `<li class="tag">${escapeHtml(feature)}</li>`
    )).join("");

    const lockedFeatures = (Array.isArray(dataset.full_feature_list) ? dataset.full_feature_list : [])
      .filter((feature) => !dataset.f2change.includes(feature));
    document.getElementById("dataset-detail-locked").innerHTML = [
      ...lockedFeatures.slice(0, 6).map((feature) => `<li class="tag tag-muted">${escapeHtml(feature)}</li>`),
      ...(lockedFeatures.length > 6 ? [`<li class="tag tag-muted">+${lockedFeatures.length - 6} more</li>`] : []),
    ].join("");

    document.getElementById("dataset-detail-actions").innerHTML = dataset.availability_status === "active"
      ? `<button class="button button-small start-dataset" type="button" data-dataset="${escapeHtml(dataset.dataset_key)}">Start Session</button>`
      : '<span class="button-disabled-note">Reference Only In This MVP</span>';

    document.getElementById("dataset-detail-guides").innerHTML = dataset.feature_guides.map((feature) => (
      `<article class="feature-guide">
        <strong>${escapeHtml(feature.feature_name)}</strong>
        <div class="secondary-copy">Type: ${escapeHtml(feature.feature_kind)} · Changeable: ${escapeHtml(feature.changeable ? "yes" : "no")} · Step: ${escapeHtml(feature.step == null ? "not recorded" : feature.step)}</div>
        <p>${escapeHtml(feature.definition)}</p>
        <p><strong>How to check:</strong> ${escapeHtml(feature.check_guidance)}</p>
        <p><strong>How to change:</strong> ${escapeHtml(feature.change_guidance)}</p>
      </article>`
    )).join("");
  }

  const newSessionButton = document.getElementById("new-session");
  const browseDatasetsButton = document.getElementById("browse-datasets");

  if (newSessionButton) {
    newSessionButton.addEventListener("click", () => createSession("bank"));
  }
  if (browseDatasetsButton) {
    browseDatasetsButton.addEventListener("click", () => {
      const datasetSection = document.getElementById("dataset-section");
      if (datasetSection) {
        datasetSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }

  document.querySelectorAll(".select-dataset").forEach((button) => {
    button.addEventListener("click", () => renderDatasetDetail(button.dataset.dataset || defaultDatasetKey));
  });

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    if (target.classList.contains("start-dataset")) {
      createSession(target.dataset.dataset || "bank");
    }
  });

  if (defaultDatasetKey) {
    renderDatasetDetail(defaultDatasetKey);
  } else if (datasets.length > 0) {
    renderDatasetDetail(datasets[0].dataset_key);
  }
})();
