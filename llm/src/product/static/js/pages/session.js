(function () {
  const UFCEClient = window.UFCEClient || {};
  const escapeHtml = UFCEClient.escapeHtml || ((value) => String(value));
  const initialPageData = UFCEClient.readJsonScript && UFCEClient.readJsonScript("session-page-data");
  if (!initialPageData) {
    return;
  }

  const apiPrefix = initialPageData.apiPrefix;
  const sessionId = initialPageData.sessionId;
  const FIELD_ORDER = ["Income", "Family", "CCAvg", "Education", "Mortgage", "SecuritiesAccount", "CDAccount", "Online", "CreditCard"];
  let currentPageData = {
    ...initialPageData,
    turns: Array.isArray(initialPageData.turns) ? initialPageData.turns : [],
  };
  let pendingClientTurns = [];
  let currentActiveTab = "chat";
  let disclosureState = {
    advancedControlsOpen: null,
    inlineDetailOpenId: null,
    stateRailOpenId: null,
  };
  let isComposerBusy = false;

  let dom = {};

  function bindElements() {
    dom.sessionPage = document.getElementById("session-page");
    dom.datasetBadge = document.getElementById("dataset-badge");
    dom.currentStateBadge = document.getElementById("current-state-badge");
    dom.lifecycleBadge = document.getElementById("lifecycle-badge");
    dom.sessionMetaLine = document.getElementById("session-meta-line");
    dom.chatDatasetBadge = document.getElementById("chat-dataset-badge");
    dom.chatStateBadge = document.getElementById("chat-state-badge");
    dom.chatMetaLine = document.getElementById("chat-meta-line");
    dom.chatTranscript = document.getElementById("chat-transcript");
    dom.transcriptItems = document.getElementById("transcript-items");
    dom.transcriptPendingList = document.getElementById("transcript-pending-list");
    dom.errorBox = document.getElementById("error-box");
    dom.composerBar = document.getElementById("chat-composer-bar");
    dom.composerModeChip = document.getElementById("composer-mode-chip");
    dom.composerTitle = document.getElementById("composer-title");
    dom.composerHelp = document.getElementById("composer-help");
    dom.composerInput = document.getElementById("composer-input");
    dom.composerSubmit = document.getElementById("composer-submit");
    dom.newCaseButton = document.getElementById("new-case");
    dom.closeButton = document.getElementById("close-session");
    dom.stateRail = document.getElementById("state-rail");
    dom.advancedControls = document.getElementById("advanced-refinement-controls");
    dom.previewModal = document.getElementById("preview-modal");
    dom.previewTitle = document.getElementById("preview-title");
    dom.previewContent = document.getElementById("preview-content");
  }

  function currentSession() {
    return currentPageData.session || {};
  }

  function currentSessionRenderHints() {
    const session = currentSession();
    return session && typeof session.render_hints === "object" ? session.render_hints : {};
  }

  function currentComposerContext() {
    return currentPageData.composerContext || {};
  }

  function currentSubmitTarget() {
    if (!dom.composerBar) {
      return currentComposerContext().submit_target || null;
    }
    return dom.composerBar.dataset.submitTarget || currentComposerContext().submit_target || null;
  }

  function isDesktopLayout() {
    return window.matchMedia("(min-width: 1121px)").matches;
  }

  function isNearBottom(container) {
    if (!container) {
      return true;
    }
    const distance = container.scrollHeight - container.scrollTop - container.clientHeight;
    return distance < 120;
  }

  function scrollTranscriptToBottom() {
    if (!dom.chatTranscript) {
      return;
    }
    dom.chatTranscript.scrollTop = dom.chatTranscript.scrollHeight;
  }

  function showError(message) {
    if (!dom.errorBox) {
      return;
    }
    dom.errorBox.hidden = false;
    dom.errorBox.textContent = message;
  }

  function clearError() {
    if (!dom.errorBox) {
      return;
    }
    dom.errorBox.hidden = true;
    dom.errorBox.textContent = "";
  }

  function createPendingTurn(userInput, submitTarget) {
    return {
      localId: `local-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
      submitTarget,
      userInput,
      status: "sending",
      errorMessage: "",
    };
  }

  function pendingTurnItems(turn) {
    const sending = turn.status === "sending";
    const items = [
      {
        kind: "user_message",
        id: `${turn.localId}:user`,
        turn_id: turn.localId,
        text: turn.userInput,
        pending: sending,
        failed: !sending,
        error_message: sending ? "" : (turn.errorMessage || "Message was not sent. Try again."),
      },
    ];
    if (sending) {
      items.push({
        kind: "assistant_message",
        id: `${turn.localId}:assistant`,
        turn_id: turn.localId,
        text: turn.submitTarget === "refinements" ? "Applying the refinement..." : "Working on it...",
        pending: true,
      });
    }
    return items;
  }

  function renderPendingTurns() {
    if (!dom.transcriptPendingList) {
      return;
    }
    const items = pendingClientTurns.flatMap(pendingTurnItems);
    dom.transcriptPendingList.innerHTML = items.map(renderTranscriptItem).join("");
    syncInlineDetailLabels();
  }

  function markPendingTurnFailed(localId, errorMessage) {
    pendingClientTurns = pendingClientTurns.map((turn) => (
      turn.localId === localId
        ? { ...turn, status: "failed", errorMessage: errorMessage || "Request failed." }
        : turn
    ));
    renderPendingTurns();
  }

  function removePendingTurn(localId) {
    pendingClientTurns = pendingClientTurns.filter((turn) => turn.localId !== localId);
    renderPendingTurns();
  }

  function syncActiveTabUI() {
    if (!dom.sessionPage) {
      return;
    }
    dom.sessionPage.dataset.activeTab = currentActiveTab;
    document.querySelectorAll("[data-session-tab]").forEach((button) => {
      const tab = button.getAttribute("data-session-tab");
      const active = tab === currentActiveTab;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
  }

  function setActiveTab(tabName) {
    currentActiveTab = tabName === "context" ? "context" : "chat";
    syncActiveTabUI();
  }

  function stringOrNull(value) {
    return typeof value === "string" && value ? value : null;
  }

  function pageStateFromSessionRenderHints(session) {
    const renderHints = session && typeof session.render_hints === "object" ? session.render_hints : null;
    const pageState = renderHints ? stringOrNull(renderHints.page_state) : null;
    if (pageState) {
      return pageState;
    }
    if ((session.turn_count || 0) === 0) {
      return "fresh";
    }
    return currentPageData.pageState || "restart_required";
  }

  function markerToneFromActionType(actionType) {
    if (["provide_missing_fields", "clarify_refinement"].includes(actionType)) {
      return "info";
    }
    if (actionType === "no_action_required") {
      return "success";
    }
    if (["relax_constraints_or_restart", "start_new_case"].includes(actionType)) {
      return "warning";
    }
    return null;
  }

  function pageStateLabel(pageState) {
    if (pageState === "clarification") {
      return "needs clarification";
    }
    if (pageState === "runtime_success") {
      return "runtime success";
    }
    if (pageState === "runtime_reject") {
      return "runtime reject";
    }
    if (pageState === "refinement_clarification") {
      return "refinement clarification";
    }
    if (pageState === "restart_required") {
      return "restart required";
    }
    return pageState || "fresh";
  }

  function badgeTone(state) {
    if (["RUNTIME_SUCCESS", "runtime_success_view", "runtime_success", "success"].includes(state)) {
      return "success";
    }
    if (["RUNTIME_REJECT", "runtime_reject_view", "runtime_reject", "restart_required", "warning"].includes(state)) {
      return "warning";
    }
    if (["NEEDS_CLARIFICATION", "needs_clarification_input", "refinement_clarification_view", "clarification", "refinement_clarification", "info"].includes(state)) {
      return "info";
    }
    if (["CONFLICT", "UNSUPPORTED_REQUEST", "PARSER_FAILURE", "danger"].includes(state)) {
      return "danger";
    }
    if (state === "archived") {
      return "warning";
    }
    return "neutral";
  }

  function restartHelperCopy(session) {
    if (session.case_completion_reason === "runtime_success") {
      return "This case is complete. A recommendation has been provided, so start a new case to check another profile.";
    }
    if (session.case_completion_reason === "runtime_reject") {
      return "This case is complete, but no viable recommendation was found under the current request. Start a new case to try a different profile or constraints.";
    }
    if (session.case_completion_reason === "clarification_limit_reached") {
      return "The clarification limit was reached for this case. Start a new case with one complete bank profile.";
    }
    if (session.case_completion_reason === "conflict") {
      return "This case ended because the request contained conflicting values. Start a new case with one corrected bank profile.";
    }
    if (session.case_completion_reason === "unsupported_request") {
      return "This request is outside the supported bank-profile flow. Start a new case with one complete bank profile.";
    }
    if (session.case_completion_reason === "parser_failure") {
      return "The request could not be safely interpreted. Start a new case and resubmit one complete bank profile.";
    }
    return "This case has been completed. Start a new case whenever you're ready.";
  }

  function formatFieldList(fields) {
    const items = Array.isArray(fields) ? fields.filter((item) => typeof item === "string" && item.trim()) : [];
    if (items.length === 0) {
      return "";
    }
    if (items.length === 1) {
      return items[0];
    }
    if (items.length === 2) {
      return `${items[0]} and ${items[1]}`;
    }
    return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
  }

  function runtimeChangedFeatureCount(explanationPayload) {
    if (!explanationPayload || typeof explanationPayload !== "object") {
      return null;
    }
    const counterfactualSummary = explanationPayload.counterfactual_summary;
    if (counterfactualSummary && typeof counterfactualSummary === "object" && counterfactualSummary.profile_diff && typeof counterfactualSummary.profile_diff === "object") {
      return Object.keys(counterfactualSummary.profile_diff).length;
    }
    if (Array.isArray(explanationPayload.changed_fields)) {
      return explanationPayload.changed_fields.length;
    }
    return null;
  }

  function runtimeRejectMode(explanationPayload) {
    if (!explanationPayload || typeof explanationPayload !== "object") {
      return null;
    }
    const reasonCodes = Array.isArray(explanationPayload.reason_codes)
      ? explanationPayload.reason_codes.filter((item) => typeof item === "string")
      : [];
    if (reasonCodes.includes("REQUEST_CONSTRAINTS_BLOCKED")) {
      return "constraints_blocked";
    }
    if (reasonCodes.includes("INVALID_COUNTERFACTUAL_BLOCKED")) {
      return "invalid_counterfactual_blocked";
    }
    if (reasonCodes.length > 0) {
      return "runtime_reject";
    }
    return null;
  }

  function buildLatestRuntimeSummary(turn) {
    if (!turn || typeof turn !== "object") {
      return null;
    }
    const explanationPayload = turn.explanation_payload && typeof turn.explanation_payload === "object"
      ? turn.explanation_payload
      : {};
    const summaryType = stringOrNull(explanationPayload.summary_type);
    const changedFeatureCount = runtimeChangedFeatureCount(explanationPayload);
    if (turn.public_state === "RUNTIME_SUCCESS") {
      if (summaryType === "no_recourse_needed") {
        return {
          kind: "success",
          headline: "Current profile already qualifies",
          supporting_copy: "The current bank profile already reaches the desired outcome with no further changes.",
          summary_type: summaryType,
          changed_feature_count: changedFeatureCount,
          reject_mode: null,
        };
      }
      if (summaryType === "counterfactual_found") {
        return {
          kind: "success",
          headline: "Validated counterfactual recommendation",
          supporting_copy: "A runtime-backed recommendation is available for the current request.",
          summary_type: summaryType,
          changed_feature_count: changedFeatureCount,
          reject_mode: null,
        };
      }
      return {
        kind: "success",
        headline: "Runtime-backed recommendation",
        supporting_copy: "Runtime completed with a recommendation for the current request.",
        summary_type: summaryType,
        changed_feature_count: changedFeatureCount,
        reject_mode: null,
      };
    }
    const rejectMode = runtimeRejectMode(explanationPayload);
    if (rejectMode === "constraints_blocked") {
      return {
        kind: "reject",
        headline: "Current constraints block a recommendation",
        supporting_copy: "No recommendation can be shown under the active request-specific constraints.",
        summary_type: summaryType,
        changed_feature_count: changedFeatureCount,
        reject_mode: rejectMode,
      };
    }
    if (rejectMode === "invalid_counterfactual_blocked") {
      return {
        kind: "reject",
        headline: "No safe recommendation available",
        supporting_copy: "A candidate was generated, but it could not be shown after validation.",
        summary_type: summaryType,
        changed_feature_count: changedFeatureCount,
        reject_mode: rejectMode,
      };
    }
    return {
      kind: "reject",
      headline: "No recommendation available",
      supporting_copy: "Runtime completed without a feasible recommendation for the current request.",
      summary_type: summaryType,
      changed_feature_count: changedFeatureCount,
      reject_mode: rejectMode,
    };
  }

  function turnCarriesRuntimeOutcome(turn, latestRuntimeBackedTurnId) {
    if (!turn || typeof turn !== "object") {
      return false;
    }
    if (!["RUNTIME_SUCCESS", "RUNTIME_REJECT"].includes(turn.public_state)) {
      return false;
    }
    if (latestRuntimeBackedTurnId && turn.turn_id === latestRuntimeBackedTurnId) {
      return true;
    }
    if (turn.explanation_payload && typeof turn.explanation_payload === "object") {
      return true;
    }
    if (turn.refinement_status === "applied") {
      return true;
    }
    return turn.turn_kind === "message";
  }

  function turnIsOriginalClarification(turn) {
    if (!turn || typeof turn !== "object") {
      return false;
    }
    const clarificationPayload = turn.clarification_payload;
    const clarificationType = clarificationPayload && typeof clarificationPayload === "object"
      ? clarificationPayload.clarification_type
      : null;
    if (clarificationType === "refinement_clarification") {
      return false;
    }
    if (turn.turn_kind === "refinement" && turn.refinement_status === "clarification_required") {
      return false;
    }
    return turn.public_state === "NEEDS_CLARIFICATION" || Boolean(clarificationPayload && typeof clarificationPayload === "object");
  }

  function hasPendingOriginalClarification(session) {
    const uiReview = session.ui_review;
    if (uiReview && uiReview.clarification_type === "refinement_clarification") {
      return false;
    }
    return Boolean(session.has_pending_clarification || session.current_public_state === "NEEDS_CLARIFICATION");
  }

  function findLatestRuntimeTurn(session, turns, latestVisibleTurn) {
    if (turnCarriesRuntimeOutcome(latestVisibleTurn, session.latest_runtime_backed_turn_id)) {
      return latestVisibleTurn;
    }
    if (session.latest_runtime_backed_turn_id) {
      for (const turn of turns) {
        if (turn.turn_id !== session.latest_runtime_backed_turn_id) {
          continue;
        }
        if (turnCarriesRuntimeOutcome(turn, session.latest_runtime_backed_turn_id)) {
          return turn;
        }
      }
    }
    for (const turn of turns) {
      if (turnCarriesRuntimeOutcome(turn, session.latest_runtime_backed_turn_id)) {
        return turn;
      }
    }
    return null;
  }

  function resolvePageState(session, latestVisibleTurn, latestRuntimeSummary) {
    const sameCaseContinuationAllowed = Boolean(session.refinement_allowed);
    if (session.restart_required && !sameCaseContinuationAllowed) {
      return "restart_required";
    }
    if (session.has_pending_refinement_clarification) {
      return "refinement_clarification";
    }
    if (turnIsOriginalClarification(latestVisibleTurn)) {
      return "clarification";
    }
    if (!latestVisibleTurn && hasPendingOriginalClarification(session)) {
      return "clarification";
    }
    if (turnCarriesRuntimeOutcome(latestVisibleTurn, session.latest_runtime_backed_turn_id)) {
      if (latestVisibleTurn.public_state === "RUNTIME_SUCCESS") {
        return "runtime_success";
      }
      if (latestVisibleTurn.public_state === "RUNTIME_REJECT") {
        return "runtime_reject";
      }
    }
    if (latestRuntimeSummary) {
      return latestRuntimeSummary.kind === "success" ? "runtime_success" : "runtime_reject";
    }
    if ((session.turn_count || 0) === 0) {
      return "fresh";
    }
    if (session.current_public_state === "RUNTIME_SUCCESS") {
      return "runtime_success";
    }
    if (session.current_public_state === "RUNTIME_REJECT") {
      return "runtime_reject";
    }
    if (session.current_public_state === "NEEDS_CLARIFICATION") {
      return "clarification";
    }
    return "restart_required";
  }

  function resolveComposerMode(session, pageState) {
    if (session.is_read_only || pageState === "restart_required") {
      return "disabled";
    }
    if (["runtime_success", "runtime_reject", "refinement_clarification"].includes(pageState)) {
      return "refinement";
    }
    return "message";
  }

  function buildComposerContext(session, composerMode, pageState) {
    const renderHints = session && typeof session.render_hints === "object" ? session.render_hints : null;
    const serverComposerContext = renderHints && renderHints.composer_context && typeof renderHints.composer_context === "object"
      ? renderHints.composer_context
      : null;
    if (serverComposerContext) {
      return serverComposerContext;
    }
    if (composerMode === "disabled") {
      return {
        mode: "disabled",
        submit_target: null,
        mode_chip_text: null,
        title: null,
        help_text: restartHelperCopy(session),
        placeholder: null,
        button_label: null,
        advanced_controls_relevant: false,
        hidden: true,
        disabled: true,
      };
    }
    if (composerMode === "refinement") {
      let helpText = "Continue the current case in natural language. The runtime flow stays unchanged, and advanced structured controls remain available on demand in the context pane.";
      let placeholder = "Keep the current case but refine it naturally, for example: Do not change Income. Keep Mortgage below 120.";
      if (pageState === "refinement_clarification") {
        helpText = "Clarify the pending refinement so the current case can continue.";
        placeholder = "Clarify the refinement intent naturally, for example: Keep max changed features at one.";
      }
      return {
        mode: "refinement",
        submit_target: "refinements",
        mode_chip_text: "Continuing this case",
        title: "Refine this case",
        help_text: helpText,
        placeholder,
        button_label: "Apply Refinement",
        advanced_controls_relevant: true,
        hidden: false,
        disabled: Boolean(session.is_read_only || !session.refinement_allowed),
      };
    }
    let helpText = "Describe the target bank profile naturally.";
    let placeholder = "Describe the target bank profile naturally.";
    if (pageState === "clarification") {
      helpText = "Add the missing bank-profile details to continue the current case.";
      placeholder = "Add the missing profile details to continue this case.";
    }
    return {
      mode: "message",
      submit_target: "messages",
      mode_chip_text: null,
      title: "Message",
      help_text: helpText,
      placeholder,
      button_label: "Send Message",
      advanced_controls_relevant: false,
      hidden: false,
      disabled: Boolean(session.is_read_only),
    };
  }

  function showAdvancedControlsByDefault(session) {
    const renderHints = session && typeof session.render_hints === "object" ? session.render_hints : null;
    const composerContext = renderHints && renderHints.composer_context && typeof renderHints.composer_context === "object"
      ? renderHints.composer_context
      : null;
    return Boolean(
      renderHints
      && renderHints.page_state === "refinement_clarification"
      && composerContext
      && composerContext.advanced_controls_relevant
    );
  }

  function buildConstraintSummaryLine(items, emptyText) {
    if (!Array.isArray(items) || items.length === 0) {
      return emptyText;
    }
    const labels = items.slice(0, 2).map((item) => String(item.label));
    let summary = labels.join(" · ");
    if (items.length > 2) {
      summary = `${summary} · +${items.length - 2} more`;
    }
    return summary;
  }

  function buildReviewSummary(uiReview) {
    if (!uiReview || !Array.isArray(uiReview.profile_fields) || uiReview.profile_fields.length === 0) {
      return {
        headline: "No interpreted profile yet",
        summary_line: "Submit a natural-language bank request to populate the review.",
        constraints_line: "No active hard constraints.",
        preferences_line: "No active soft preferences.",
      };
    }
    const resolvedFields = uiReview.profile_fields.filter((field) => !field.missing && field.value !== null && field.value !== undefined && field.display_value !== "Not provided");
    const totalFields = uiReview.profile_fields.length;
    const providedFields = resolvedFields.length;
    const previewPairs = resolvedFields.slice(0, 3).map((field) => `${field.label} ${field.display_value}`);
    const summaryLine = previewPairs.length > 0 ? previewPairs.join(" · ") : `${providedFields}/${totalFields} fields interpreted`;
    return {
      headline: `${providedFields}/${totalFields} profile fields interpreted`,
      summary_line: summaryLine,
      constraints_line: buildConstraintSummaryLine(uiReview.constraints, "No active hard constraints."),
      preferences_line: buildConstraintSummaryLine(uiReview.preferences, "No active soft preferences."),
    };
  }

  function buildClarificationSummaryBody(uiReview, latestTurn) {
    if (uiReview && uiReview.clarification_message) {
      return uiReview.clarification_message;
    }
    if (latestTurn && typeof latestTurn.assistant_text === "string") {
      return latestTurn.assistant_text;
    }
    return "More details are required before runtime can continue.";
  }

  function buildNextActionSummary(session, pageState, latestTurn, latestRuntimeSummary) {
    const renderHints = session && typeof session.render_hints === "object" ? session.render_hints : null;
    if (pageState === "fresh") {
      return null;
    }
    const uiResponseSummary = extractUiResponseSummary(latestTurn);
    if (uiResponseSummary) {
      const nextActions = Array.isArray(uiResponseSummary.next_actions)
        ? uiResponseSummary.next_actions.filter((item) => item && typeof item === "object")
        : [];
      let headline = renderHints && typeof renderHints.primary_chat_text === "string"
        ? renderHints.primary_chat_text
        : String(uiResponseSummary.headline || "");
      let body = renderHints ? nextActionBodyFromRenderHints(renderHints) : String(uiResponseSummary.short_summary || "");
      if (!body && latestRuntimeSummary) {
        body = latestRuntimeSummary.supporting_copy + (pageState === "restart_required" ? " " + restartHelperCopy(session) : "");
      }
      const facts = renderHints && Array.isArray(renderHints.primary_action_items) && renderHints.primary_action_items.length > 0
        ? renderHints.primary_action_items.filter((item) => typeof item === "string")
        : nextActions.map((item) => String(item.label || "")).filter(Boolean);
      return {
        title: renderHints && renderHints.primary_action_type === "no_action_required" ? "Result" : "Next Action",
        headline: headline,
        summary_line: headline,
        body: body,
        facts: facts,
        tone: String(uiResponseSummary.tone || "info"),
        response_kind: String(uiResponseSummary.response_kind || ""),
        changed_items: Array.isArray(uiResponseSummary.changed_items)
          ? uiResponseSummary.changed_items.filter((item) => item && typeof item === "object")
          : [],
        blocked_reasons: Array.isArray(uiResponseSummary.blocked_reasons)
          ? uiResponseSummary.blocked_reasons.filter((item) => item && typeof item === "object")
          : [],
        next_actions: nextActions,
      };
    }
    if (!renderHints) {
      if (latestRuntimeSummary) {
        return {
          title: latestRuntimeSummary.kind === "success" ? "Result" : "Next Action",
          headline: latestRuntimeSummary.headline,
          summary_line: latestRuntimeSummary.headline,
          body: latestRuntimeSummary.supporting_copy + (pageState === "restart_required" ? " " + restartHelperCopy(session) : ""),
          facts: buildInlineOutcomeFacts(latestRuntimeSummary)
        };
      }
      return {
        title: "Next Action",
        headline: "Case Complete",
        summary_line: restartHelperCopy(session),
        body: restartHelperCopy(session),
        facts: [],
      };
    }
    
    let headline = renderHints.primary_chat_text || "";
    let body = nextActionBodyFromRenderHints(renderHints);
    if (!body || body === headline) {
      if (latestRuntimeSummary) {
        headline = latestRuntimeSummary.headline;
        body = latestRuntimeSummary.supporting_copy + (pageState === "restart_required" ? " " + restartHelperCopy(session) : "");
      } else {
        body = "";
      }
    }

    return {
      title: renderHints.primary_action_type === "no_action_required" ? "Result" : "Next Action",
      headline: headline,
      summary_line: headline,
      body: body,
      facts: Array.isArray(renderHints.primary_action_items)
        ? renderHints.primary_action_items.filter((item) => typeof item === "string")
        : (latestRuntimeSummary ? buildInlineOutcomeFacts(latestRuntimeSummary) : []),
      tone: null,
      response_kind: "",
      changed_items: [],
      blocked_reasons: [],
      next_actions: [],
    };
  }

  function extractUiResponseSummary(turn) {
    if (!turn || typeof turn !== "object") {
      return null;
    }
    const summary = turn.ui_response_summary;
    if (!summary || typeof summary !== "object") {
      return null;
    }
    if (
      typeof summary.response_kind !== "string"
      || typeof summary.tone !== "string"
      || typeof summary.headline !== "string"
      || typeof summary.short_summary !== "string"
    ) {
      return null;
    }
    return summary;
  }

  function nextActionBodyFromRenderHints(renderHints) {
    const actionType = stringOrNull(renderHints.primary_action_type) || "none";
    if (actionType === "provide_missing_fields") {
      return "Reply with the missing fields to continue the current case.";
    }
    if (actionType === "no_action_required") {
      return "Your analysis is complete. You can refine constraints if you'd like a different recommendation, or start a new case.";
    }
    if (actionType === "relax_constraints_or_restart") {
      return "No path to approval was found under your current constraints. Try relaxing some restrictions or start a new case with a different profile.";
    }
    if (actionType === "clarify_refinement") {
      return "I need more details about how you'd like to adjust the constraints before I can re-run the analysis.";
    }
    if (actionType === "start_new_case") {
      return "This case is complete. Start a new case to check another bank profile.";
    }
    if (actionType === "start_case") {
      return "Describe your bank profile in natural language to begin a loan assessment.";
    }
    return stringOrNull(renderHints.supporting_detail_body) || "";
  }

  function buildChatHeaderSummary(session, pageState) {
    let metaLine = `Session ${session.session_id} · ${session.turn_count} turns`;
    if ((session.clarification_turns_used || 0) > 0) {
      metaLine = `${metaLine} · ${session.clarification_turns_used} clarification turns`;
    }
    return {
      dataset_label: `Dataset ${session.dataset_key}`,
      state_label: pageStateLabel(pageState),
      state_tone: pageState,
      meta_line: metaLine,
    };
  }

  function buildContextSections(pageState, reviewSummary, nextActionSummary) {
    const sections = [
      {
        id: "review",
        title: "Review",
        summary_line: reviewSummary.summary_line,
      },
    ];
    if (nextActionSummary) {
      sections.push({
        id: "next-action",
        title: nextActionSummary.title,
        summary_line: nextActionSummary.headline,
      });
    }
    sections.push({
      id: "technical",
      title: "Technical Details",
      summary_line: "Artifacts, payloads, and raw traces",
    });
    return sections;
  }

  function buildClarificationMessage(clarificationPayload, refinementStatus) {
    if (refinementStatus === "clarification_required" && (!clarificationPayload || typeof clarificationPayload !== "object")) {
      return "Clarification is required before the refinement can be applied.";
    }
    if (!clarificationPayload || typeof clarificationPayload !== "object") {
      return null;
    }
    const clarificationType = String(clarificationPayload.clarification_type || "");
    const replyStrategy = String(clarificationPayload.reply_strategy || "");
    const conflicts = Array.isArray(clarificationPayload.conflicts)
      ? clarificationPayload.conflicts.filter((item) => typeof item === "string")
      : [];
    const missingFields = Array.isArray(clarificationPayload.missing_fields)
      ? clarificationPayload.missing_fields.filter((item) => typeof item === "string")
      : [];
    const carriedForwardFields = Array.isArray(clarificationPayload.carried_forward_fields)
      ? clarificationPayload.carried_forward_fields.filter((item) => typeof item === "string")
      : [];
    if (clarificationType === "clarification_limit_reached") {
      return "The clarification limit was reached for this case. Start a new case with one complete bank profile.";
    }
    if (clarificationType === "refinement_clarification") {
      return conflicts[0] || "I need a bit more detail about your refinement before I can apply it.";
    }
    if (clarificationType === "conflict_resolution") {
      if (conflicts.length > 0) {
        return `Your request contains conflicting instructions: ${conflicts.join("; ")}. Start a new case and submit one corrected bank profile.`;
      }
      return "Your request contains conflicting instructions. Start a new case and submit one corrected bank profile.";
    }
    if (replyStrategy === "missing_fields_only" && missingFields.length > 0) {
      if (carriedForwardFields.length > 0) {
        return `Reply with only the missing fields: ${formatFieldList(missingFields)}. I'll keep the values already provided for ${formatFieldList(carriedForwardFields)}.`;
      }
      return `Reply with only the missing fields: ${formatFieldList(missingFields)}.`;
    }
    if (replyStrategy === "start_new_case") {
      return String(clarificationPayload.next_required_input || "Start a new case and submit one corrected bank profile.");
    }
    if (missingFields.length > 0) {
      if (carriedForwardFields.length > 0) {
        return `Reply with only the missing fields: ${formatFieldList(missingFields)}. I'll keep the values already provided for ${formatFieldList(carriedForwardFields)}.`;
      }
      return `Reply with only the missing fields: ${formatFieldList(missingFields)}.`;
    }
    return stringOrNull(clarificationPayload.next_required_input);
  }

  function buildStreamStateMarker(turn) {
    const renderHints = turn && turn.render_hints && typeof turn.render_hints === "object"
      ? turn.render_hints
      : null;
    const label = renderHints ? stringOrNull(renderHints.state_marker_label) : null;
    if (!label) {
      return null;
    }
    const tone = markerToneFromActionType(stringOrNull(renderHints.primary_action_type) || "none");
    if (!tone) {
      return null;
    }
    return { tone, label };
  }

  function buildInlineOutcomeFacts(summary) {
    const facts = [];
    if (summary.summary_type === "no_recourse_needed") {
      facts.push("No changes needed");
    } else if (summary.summary_type === "counterfactual_found") {
      facts.push("Runtime-validated result");
    } else if (typeof summary.summary_type === "string" && summary.summary_type) {
      facts.push(summary.summary_type.replaceAll("_", " "));
    }
    if (Number.isInteger(summary.changed_feature_count)) {
      const plural = summary.changed_feature_count === 1 ? "" : "s";
      facts.push(`${summary.changed_feature_count} key change${plural}`);
    } else if (summary.reject_mode === "constraints_blocked") {
      facts.push("Blocked by active constraints");
    } else if (summary.reject_mode === "invalid_counterfactual_blocked") {
      facts.push("Validation blocked exposure");
    }
    return facts.slice(0, 2);
  }

  function buildFreshWelcomeMessage(session) {
    if (session.dataset_key === "grad") {
      return "Describe one graduate-admission profile to start a new case.";
    }
    return "Describe one bank profile in natural language to start a new case.";
  }

  function buildTranscriptVisibleAssistantText(turn, latestRuntimeBackedTurnId) {
    if (!turn || typeof turn !== "object") {
      return "";
    }
    const renderHints = turn.render_hints && typeof turn.render_hints === "object"
      ? turn.render_hints
      : null;
    if (renderHints && stringOrNull(renderHints.primary_chat_text)) {
      return renderHints.primary_chat_text;
    }
    return String(turn.assistant_text || "");
  }

  function buildInlineDetailToggleItem(turn, latestRuntimeBackedTurnId) {
    if (!turn || typeof turn !== "object") {
      return null;
    }
    const turnId = stringOrNull(turn.turn_id) || `turn-${turn.turn_index || "unknown"}`;
    const renderHints = turn.render_hints && typeof turn.render_hints === "object"
      ? turn.render_hints
      : null;
    if (!renderHints) {
      return null;
    }
    const detailTitle = stringOrNull(renderHints.supporting_detail_title);
    const detailBody = stringOrNull(renderHints.supporting_detail_body) || "";
    const facts = Array.isArray(renderHints.supporting_detail_facts)
      ? renderHints.supporting_detail_facts.filter((item) => typeof item === "string")
      : [];
    if (!detailTitle && !detailBody && facts.length === 0) {
      return null;
    }
    return {
      kind: "inline_detail_toggle",
      id: `${turnId}:detail`,
      turn_id: turn.turn_id,
      closed_label: "View details",
      open_label: "Hide details",
      detail_title: detailTitle || "Details",
      detail_body: detailBody,
      facts: facts.slice(0, 3),
    };
  }

  function buildTurnTranscriptItems(turn, latestRuntimeBackedTurnId) {
    const turnId = stringOrNull(turn.turn_id) || `turn-${turn.turn_index || "unknown"}`;
    const items = [
      {
        kind: "user_message",
        id: `${turnId}:user`,
        turn_id: turn.turn_id,
        text: String(turn.user_input || ""),
        pending: false,
        failed: false,
      },
    ];
    const marker = buildStreamStateMarker(turn);
    if (marker) {
      items.push({
        kind: "system_marker",
        id: `${turnId}:marker`,
        turn_id: turn.turn_id,
        tone: marker.tone,
        label: marker.label,
      });
    }
    items.push({
      kind: "assistant_message",
      id: `${turnId}:assistant`,
      turn_id: turn.turn_id,
      text: buildTranscriptVisibleAssistantText(turn, latestRuntimeBackedTurnId),
      pending: false,
    });
    const detailToggle = buildInlineDetailToggleItem(turn, latestRuntimeBackedTurnId);
    if (detailToggle) {
      items.push(detailToggle);
    }
    return items;
  }

  function buildTranscriptItems(session, turns, pageState) {
    if (!Array.isArray(turns) || turns.length === 0) {
      const renderHints = session && typeof session.render_hints === "object" ? session.render_hints : null;
      return [
        {
          kind: "assistant_message",
          id: "welcome-assistant",
          turn_id: null,
          text: renderHints && stringOrNull(renderHints.primary_chat_text)
            ? renderHints.primary_chat_text
            : buildFreshWelcomeMessage(session),
          pending: false,
        },
      ];
    }
    const items = [];
    turns.slice().reverse().forEach((turn) => {
      items.push(...buildTurnTranscriptItems(turn, session.latest_runtime_backed_turn_id));
    });
    /* Duplicate restart note removed — the per-turn assistant bubble already carries the outcome text. */
    return items;
  }

  function rebuildDerivedPageData(sessionDetail, latestTurnOverride) {
    if (sessionDetail) {
      currentPageData.session = sessionDetail;
    }
    if (latestTurnOverride !== undefined) {
      currentPageData.latestTurn = latestTurnOverride;
    } else if (!currentPageData.latestTurn && currentPageData.turns.length > 0) {
      currentPageData.latestTurn = currentPageData.turns[0];
    }
    const latestVisibleTurn = currentPageData.latestTurn || null;
    const renderHints = currentSessionRenderHints();
    const latestRuntimeTurn = findLatestRuntimeTurn(currentSession(), currentPageData.turns, latestVisibleTurn);
    const latestRuntimeSummary = buildLatestRuntimeSummary(latestRuntimeTurn);
    const pageState = pageStateFromSessionRenderHints(currentSession());
    const composerMode = (renderHints.composer_context && stringOrNull(renderHints.composer_context.mode)) || currentPageData.composerMode || "message";
    currentPageData.pageState = pageState;
    currentPageData.latestRuntimeTurn = latestRuntimeTurn;
    currentPageData.latestRuntimeSummary = latestRuntimeSummary;
    currentPageData.composerMode = composerMode;
    currentPageData.composerContext = buildComposerContext(currentSession(), composerMode, pageState);
    currentPageData.reviewSummary = buildReviewSummary(currentSession().ui_review || null);
    currentPageData.nextActionSummary = buildNextActionSummary(currentSession(), pageState, latestVisibleTurn, latestRuntimeSummary);
    currentPageData.chatHeaderSummary = buildChatHeaderSummary(currentSession(), pageState);
    currentPageData.contextSections = buildContextSections(pageState, currentPageData.reviewSummary, currentPageData.nextActionSummary);
    currentPageData.showAdvancedControlsByDefault = showAdvancedControlsByDefault(currentSession());
    currentPageData.transcriptItems = buildTranscriptItems(currentSession(), currentPageData.turns, pageState);
  }

  function renderTranscriptItem(item) {
    const turnIdAttr = item.turn_id ? ` data-turn-id="${escapeHtml(item.turn_id)}"` : "";
    if (item.kind === "user_message") {
      const statusHtml = item.pending
        ? '<span class="chat-message-status">Sending...</span>'
        : (item.failed ? `<span class="chat-message-status chat-message-status-error">${escapeHtml(item.error_message || "Send failed.")}</span>` : "");
      return `
        <article class="chat-message chat-message--user${item.pending ? " is-pending" : ""}${item.failed ? " is-failed" : ""}" data-transcript-item-id="${escapeHtml(item.id)}"${turnIdAttr}>
          <div class="chat-bubble chat-bubble-user">
            <span class="chat-role">User</span>
            <p>${escapeHtml(item.text || "")}</p>
            ${statusHtml}
          </div>
        </article>
      `;
    }
    if (item.kind === "assistant_message") {
      const statusHtml = item.pending
        ? '<span class="chat-message-status">Working...</span>'
        : "";
      return `
        <article class="chat-message chat-message--assistant${item.pending ? " is-pending" : ""}" data-transcript-item-id="${escapeHtml(item.id)}"${turnIdAttr}>
          <div class="chat-bubble chat-bubble-assistant">
            <span class="chat-role">Assistant</span>
            <p>${escapeHtml(item.text || "")}</p>
            ${statusHtml}
          </div>
        </article>
      `;
    }
    if (item.kind === "system_marker") {
      return `
        <div class="chat-marker chat-marker-${escapeHtml(item.tone || "neutral")}" data-transcript-item-id="${escapeHtml(item.id)}"${turnIdAttr}>
          <span class="pill pill-${escapeHtml(badgeTone(item.tone || "neutral"))}">${escapeHtml(item.label || "")}</span>
        </div>
      `;
    }
    if (item.kind === "inline_detail_toggle") {
      const open = disclosureState.inlineDetailOpenId === item.id ? " open" : "";
      const factsHtml = Array.isArray(item.facts) && item.facts.length > 0
        ? `<div class="fact-row">${item.facts.map((fact) => `<span class="fact-chip">${escapeHtml(fact)}</span>`).join("")}</div>`
        : "";
      return `
        <details class="chat-inline-detail" data-inline-detail-id="${escapeHtml(item.id)}"${turnIdAttr}${open}>
          <summary class="chat-inline-detail-summary">
            <span class="chat-inline-detail-label" data-inline-label data-closed-label="${escapeHtml(item.closed_label || "View details")}" data-open-label="${escapeHtml(item.open_label || "Hide details")}">${escapeHtml(item.closed_label || "View details")}</span>
          </summary>
          <div class="chat-inline-detail-panel">
            <strong class="chat-inline-detail-title">${escapeHtml(item.detail_title || "Details")}</strong>
            ${item.detail_body ? `<p class="chat-inline-detail-copy">${escapeHtml(item.detail_body)}</p>` : ""}
            ${factsHtml}
          </div>
        </details>
      `;
    }
    return "";
  }

  function renderTranscript() {
    if (!dom.transcriptItems) {
      return;
    }
    dom.transcriptItems.innerHTML = (currentPageData.transcriptItems || []).map(renderTranscriptItem).join("");
    renderPendingTurns();
  }

  function renderConstraintItems(items) {
    if (!Array.isArray(items) || items.length === 0) {
      return '<p class="supporting-copy">No items available.</p>';
    }
    return items.map((item) => `
      <div class="constraint-item">
        <strong>${escapeHtml(item.label || "")}</strong>
        <span>${escapeHtml(item.display_value || "")}</span>
      </div>
    `).join("");
  }

  function renderProfileFieldCard(field, profileEditable) {
    const missingClass = field.missing ? " missing" : "";
    const value = field.value === null || field.value === undefined ? "" : String(field.value);
    let content = `<span class="field-value">${escapeHtml(field.display_value || "")}</span>`;
    if (profileEditable) {
      if (field.feature_kind === "binary") {
        content = `
          <select class="profile-field-input" data-profile-field="${escapeHtml(field.field_name)}" data-feature-kind="${escapeHtml(field.feature_kind)}">
            <option value="" ${value === "" ? "selected" : ""}>Not set</option>
            <option value="1" ${value === "1" ? "selected" : ""}>Yes</option>
            <option value="0" ${value === "0" ? "selected" : ""}>No</option>
          </select>
          <div class="chip-row">
            <button class="mini-chip quick-binary" type="button" data-target-field="${escapeHtml(field.field_name)}" data-target-value="1">Yes</button>
            <button class="mini-chip quick-binary" type="button" data-target-field="${escapeHtml(field.field_name)}" data-target-value="0">No</button>
          </div>
        `;
      } else {
        content = `
          <input
            class="profile-field-input"
            data-profile-field="${escapeHtml(field.field_name)}"
            data-feature-kind="${escapeHtml(field.feature_kind)}"
            type="number"
            step="${field.step === null || field.step === undefined ? "any" : escapeHtml(field.step)}"
            value="${escapeHtml(value)}"
          >
        `;
      }
    }
    return `
      <div class="field-card${missingClass}">
        <span class="field-label">${escapeHtml(field.label || field.field_name || "")}</span>
        ${content}
        ${field.missing ? '<span class="field-note">Missing</span>' : ""}
      </div>
    `;
  }

  function renderReviewCard() {
    const reviewSummary = currentPageData.reviewSummary || buildReviewSummary(currentSession().ui_review || null);
    const uiReview = currentSession().ui_review;
    const open = disclosureState.stateRailOpenId === "review-card" ? " open" : "";
    const profileEditable = Boolean(uiReview && uiReview.profile_editable);
    const profileGrid = uiReview && Array.isArray(uiReview.profile_fields) && uiReview.profile_fields.length > 0
      ? `<div class="review-grid" id="profile-review-grid">${uiReview.profile_fields.map((field) => renderProfileFieldCard(field, profileEditable)).join("")}</div>`
      : '<p class="supporting-copy">No structured review is available yet.</p>';
    const constraints = uiReview && Array.isArray(uiReview.constraints) && uiReview.constraints.length > 0
      ? `<div class="constraint-grid">${renderConstraintItems(uiReview.constraints)}</div>`
      : '<p class="supporting-copy">No active hard constraints.</p>';
    const preferences = uiReview && Array.isArray(uiReview.preferences) && uiReview.preferences.length > 0
      ? `<div class="constraint-grid">${renderConstraintItems(uiReview.preferences)}</div>`
      : '<p class="supporting-copy">No active soft preferences.</p>';
    const footer = profileEditable
      ? '<button id="apply-profile-edits" class="button button-ghost" type="button">Use Review Edits</button>'
      : '<p class="supporting-copy" id="review-readonly-copy">This review is read-only in the current state.</p>';
    return `
      <details id="review-card" class="context-section" data-main-section="true"${open}>
        <summary class="context-section-summary">
          <div>
            <span class="context-section-title">Review</span>
            <p id="review-card-summary-line" class="context-summary-line">${escapeHtml(reviewSummary.summary_line || "")}</p>
          </div>
          <span class="context-summary-meta">Open</span>
        </summary>
        <div class="context-section-body">
          <div class="context-summary-block">
            <strong id="review-headline">${escapeHtml(reviewSummary.headline || "")}</strong>
            <p id="review-summary-line">${escapeHtml(reviewSummary.summary_line || "")}</p>
            <p id="review-constraints-line">${escapeHtml(reviewSummary.constraints_line || "")}</p>
            <p id="review-preferences-line">${escapeHtml(reviewSummary.preferences_line || "")}</p>
          </div>
          <details class="context-subdetail review-full-detail">
            <summary>View full review</summary>
            ${profileGrid}
            <div class="section-block" id="review-constraints-section">
              <span class="section-label">Constraints</span>
              ${constraints}
            </div>
            <div class="section-block" id="review-preferences-section">
              <span class="section-label">Preferences</span>
              ${preferences}
            </div>
            ${footer}
          </details>
        </div>
      </details>
    `;
  }

  function renderResultCard() {
    const nextActionSummary = currentPageData.nextActionSummary;
    if (!nextActionSummary) {
      return "";
    }
    const open = disclosureState.stateRailOpenId === "result-card" ? " open" : "";
    const factsHtml = Array.isArray(nextActionSummary.facts) && nextActionSummary.facts.length > 0
      ? `<div id="result-facts-row" class="fact-row">${nextActionSummary.facts.map((fact) => `<span class="fact-chip">${escapeHtml(fact)}</span>`).join("")}</div>`
      : "";
    const responseSummaryCard = renderResponseSummaryCard(nextActionSummary);
    const restartButton = currentPageData.pageState === "restart_required"
      ? '<button class="button" type="button" data-start-new-case="true">Start New Case</button>'
      : "";
    return `
      <details id="result-card" class="context-section" data-main-section="true"${open}>
        <summary class="context-section-summary">
          <div>
            <span class="context-section-title">${escapeHtml(nextActionSummary.title || "Next Action")}</span>
            <p id="result-card-summary-line" class="context-summary-line">${escapeHtml(nextActionSummary.headline || "")}</p>
          </div>
          <span class="context-summary-meta">Open</span>
        </summary>
        <div class="context-section-body">
          <div class="context-summary-block">
            <strong id="result-headline">${escapeHtml(nextActionSummary.headline || "")}</strong>
            <p id="result-summary-copy">${escapeHtml(nextActionSummary.body || "")}</p>
          </div>
          ${factsHtml}
          ${responseSummaryCard}
          ${restartButton}
        </div>
      </details>
    `;
  }

  function renderResponseSummaryCard(summary) {
    if (!summary || typeof summary !== "object" || !summary.tone) {
      return "";
    }
    const tone = String(summary.tone || "info");
    const toneLabel = tone === "success"
      ? "Success"
      : tone === "warning"
        ? "Blocked"
        : tone === "danger"
          ? "Warning"
          : "Needs input";
    const changedItems = Array.isArray(summary.changed_items) ? summary.changed_items : [];
    const blockedReasons = Array.isArray(summary.blocked_reasons) ? summary.blocked_reasons : [];
    const nextActions = Array.isArray(summary.next_actions) ? summary.next_actions : [];
    const changedHtml = changedItems.length > 0
      ? `
      <div class="section-block change-list-block">
        <span class="section-label">What changed</span>
        <ul class="change-list">
          ${changedItems.map((item) => {
            if (item.user_facing_text) {
              return `<li class="change-item">${escapeHtml(String(item.user_facing_text))}</li>`;
            }
            return `<li class="change-item"><strong>${escapeHtml(String(item.display_name || item.field_name || ""))}</strong> <span class="change-arrow" aria-hidden="true">→</span> <span>${escapeHtml(String(item.before))} → ${escapeHtml(String(item.after))}</span></li>`;
          }).join("")}
        </ul>
      </div>
      `
      : "";
    const blockedHtml = blockedReasons.length > 0
      ? `
      <div class="section-block blocked-reason-list-block">
        <span class="section-label">Why blocked</span>
        <ul class="blocked-reason-list">
          ${blockedReasons.map((item) => {
            const fields = Array.isArray(item.fields) && item.fields.length > 0
              ? `<div class="fact-row">${item.fields.map((fieldName) => `<span class="fact-chip">${escapeHtml(String(fieldName))}</span>`).join("")}</div>`
              : "";
            return `<li class="blocked-reason-item"><strong>${escapeHtml(String(item.title || ""))}</strong><p>${escapeHtml(String(item.detail || ""))}</p>${fields}</li>`;
          }).join("")}
        </ul>
      </div>
      `
      : "";
    const actionHtml = nextActions.length > 0
      ? `
      <div class="section-block next-action-list-block">
        <span class="section-label">Next action</span>
        <ul class="next-action-list">
          ${nextActions.map((item) => `<li class="next-action-item${item.primary ? " is-primary" : ""}"><strong>${escapeHtml(String(item.label || ""))}</strong><p>${escapeHtml(String(item.detail || ""))}</p></li>`).join("")}
        </ul>
      </div>
      `
      : "";
    return `
      <div class="response-summary-card response-summary-card--${escapeHtml(tone)}">
        <span class="pill pill-${escapeHtml(badgeTone(tone))}">${escapeHtml(toneLabel)}</span>
        <strong>${escapeHtml(String(summary.headline || ""))}</strong>
        <p>${escapeHtml(String(summary.body || ""))}</p>
      </div>
      ${changedHtml}
      ${blockedHtml}
      ${actionHtml}
    `;
  }

  function renderAdvancedControls() {
    if (!currentComposerContext().advanced_controls_relevant) {
      return "";
    }
    const active = currentSession().active_constraint_spec || {};
    const disallowedChanges = new Set(Array.isArray(active.disallowed_changes) ? active.disallowed_changes.map(String) : []);
    const numericBounds = active.numeric_bounds && typeof active.numeric_bounds === "object" ? active.numeric_bounds : {};
    const disabled = Boolean(currentSession().is_read_only || !currentSession().refinement_allowed);
    const open = disclosureState.advancedControlsOpen === null
      ? Boolean(currentPageData.showAdvancedControlsByDefault)
      : disclosureState.advancedControlsOpen;
    const disabledAttr = disabled ? " disabled" : "";
    return `
      <details id="advanced-refinement-controls" class="context-section context-section-advanced"${open ? " open" : ""}>
        <summary class="context-section-summary">
          <div>
            <span class="context-section-title">Advanced Structured Controls</span>
            <p class="context-summary-line">Blocked fields, numeric bounds, change limits, and soft preferences.</p>
          </div>
          <span class="context-summary-meta">Open</span>
        </summary>
        <div class="context-section-body">
          <div class="control-grid">
            <div class="control-card">
              <span class="section-label">Blocked Fields</span>
              <div class="chip-row" id="blocked-field-controls">
                ${FIELD_ORDER.map((fieldName) => {
                  const selected = disallowedChanges.has(fieldName);
                  return `<button class="constraint-chip${selected ? " is-selected" : ""}" type="button" data-blocked-field="${escapeHtml(fieldName)}" aria-pressed="${selected ? "true" : "false"}">${escapeHtml(fieldName)}</button>`;
                }).join("")}
              </div>
            </div>
            <div class="control-card">
              <span class="section-label">Numeric Bounds</span>
              <div class="bound-grid" id="numeric-bound-controls">
                ${["Income", "CCAvg", "Mortgage"].map((fieldName) => {
                  const bound = numericBounds[fieldName] && typeof numericBounds[fieldName] === "object" ? numericBounds[fieldName] : {};
                  return `
                    <div class="bound-card">
                      <strong>${escapeHtml(fieldName)}</strong>
                      <div class="bound-inputs">
                        <input type="number" step="any" placeholder="Min" data-bound-field="${escapeHtml(fieldName)}" data-bound-key="min" value="${escapeHtml(bound.min === undefined ? "" : String(bound.min))}"${disabledAttr}>
                        <input type="number" step="any" placeholder="Max" data-bound-field="${escapeHtml(fieldName)}" data-bound-key="max" value="${escapeHtml(bound.max === undefined ? "" : String(bound.max))}"${disabledAttr}>
                      </div>
                    </div>
                  `;
                }).join("")}
              </div>
            </div>
            <div class="control-card">
              <span class="section-label">Change Limit And Preference</span>
              <div class="bound-grid">
                <div class="bound-card">
                  <strong>Max changed features</strong>
                  <select id="max-changed-features"${disabledAttr}>
                    <option value="" ${active.max_changed_features === undefined ? "selected" : ""}>No limit</option>
                    <option value="1" ${active.max_changed_features === 1 ? "selected" : ""}>1</option>
                    <option value="2" ${active.max_changed_features === 2 ? "selected" : ""}>2</option>
                    <option value="3" ${active.max_changed_features === 3 ? "selected" : ""}>3</option>
                  </select>
                </div>
                <div class="bound-card">
                  <strong>Soft preference</strong>
                  <label class="constraint-item">
                    <input id="prefer-fewer-changes" type="checkbox" ${active.prefer_fewer_changes ? "checked" : ""}${disabledAttr}>
                    <span>Prefer fewer changes</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
          <button id="apply-structured-refinement" class="button button-ghost button-block" type="button"${disabledAttr}>Apply Structured Refinement</button>
        </div>
      </details>
    `;
  }

  function renderJsonBlock(value) {
    return escapeHtml(JSON.stringify(value, null, 2));
  }

  function renderTechnicalTurn(turn) {
    const artifactRefs = turn.artifact_refs || {};
    const previewUrls = artifactRefs.preview_urls || {};
    const downloadUrls = artifactRefs.download_urls || {};
    const artifactRows = Object.keys(downloadUrls).length > 0
      ? Object.entries(downloadUrls).map(([fileName, downloadUrl]) => `
          <div class="artifact-row">
            <a href="${escapeHtml(downloadUrl)}">${escapeHtml(fileName)}</a>
            ${Object.prototype.hasOwnProperty.call(previewUrls, fileName)
              ? `<button class="button button-ghost preview-button" type="button" data-preview-url="${escapeHtml(previewUrls[fileName])}" data-preview-name="${escapeHtml(fileName)}">Preview</button>`
              : ""}
          </div>
        `).join("")
      : '<p class="supporting-copy">No artifacts were saved for this turn.</p>';
    return `
      <article class="technical-turn" data-turn-id="${escapeHtml(turn.turn_id)}">
        <div class="technical-head">
          <div class="pill-row">
            <span class="pill pill-${escapeHtml(badgeTone(turn.public_state))}">${escapeHtml(turn.public_state || "")}</span>
            <span class="pill pill-neutral">Turn ${escapeHtml(turn.turn_index)}</span>
            ${turn.turn_kind === "refinement" ? '<span class="pill pill-info">Refinement</span>' : ""}
          </div>
        </div>
        <details class="nested-detail">
          <summary>Debug summary</summary>
          <pre>${renderJsonBlock(turn.debug_summary || {})}</pre>
        </details>
        ${turn.clarification_payload ? `
          <details class="nested-detail">
            <summary>Clarification payload</summary>
            <pre>${renderJsonBlock(turn.clarification_payload)}</pre>
          </details>
        ` : ""}
        ${turn.explanation_payload ? `
          <details class="nested-detail">
            <summary>Explanation payload</summary>
            <pre>${renderJsonBlock(turn.explanation_payload)}</pre>
          </details>
        ` : ""}
        ${turn.constraint_feedback_delta ? `
          <details class="nested-detail">
            <summary>Refinement delta</summary>
            <pre>${renderJsonBlock(turn.constraint_feedback_delta)}</pre>
          </details>
        ` : ""}
        <div class="section-block">
          <span class="section-label">Artifacts</span>
          ${artifactRows}
        </div>
      </article>
    `;
  }

  function renderTechnicalDrawer() {
    const open = disclosureState.stateRailOpenId === "technical-drawer" ? " open" : "";
    return `
      <details id="technical-drawer" class="context-section technical-drawer" data-main-section="true"${open}>
        <summary class="context-section-summary">
          <div>
            <span class="context-section-title">Technical Details</span>
            <p class="context-summary-line">Artifacts, payloads, and raw traces</p>
          </div>
          <span class="context-summary-meta">Open</span>
        </summary>
        <div class="context-section-body">
          <div class="supporting-copy">Artifacts, previews, debug summaries, and raw payloads stay available here without dominating the session flow.</div>
          <div id="technical-turns" class="technical-turns">
            ${currentPageData.turns.map(renderTechnicalTurn).join("")}
          </div>
        </div>
      </details>
    `;
  }

  function renderContextRail() {
    if (!dom.stateRail) {
      return;
    }
    dom.stateRail.dataset.pageState = currentPageData.pageState || "";
    dom.stateRail.innerHTML = [
      renderReviewCard(),
      renderResultCard(),
      renderAdvancedControls(),
      renderTechnicalDrawer(),
    ].join("");
  }

  function setBadgeContent(element, tone, text, pageState) {
    if (!element) {
      return;
    }
    element.className = `pill pill-${badgeTone(tone)}`;
    element.textContent = text;
    if (pageState) {
      element.dataset.pageState = pageState;
    }
  }

  function updateHeaderSummary() {
    const summary = currentPageData.chatHeaderSummary || buildChatHeaderSummary(currentSession(), currentPageData.pageState);
    if (dom.datasetBadge) {
      dom.datasetBadge.textContent = `Dataset ${currentSession().dataset_key || ""}`;
    }
    setBadgeContent(dom.currentStateBadge, currentPageData.pageState, pageStateLabel(currentPageData.pageState), currentPageData.pageState);
    if (dom.lifecycleBadge) {
      const lifecycleStatus = currentSession().lifecycle_status || "active";
      dom.lifecycleBadge.hidden = lifecycleStatus === "active";
      dom.lifecycleBadge.className = `pill pill-${badgeTone(lifecycleStatus)}`;
      dom.lifecycleBadge.textContent = lifecycleStatus;
    }
    if (dom.sessionMetaLine) {
      dom.sessionMetaLine.textContent = summary.meta_line || "";
    }
    if (dom.chatDatasetBadge) {
      dom.chatDatasetBadge.textContent = summary.dataset_label || "";
    }
    if (dom.chatStateBadge) {
      dom.chatStateBadge.className = `pill pill-${badgeTone(summary.state_tone || currentPageData.pageState)}`;
      dom.chatStateBadge.textContent = summary.state_label || "";
    }
    if (dom.chatMetaLine) {
      dom.chatMetaLine.textContent = summary.meta_line || "";
    }
    if (dom.sessionPage) {
      dom.sessionPage.dataset.pageState = currentPageData.pageState || "";
    }
  }

  function applyComposerContext() {
    if (!dom.composerBar || !dom.composerInput || !dom.composerSubmit) {
      return;
    }
    const context = currentComposerContext();
    dom.composerBar.dataset.composerMode = currentPageData.composerMode || context.mode || "";
    dom.composerBar.dataset.submitTarget = context.submit_target || "";
    dom.composerBar.hidden = Boolean(context.hidden);
    if (dom.composerModeChip) {
      dom.composerModeChip.hidden = !context.mode_chip_text;
      dom.composerModeChip.textContent = context.mode_chip_text || "";
    }
    if (dom.composerTitle) {
      dom.composerTitle.hidden = !context.title;
      dom.composerTitle.textContent = context.title || "";
    }
    if (dom.composerHelp) {
      dom.composerHelp.textContent = context.help_text || "";
    }
    dom.composerInput.placeholder = context.placeholder || "";
    dom.composerInput.disabled = Boolean(context.disabled || isComposerBusy);
    dom.composerSubmit.disabled = Boolean(context.disabled || isComposerBusy);
    dom.composerSubmit.textContent = isComposerBusy
      ? (currentSubmitTarget() === "refinements" ? "Applying..." : "Sending...")
      : (context.button_label || "Send Message");
  }

  function syncContextSectionLabels() {
    document.querySelectorAll(".context-section").forEach((detailsEl) => {
      const metaEl = detailsEl.querySelector(".context-summary-meta");
      if (metaEl) {
        metaEl.textContent = detailsEl.open ? "Hide" : "Open";
      }
    });
  }

  function syncInlineDetailLabels() {
    document.querySelectorAll(".chat-inline-detail").forEach((detailsEl) => {
      const labelEl = detailsEl.querySelector("[data-inline-label]");
      if (!labelEl) {
        return;
      }
      const closedLabel = labelEl.getAttribute("data-closed-label") || "View details";
      const openLabel = labelEl.getAttribute("data-open-label") || "Hide details";
      labelEl.textContent = detailsEl.open ? openLabel : closedLabel;
    });
  }

  function renderPageState() {
    updateHeaderSummary();
    renderTranscript();
    renderContextRail();
    bindElements();
    applyComposerContext();
    syncActiveTabUI();
    syncContextSectionLabels();
    syncInlineDetailLabels();
  }

  function normalizeFieldValue(rawValue, featureKind) {
    if (rawValue === "" || rawValue === null || rawValue === undefined) {
      return null;
    }
    if (featureKind === "binary") {
      return Number(rawValue) === 1 ? 1 : 0;
    }
    if (featureKind === "int") {
      return parseInt(rawValue, 10);
    }
    return parseFloat(rawValue);
  }

  function formatProfileSegment(fieldName, value, featureKind) {
    if (featureKind === "binary") {
      return `${fieldName} ${Number(value) === 1 ? "yes" : "no"}`;
    }
    return `${fieldName} ${value}`;
  }

  function buildStructuredProfileMessage() {
    const uiReview = currentSession().ui_review || {};
    const fields = Array.isArray(uiReview.profile_fields) ? uiReview.profile_fields : [];
    const mergedValues = {};
    fields.forEach((field) => {
      if (field.value !== null && field.value !== undefined && field.value !== "") {
        mergedValues[field.field_name] = field.value;
      }
    });
    document.querySelectorAll(".profile-field-input").forEach((inputEl) => {
      const fieldName = inputEl.dataset.profileField;
      const featureKind = inputEl.dataset.featureKind || "float";
      const normalized = normalizeFieldValue(inputEl.value, featureKind);
      if (normalized !== null && !Number.isNaN(normalized)) {
        mergedValues[fieldName] = normalized;
      }
    });
    const segments = [];
    fields.forEach((field) => {
      if (!Object.prototype.hasOwnProperty.call(mergedValues, field.field_name)) {
        return;
      }
      segments.push(formatProfileSegment(field.field_name, mergedValues[field.field_name], field.feature_kind));
    });
    return segments.length > 0 ? `${segments.join(", ")}.` : "";
  }

  function numericBoundSentence(fieldName, minimum, maximum) {
    if (minimum !== "" && maximum !== "") {
      return `Keep ${fieldName} between ${minimum} and ${maximum}.`;
    }
    if (minimum !== "") {
      return `Keep ${fieldName} at least ${minimum}.`;
    }
    if (maximum !== "") {
      return `Keep ${fieldName} at most ${maximum}.`;
    }
    return "";
  }

  function buildStructuredRefinementMessage() {
    const active = currentSession().active_constraint_spec || {};
    const currentBlocked = new Set((active.disallowed_changes || []).map(String));
    const nextBlocked = new Set();

    document.querySelectorAll("[data-blocked-field]").forEach((button) => {
      if (button.getAttribute("aria-pressed") === "true") {
        nextBlocked.add(button.dataset.blockedField);
      }
    });

    const sentences = [];
    FIELD_ORDER.forEach((fieldName) => {
      const wasBlocked = currentBlocked.has(fieldName);
      const isBlocked = nextBlocked.has(fieldName);
      if (isBlocked && !wasBlocked) {
        sentences.push(`Do not change ${fieldName}.`);
      } else if (!isBlocked && wasBlocked) {
        sentences.push(`Allow ${fieldName} to change.`);
      }
    });

    const activeBounds = active.numeric_bounds || {};
    ["Income", "CCAvg", "Mortgage"].forEach((fieldName) => {
      const currentBounds = activeBounds[fieldName] || {};
      const minEl = document.querySelector(`[data-bound-field="${fieldName}"][data-bound-key="min"]`);
      const maxEl = document.querySelector(`[data-bound-field="${fieldName}"][data-bound-key="max"]`);
      const minValue = minEl ? minEl.value.trim() : "";
      const maxValue = maxEl ? maxEl.value.trim() : "";
      const currentMin = currentBounds.min === undefined ? "" : String(currentBounds.min);
      const currentMax = currentBounds.max === undefined ? "" : String(currentBounds.max);
      if (minValue === "" && maxValue === "" && (currentMin !== "" || currentMax !== "")) {
        sentences.push(`Remove the bound on ${fieldName}.`);
        return;
      }
      if (minValue !== currentMin || maxValue !== currentMax) {
        const sentence = numericBoundSentence(fieldName, minValue, maxValue);
        if (sentence) {
          sentences.push(sentence);
        }
      }
    });

    const currentMaxChanged = active.max_changed_features === undefined ? "" : String(active.max_changed_features);
    const maxChangedEl = document.getElementById("max-changed-features");
    const nextMaxChanged = maxChangedEl ? maxChangedEl.value : "";
    if (nextMaxChanged !== currentMaxChanged) {
      if (nextMaxChanged === "") {
        sentences.push("Remove the maximum changed feature limit.");
      } else {
        const plural = nextMaxChanged === "1" ? "" : "s";
        sentences.push(`Allow at most ${nextMaxChanged} changed feature${plural}.`);
      }
    }

    const currentPrefer = Boolean(active.prefer_fewer_changes);
    const preferFewerEl = document.getElementById("prefer-fewer-changes");
    const nextPrefer = preferFewerEl ? preferFewerEl.checked : false;
    if (nextPrefer !== currentPrefer) {
      sentences.push(nextPrefer ? "Prefer fewer changes." : "Remove the preference for fewer changes.");
    }

    return sentences.join(" ");
  }

  function updateQuickBinarySelection(fieldName, value) {
    const selectEl = document.querySelector(`[data-profile-field="${fieldName}"]`);
    if (selectEl instanceof HTMLSelectElement) {
      selectEl.value = value || "";
    }
  }

  async function fetchSessionDetailJson() {
    const { response, payload } = await UFCEClient.fetchJson(`${apiPrefix}/sessions/${sessionId}`);
    if (!response.ok) {
      throw new Error(payload.detail || `Failed to refresh session state (HTTP ${response.status})`);
    }
    return payload;
  }

  async function refreshFromSessionJson(options = {}) {
    const originatedFromUserAction = Boolean(options.originatedFromUserAction);
    const shouldAutoScroll = originatedFromUserAction || isNearBottom(dom.chatTranscript);
    const sessionDetail = await fetchSessionDetailJson();
    rebuildDerivedPageData(sessionDetail, currentPageData.latestTurn || null);
    renderPageState();
    const clarificationState = currentPageData.pageState === "clarification" || currentPageData.pageState === "refinement_clarification";
    if (clarificationState && dom.composerInput && !dom.composerInput.disabled) {
      dom.composerInput.focus();
    }
    if (shouldAutoScroll) {
      scrollTranscriptToBottom();
    }
  }

  async function submitComposer(overrideText) {
    clearError();
    const submitTarget = currentSubmitTarget();
    const composerInput = dom.composerInput;
    const composerSubmit = dom.composerSubmit;
    if (!submitTarget || !composerInput || !composerSubmit) {
      return;
    }

    const rawValue = overrideText == null ? composerInput.value.trim() : String(overrideText).trim();
    if (!rawValue || composerSubmit.disabled) {
      return;
    }

    const pendingTurn = createPendingTurn(rawValue, submitTarget);
    pendingClientTurns.push(pendingTurn);
    renderPendingTurns();

    if (overrideText == null) {
      composerInput.value = "";
    }

    isComposerBusy = true;
    applyComposerContext();
    scrollTranscriptToBottom();

    const endpoint = submitTarget === "refinements"
      ? `${apiPrefix}/sessions/${sessionId}/refinements`
      : `${apiPrefix}/sessions/${sessionId}/messages`;
    const requestBody = submitTarget === "refinements"
      ? { user_feedback: rawValue }
      : { user_input: rawValue };

    try {
      const { response, payload } = await UFCEClient.fetchJson(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      if (!response.ok) {
        markPendingTurnFailed(pendingTurn.localId, payload.detail || "Request failed.");
        if (response.status === 409) {
          try {
            await refreshFromSessionJson();
          } catch (_error) {
            // Keep the failed inline state even if refresh also fails.
          }
        }
        if (overrideText == null && composerInput && !composerInput.value.trim()) {
          composerInput.value = rawValue;
        }
        showError(payload.detail || "Request failed.");
        return;
      }

      removePendingTurn(pendingTurn.localId);
      currentPageData.turns = [payload, ...currentPageData.turns.filter((turn) => turn.turn_id !== payload.turn_id)];
      currentPageData.latestTurn = payload;
      try {
        await refreshFromSessionJson({ originatedFromUserAction: true });
      } catch (error) {
        rebuildDerivedPageData(currentSession(), payload);
        renderPageState();
        scrollTranscriptToBottom();
        showError(error instanceof Error ? error.message : "Session state refresh failed.");
      }
    } catch (error) {
      markPendingTurnFailed(
        pendingTurn.localId,
        error instanceof Error ? error.message : "Request failed."
      );
      if (overrideText == null && composerInput && !composerInput.value.trim()) {
        composerInput.value = rawValue;
      }
      showError(error instanceof Error ? error.message : "Request failed.");
    } finally {
      isComposerBusy = false;
      bindElements();
      applyComposerContext();
    }
  }

  async function submitStructuredProfileEdits() {
    const generated = buildStructuredProfileMessage();
    if (!generated) {
      showError("Fill at least one structured field before sending review edits.");
      return;
    }
    await submitComposer(generated);
  }

  async function submitStructuredRefinement() {
    const generated = buildStructuredRefinementMessage();
    if (!generated) {
      showError("Adjust at least one structured refinement control before sending.");
      return;
    }
    await submitComposer(generated);
  }

  async function createSession() {
    clearError();
    const { response, payload } = await UFCEClient.fetchJson(`${apiPrefix}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_key: currentSession().dataset_key || "bank" }),
    });
    if (!response.ok) {
      showError(payload.detail || "Failed to start a new case.");
      return;
    }
    window.location.href = `/sessions/${payload.session_id}`;
  }

  async function archiveSession() {
    clearError();
    if (!window.confirm("Archive this session and make it read-only?")) {
      return;
    }
    try {
      const { response, payload } = await UFCEClient.fetchJson(`${apiPrefix}/sessions/${sessionId}/archive`, { method: "POST" });
      if (!response.ok) {
        showError(payload.detail || "Failed to archive session.");
        return;
      }
      window.location.href = "/";
    } catch (error) {
      showError(error instanceof Error ? error.message : "Failed to archive session.");
    }
  }

  async function openPreview(previewUrl, fileName) {
    clearError();
    try {
      const { response, payload } = await UFCEClient.fetchJson(previewUrl);
      if (!response.ok) {
        showError(payload.detail || "Preview failed.");
        return;
      }
      if (!dom.previewModal || !dom.previewTitle || !dom.previewContent) {
        return;
      }
      dom.previewTitle.textContent = fileName;
      dom.previewContent.textContent = payload.content;
      dom.previewModal.hidden = false;
    } catch (error) {
      showError(error instanceof Error ? error.message : "Preview failed.");
    }
  }

  function closePreview() {
    if (!dom.previewModal || !dom.previewTitle || !dom.previewContent) {
      return;
    }
    dom.previewModal.hidden = true;
    dom.previewTitle.textContent = "Artifact Preview";
    dom.previewContent.textContent = "";
  }

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    if (target.id === "composer-submit") {
      submitComposer(null);
      return;
    }
    if (target.id === "apply-profile-edits") {
      submitStructuredProfileEdits();
      return;
    }
    if (target.id === "apply-structured-refinement") {
      submitStructuredRefinement();
      return;
    }
    if (target.classList.contains("quick-binary")) {
      updateQuickBinarySelection(target.dataset.targetField, target.dataset.targetValue);
      return;
    }
    if (target.hasAttribute("data-blocked-field")) {
      const pressed = target.getAttribute("aria-pressed") === "true";
      target.setAttribute("aria-pressed", pressed ? "false" : "true");
      target.classList.toggle("is-selected", !pressed);
      return;
    }
    if (target.classList.contains("preview-button")) {
      openPreview(target.dataset.previewUrl, target.dataset.previewName || "Artifact Preview");
      return;
    }
    if (target.hasAttribute("data-start-new-case") || target.id === "new-case") {
      createSession();
      return;
    }
    if (target.id === "close-session") {
      archiveSession();
      return;
    }
    if (target.id === "preview-close") {
      closePreview();
      return;
    }
    if (target.hasAttribute("data-session-tab")) {
      setActiveTab(target.getAttribute("data-session-tab"));
    }
  });

  document.addEventListener("keydown", (event) => {
    if (!(event.target instanceof HTMLElement)) {
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && event.target.id === "composer-input") {
      submitComposer(null);
    }
    if (event.key === "Escape" && !dom.previewModal?.hidden) {
      closePreview();
    }
  });

  document.addEventListener("toggle", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLDetailsElement)) {
      return;
    }

    if (target.classList.contains("chat-inline-detail")) {
      if (target.open) {
        disclosureState.inlineDetailOpenId = target.dataset.inlineDetailId || null;
        document.querySelectorAll(".chat-inline-detail").forEach((detailsEl) => {
          if (detailsEl !== target) {
            detailsEl.open = false;
          }
        });
      } else if (disclosureState.inlineDetailOpenId === target.dataset.inlineDetailId) {
        disclosureState.inlineDetailOpenId = null;
      }
      syncInlineDetailLabels();
      return;
    }

    if (target.classList.contains("context-section")) {
      if (target.id === "advanced-refinement-controls") {
        disclosureState.advancedControlsOpen = target.open;
      }
      if (target.open) {
        disclosureState.stateRailOpenId = target.id;
        if (isDesktopLayout()) {
          document.querySelectorAll('.context-section').forEach((detailsEl) => {
            if (detailsEl !== target) {
              detailsEl.open = false;
            }
          });
        }
      } else if (disclosureState.stateRailOpenId === target.id) {
        disclosureState.stateRailOpenId = null;
      }
      syncContextSectionLabels();
    }
  });

  document.addEventListener("click", (event) => {
    if (event.target === dom.previewModal) {
      closePreview();
    }
  });

  window.addEventListener("resize", () => {
    if (isDesktopLayout()) {
      setActiveTab("chat");
    }
  });

  bindElements();
  rebuildDerivedPageData(currentSession(), currentPageData.latestTurn || null);
  renderPageState();
  if (isDesktopLayout()) {
    scrollTranscriptToBottom();
  }
})();
