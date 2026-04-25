(function () {
  const UFCEClient = window.UFCEClient || (window.UFCEClient = {});

  UFCEClient.escapeHtml = function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  };

  UFCEClient.readJsonScript = function readJsonScript(id) {
    const element = document.getElementById(id);
    if (!element) {
      return null;
    }
    try {
      return JSON.parse(element.textContent || "null");
    } catch (_error) {
      return null;
    }
  };
})();
