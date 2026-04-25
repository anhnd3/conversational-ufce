(function () {
  const UFCEClient = window.UFCEClient || (window.UFCEClient = {});

  UFCEClient.fetchJson = async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json();
    return { response, payload };
  };
})();
