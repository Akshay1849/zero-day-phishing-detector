document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const input = document.getElementById('url-input');
  const analyzeButton = document.getElementById('analyze-button');
  const buttonLabel = document.getElementById('button-label');
  const buttonLoading = document.getElementById('button-loading');
  const resultCard = document.getElementById('result-card');
  const errorCard = document.getElementById('error-card');
  const resultTitle = document.getElementById('result-title');
  const resultConfidence = document.getElementById('result-confidence');
  const resultRisk = document.getElementById('result-risk');
  const resultModelLabel = document.getElementById('result-model');
  const indicatorList = document.getElementById('indicator-list');
  const clearButton = document.getElementById('clear-button');

  function setLoading(isLoading) {
    analyzeButton.disabled = isLoading;
    buttonLabel.hidden = isLoading;
    buttonLoading.hidden = !isLoading;
  }

  function showError(message) {
    errorCard.textContent = message;
    errorCard.hidden = false;
    resultCard.hidden = true;
    clearButton.hidden = false;
  }

  function showResult(payload) {
    const phishing = payload.label === 'Phishing';
    resultTitle.textContent = phishing ? 'PHISHING URL' : 'LEGITIMATE URL';
    resultTitle.className = phishing ? 'phishing' : 'legitimate';

    const badge = document.getElementById('result-badge');
    badge.textContent = phishing ? '⚠' : '✓';
    badge.className = 'result-badge ' + (phishing ? 'phishing' : 'legitimate');

    resultConfidence.textContent = `Confidence: ${(payload.confidence * 100).toFixed(2)}%`;

    // Show phishing / legitimate probabilities for hybrid, risk level for RF
    if (payload.phishing_probability !== undefined) {
      resultRisk.textContent =
        `Phishing prob: ${(payload.phishing_probability * 100).toFixed(1)}% · ` +
        `Legitimate prob: ${(payload.legitimate_probability * 100).toFixed(1)}%`;
    } else if (payload.risk_level) {
      resultRisk.textContent = `Risk level: ${payload.risk_level}`;
    } else {
      resultRisk.textContent = '';
    }

    resultModelLabel.textContent = `Model: ${payload.model}`;

    indicatorList.replaceChildren();
    if (Array.isArray(payload.indicators)) {
      payload.indicators.forEach((indicator) => {
        const item = document.createElement('li');
        item.className = indicator.status;
        item.textContent = `${indicator.status === 'warn' ? '⚠' : '✓'} ${indicator.text}`;
        indicatorList.appendChild(item);
      });
    }
    errorCard.hidden = true;
    resultCard.hidden = false;
    clearButton.hidden = false;
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const url = input.value.trim();
    if (!url) {
      showError('Enter a URL.');
      return;
    }

    // Determine which model/endpoint to use
    const modelValue = document.querySelector('input[name="model"]:checked')?.value || 'rf';
    const selection = modelValue === 'hybrid'
      ? {endpoint: '/api/predict-url-hybrid', model: 'Hybrid RF + SVM'}
      : {endpoint: '/api/predict-url', model: 'Random Forest'};

    setLoading(true);
    clearButton.hidden = true;
    resultCard.hidden = true;
    errorCard.hidden = true;

    try {
      const response = await fetch(selection.endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url})
      });
      const payload = await response.json();
      if (!response.ok || !payload.success) {
        throw new Error(payload.error || 'The request was rejected.');
      }
      if (payload.model !== selection.model) {
        throw new Error('The selected model response did not match the requested model.');
      }
      showResult(payload);
    } catch (error) {
      showError(error.message);
    } finally {
      setLoading(false);
    }
  });

  clearButton.addEventListener('click', () => {
    input.value = '';
    resultCard.hidden = true;
    errorCard.hidden = true;
    clearButton.hidden = true;
    setLoading(false);
    input.focus();
  });
});
