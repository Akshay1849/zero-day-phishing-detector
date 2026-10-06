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
  const resultProbabilities = document.getElementById('result-probabilities');
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

    resultProbabilities.textContent =
      `Phishing Probability: ${(payload.phishing_probability * 100).toFixed(1)}% · ` +
      `Legitimate Probability: ${(payload.legitimate_probability * 100).toFixed(1)}%`;
    resultRisk.textContent = `Risk Level: ${payload.risk_level}`;

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

    setLoading(true);
    clearButton.hidden = true;
    resultCard.hidden = true;
    errorCard.hidden = true;

    try {
      let response;
      try {
        response = await fetch('/api/predict-url', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({url})
        });
      } catch (error) {
        showError('Unable to connect to the prediction server. Please try again.');
        return;
      }

      let responseText;
      try {
        responseText = await response.text();
      } catch (error) {
        showError('Unable to connect to the prediction server. Please try again.');
        return;
      }

      if (!responseText.trim()) {
        showError('Server returned an empty response. Please try again.');
        return;
      }

      let payload;
      try {
        payload = JSON.parse(responseText);
      } catch (error) {
        showError(
          !response.ok && response.status >= 500
            ? 'The prediction server encountered an error. Please try again.'
            : 'The prediction server returned an invalid response. Please try again.'
        );
        return;
      }

      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        showError(
          !response.ok && response.status >= 500
            ? 'The prediction server encountered an error. Please try again.'
            : 'The prediction server returned an invalid response. Please try again.'
        );
      } else if (!response.ok || !payload.success) {
        showError(payload.error || 'The request was rejected.');
      } else if (payload.model !== 'Hybrid RF + SVM') {
        showError('The prediction server returned an unexpected model.');
      } else {
        showResult(payload);
      }
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
