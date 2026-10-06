document.querySelector('#copy-bibtex').addEventListener('click', async () => {
  const code = document.querySelector('#citation-text');
  const status = document.querySelector('#citation-status');
  try {
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = 'Copied BibTeX.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(code);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or Command+C to copy.';
  }
});
