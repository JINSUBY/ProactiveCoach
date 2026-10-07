/* Preserve every paper value; switch complete metric groups on narrow screens. */
(() => {
  'use strict';
  const table = document.getElementById('table-2');
  if (!table) return;
  const wrapper = table.closest('.main-results');
  const compact = matchMedia('(max-width: 1023px)');
  const groups = ['Phase', 'Step', 'Action', 'EgoProactive'];
  const header = table.tHead.rows[0];
  const levels = table.tHead.rows[1];
  const metrics = [...table.tHead.rows[2].cells];
  const rows = [...table.querySelectorAll('tbody tr:not(.model-group)')];
  const groupRows = [...table.querySelectorAll('.model-group th')];
  const metricColumns = table.querySelector('colgroup col:last-child');
  const controls = document.createElement('div');
  controls.className = 'results-controls';
  controls.setAttribute('role', 'group');
  controls.setAttribute('aria-label', 'Table 2 benchmark and guidance level');
  let selected = 0;
  const buttons = groups.map((label, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.setAttribute('aria-controls', 'table-2');
    button.addEventListener('click', () => { selected = index; render(); });
    controls.append(button);
    return button;
  });
  wrapper.insertBefore(controls, table);
  function render() {
    const grouped = compact.matches;
    wrapper.classList.toggle('is-grouped', grouped);
    controls.hidden = !grouped;
    buttons.forEach((button, index) => button.setAttribute('aria-pressed', String(index === selected)));
    metrics.forEach((cell, index) => { cell.hidden = grouped && Math.floor(index / 4) !== selected; });
    rows.forEach(row => [...row.cells].slice(2).forEach((cell, index) => {
      cell.hidden = grouped && Math.floor(index / 4) !== selected;
    }));
    const ego = grouped && selected === 3;
    header.cells[0].rowSpan = header.cells[1].rowSpan = ego ? 2 : 3;
    header.cells[2].hidden = ego;
    header.cells[2].colSpan = grouped ? 4 : 12;
    header.cells[3].hidden = grouped && !ego;
    header.cells[3].rowSpan = ego ? 1 : 2;
    levels.hidden = ego;
    [...levels.cells].forEach((cell, index) => { cell.hidden = grouped && index !== selected; });
    metricColumns.span = grouped ? 4 : 16;
    groupRows.forEach(cell => { cell.colSpan = grouped ? 6 : 18; });
    wrapper.scrollLeft = 0;
  }
  compact.addEventListener('change', render);
  render();
})();
