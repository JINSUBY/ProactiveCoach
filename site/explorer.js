/* Ground-truth preview explorer. No video or model output is synthesized. */
(() => {
  'use strict';
  const levels = ['phase', 'step', 'action'];
  const $ = id => document.getElementById(id);
  const title = s => s[0].toUpperCase() + s.slice(1);
  const sec = n => `${Number(Number(n).toFixed(3))} s`;
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  let cases = [], record, level = 'step', cursor = 0, selected = null;
  let mediaCatalog = {}, mediaAvailable = false;
  const video = $('example-video');
  let playing = false, frame = null, previousFrame = null, clock = 0;
  const eventTime = u => u.guide_time === null ? u.start : u.guide_time;
  const stable = units => [...units].sort((a, b) => eventTime(a) - eventTime(b) || a.index - b.index);
  const phaseIndex = (lv, u) => lv === 'phase' ? u.index : lv === 'step' ? u.parent : record.units.step.find(s => s.index === u.parent)?.parent;
  const clipStart = () => mediaCatalog[record.id].excerpt_start;
  const clipEnd = () => mediaCatalog[record.id].excerpt_end;
  const clipDuration = () => clipEnd() - clipStart();
  const localTime = t => t - clipStart();
  const inExcerpt = u => (u.end > clipStart() && u.start < clipEnd()) || (u.guide_time !== null && u.guide_time >= clipStart() && u.guide_time < clipEnd());
  const scopeUnits = lv => record.units[lv].filter(u => $('phase-filter').value === 'all' || phaseIndex(lv, u) === Number($('phase-filter').value));
  const subset = lv => scopeUnits(lv).filter(inExcerpt);
  function stop() {
    playing = false; previousFrame = null; cancelAnimationFrame(frame);
    if (!video.paused) video.pause();
    $('annotation-play').textContent = mediaAvailable ? 'Play video' : 'Play annotations';
    $('annotation-play').setAttribute('aria-pressed', 'false');
  }
  function choose(lv, unit) {
    stop(); level = lv; selected = {level: lv, index: unit.index};
    renderLevels(); renderList(); renderInspector(); seek(eventTime(unit));
    $('event-status').textContent = `Selected ${lv} ${unit.number}. ${unit.guide_time === null ? 'No guidance; showing event start' : 'Guidance at'} ${sec(eventTime(unit))} in the original review clip.${eventTime(unit) < clipStart() ? ' Earlier context: playback stays at the viewing-window start.' : ''}`;
  }
  function renderLevels() {
    document.querySelectorAll('[data-explorer-level]').forEach(button => {
      const active = button.dataset.explorerLevel === level;
      button.setAttribute('aria-pressed', String(active));
    });
  }
  function renderInspector() {
    const target = $('event-inspector'); target.replaceChildren();
    if (!selected) { target.append(el('p', 'note', 'Select an event or guidance marker to view its annotation.')); return; }
    const u = record.units[selected.level].find(x => x.index === selected.index);
    target.append(el('span', `unit-tag ${selected.level}`, `${title(selected.level)} ${u.number}`));
    target.append(el('h4', '', u.text));
    const timing = el('dl', 'event-facts');
    [['Original event span', `${sec(u.start)} – ${sec(u.end)}`], ['Original guidance timestamp', u.guide_time === null ? 'None recorded' : sec(u.guide_time)], ['Parent', u.parent === null ? 'Task' : `${title(levels[levels.indexOf(selected.level) - 1])} ${record.units[levels[levels.indexOf(selected.level) - 1]].find(p => p.index === u.parent)?.number ?? u.parent}`]].forEach(([k, v]) => { timing.append(el('dt', '', k), el('dd', '', v)); });
    target.append(timing, el('blockquote', '', u.guide === null ? 'No guidance annotation for this event.' : u.guide));
    target.append(el('p', 'note', 'Guidance may precede the event.'));
  }
  function renderList() {
    const list = $('annotation-list'); list.replaceChildren();
    const units = stable(subset(level));
    $('event-count').textContent = `${units.length} ${level} events · ordered by guidance time (event start when no guide)`;
    units.forEach(u => {
      const button = el('button', 'event-card'); button.type = 'button';
      button.dataset.unitIndex = u.index;
      button.setAttribute('aria-pressed', String(selected?.level === level && selected.index === u.index));
      const stamp = el('span', 'event-stamp', u.guide_time === null ? 'No guide' : u.guide_time < clipStart() ? 'Earlier context' : sec(localTime(u.guide_time)));
      const body = el('span', 'event-copy');
      body.append(el('strong', '', `${u.number} · ${u.text}`), el('span', '', u.guide === null ? 'No guidance annotation.' : u.guide), el('small', '', `Original event: ${sec(u.start)} – ${sec(u.end)}`));
      button.append(stamp, body); button.addEventListener('click', () => choose(level, u)); list.append(button);
    });
  }
  function renderTimeline() {
    const host = $('timeline-lanes'); host.replaceChildren();
    const duration = clipDuration();
    levels.forEach(lv => {
      const row = el('div', `timeline-row ${lv}`), label = el('span', 'track-name', title(lv));
      const track = el('div', 'track'); track.setAttribute('aria-label', `${title(lv)} annotated events`);
      const units = [...subset(lv)].sort((a, b) => a.start - b.start || a.index - b.index);
      const occupied = [];
      units.forEach(u => {
        let lane = occupied.findIndex(end => end <= u.start);
        if (lane < 0) lane = occupied.length;
        occupied[lane] = u.end;
        const band = el('button', 'event-band', u.number); band.type = 'button';
        const visibleStart = Math.max(u.start, clipStart()), visibleEnd = Math.min(u.end, clipEnd());
        band.style.left = `${localTime(visibleStart) / duration * 100}%`;
        band.style.width = `${Math.max((visibleEnd - visibleStart) / duration * 100, .15)}%`;
        band.style.top = `${lane * 38 + 22}px`;
        band.dataset.level = lv; band.dataset.unitIndex = u.index;
        band.title = `${title(lv)} ${u.number}: ${u.text}. Event ${sec(u.start)} – ${sec(u.end)}`;
        band.setAttribute('aria-label', band.title);
        band.addEventListener('click', () => choose(lv, u)); if (visibleEnd > visibleStart) track.append(band);
        if (u.guide_time !== null && u.guide_time >= clipStart() && u.guide_time < clipEnd()) {
          const marker = el('button', 'guidance-marker'); marker.type = 'button';
          marker.style.left = `${localTime(u.guide_time) / duration * 100}%`; marker.style.top = `${lane * 38}px`;
          marker.dataset.level = lv; marker.dataset.unitIndex = u.index;
          marker.title = `${title(lv)} ${u.number} guidance at ${sec(u.guide_time)}: ${u.guide}`;
          marker.setAttribute('aria-label', marker.title);
          marker.addEventListener('click', () => choose(lv, u)); track.append(marker);
        }
      });
      track.style.height = `${Math.max(1, occupied.length) * 38 + 12}px`;
      track.append(el('span', 'track-playhead'));
      if (!units.length) track.append(el('span', 'note', 'No events in this phase'));
      row.append(label, track); host.append(row);
    });
    $('timeline-ticks').replaceChildren(...[0, .25, .5, .75, 1].map(f => el('span', '', sec(Number((duration * f).toFixed(3))))));
  }
  function renderGuidance() {
    levels.forEach(lv => {
      const eligible = scopeUnits(lv).filter(u => u.guide !== null && u.guide_time !== null && u.guide_time <= cursor);
      const last = eligible.length ? Math.max(...eligible.map(u => u.guide_time)) : null;
      const rows = last === null ? [] : eligible.filter(u => u.guide_time === last).sort((a, b) => a.index - b.index);
      const slot = $(`latest-${lv}`); slot.replaceChildren();
      if (!rows.length) slot.append(el('p', '', 'No guidance annotation at or before this time.'));
      else rows.forEach(u => {
        const b = el('button', 'latest-guide'); b.type = 'button';
        b.append(el('span', 'guide-issued', `${u.number} · ${u.guide_time < clipStart() ? 'Earlier context · original ' + sec(u.guide_time) : 'Playback ' + sec(localTime(u.guide_time)) + ' · original ' + sec(u.guide_time)}`), el('span', '', u.guide));
        b.addEventListener('click', () => choose(lv, u)); slot.append(b);
      });
    });
  }
  function seek(value, fromVideo = false) {
    cursor = Math.max(clipStart(), Math.min(clipEnd(), Number(value)));
    if (mediaAvailable && !fromVideo && video.readyState >= 1) {
      const target = Math.max(0, Math.min(localTime(cursor), video.duration));
      if (Math.abs(video.currentTime - target) > .001) video.currentTime = target;
    }
    $('annotation-seek').value = localTime(cursor);
    $('annotation-seek').setAttribute('aria-valuetext', `${sec(localTime(cursor))} of ${sec(clipDuration())} in video; original time ${sec(cursor)}`);
    $('cursor-time').textContent = sec(localTime(cursor));
    $('source-time').textContent = `Original review time: ${sec(cursor)} · shown from the beginning`;
    document.querySelectorAll('.track-playhead').forEach(p => p.style.left = `${localTime(cursor) / clipDuration() * 100}%`);
    document.querySelectorAll('.event-band').forEach(b => {
      const u = record.units[b.dataset.level].find(u => u.index === Number(b.dataset.unitIndex));
      b.classList.toggle('is-active', u.start <= cursor && cursor < u.end);
      b.classList.toggle('is-selected', selected?.level === b.dataset.level && selected.index === u.index);
    });
    const key = levels.map(lv => scopeUnits(lv).filter(u => u.guide_time !== null && u.guide_time <= cursor).map(u => u.index).join(',')).join('|');
    if (key !== seek.lastKey) { renderGuidance(); seek.lastKey = key; }
    const times = guidanceTimes();
    $('previous-guide').disabled = !times.some(t => t < cursor - .000001);
    $('next-guide').disabled = !times.some(t => t > cursor + .000001);
  }
  function guidanceTimes() { return [...new Set(levels.flatMap(lv => subset(lv).filter(u => u.guide_time !== null && u.guide_time >= clipStart() && u.guide_time < clipEnd()).map(u => u.guide_time)))].sort((a, b) => a - b); }
  function tick(now) {
    if (!playing) return;
    if (previousFrame !== null) clock += (now - previousFrame) / 1000 * Number($('playback-speed').value);
    previousFrame = now; seek(Math.min(clipEnd(), Math.round(clock * 1000) / 1000));
    if (cursor >= clipEnd()) stop(); else frame = requestAnimationFrame(tick);
  }
  function loadCase(id) {
    stop(); record = cases.find(x => x.id === id); selected = null; seek.lastKey = null;
    const media = mediaCatalog[id]; mediaAvailable = Boolean(media);
    $('video-panel').hidden = !mediaAvailable; $('missing-media').hidden = mediaAvailable;
    $('media-error').textContent = '';
    if (media) {
      video.src = media.src; video.poster = media.poster; video.playbackRate = Number($('playback-speed').value);
      $('download-video').href = media.src;
      $('video-caption').textContent = `${id} · ${media.dataset} · ${sec(clipDuration())} · from 0 s · silent video.`;
    } else { video.removeAttribute('src'); video.removeAttribute('poster'); }
    video.load();
    $('annotation-play').textContent = mediaAvailable ? 'Play video' : 'Play annotations';
    $('case-goal').textContent = record.goal; $('case-query').textContent = record.query;
    $('case-info').textContent = `${record.id} · ${record.video.source_dataset} · ${record.video.domain} · ${sec(clipDuration())} · from the beginning · test split`;
    $('media-record').textContent = record.video.record_id;
    $('media-interval').textContent = `Original review interval: ${sec(clipStart())} – ${sec(clipEnd())}`;
    $('media-dataset').textContent = record.video.source_dataset;
    $('clip-duration').textContent = sec(clipDuration());
    $('annotation-seek').max = clipDuration();
    const phase = $('phase-filter'); phase.replaceChildren(new Option('All phases', 'all'));
    record.units.phase.filter(inExcerpt).forEach(u => phase.append(new Option(`${u.number} · ${u.text}`, u.index)));
    renderLevels(); renderTimeline(); renderList(); renderInspector(); seek(clipStart()); $('annotation-list').scrollTop = 0;
    $('explorer-status').textContent = `${cases.length} video examples · ground-truth guidance`;
  }
  $('example-select').addEventListener('change', e => loadCase(e.target.value));
  $('phase-filter').addEventListener('change', () => { stop(); selected = null; seek.lastKey = null; renderTimeline(); renderList(); renderInspector(); seek(cursor); });
  document.querySelectorAll('[data-explorer-level]').forEach(b => b.addEventListener('click', () => { level = b.dataset.explorerLevel; selected = null; renderLevels(); renderList(); renderInspector(); seek(cursor); }));
  $('annotation-seek').addEventListener('input', e => { stop(); seek(clipStart() + Number(e.target.value)); });
  $('annotation-play').addEventListener('click', () => {
    if (!record) return;
    if (mediaAvailable) {
      if (!video.paused) { stop(); return; }
      if (video.ended) seek(clipStart());
      video.play().catch(() => { $('media-error').textContent = 'Video could not play. Try the native controls or download the clip.'; });
      return;
    }
    if (playing) { stop(); return; }
    if (cursor >= clipEnd()) seek(clipStart());
    playing = true; clock = cursor; previousFrame = null;
    $('annotation-play').textContent = 'Pause annotations'; $('annotation-play').setAttribute('aria-pressed', 'true'); frame = requestAnimationFrame(tick);
  });
  $('previous-guide').addEventListener('click', () => { stop(); const t = guidanceTimes().filter(t => t < cursor - .000001); if (t.length) seek(t[t.length - 1]); });
  $('next-guide').addEventListener('click', () => { stop(); const t = guidanceTimes().find(t => t > cursor + .000001); if (t !== undefined) seek(t); });
  $('download-case').addEventListener('click', () => {
    if (!record) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], {type: 'application/json'}));
    const link = el('a'); link.href = url; link.download = `${record.id}-ground-truth.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  video.addEventListener('loadedmetadata', () => { if (record && mediaAvailable) video.currentTime = Math.max(0, Math.min(localTime(cursor), video.duration)); });
  video.addEventListener('timeupdate', () => { if (record && mediaAvailable) seek(clipStart() + Math.round(video.currentTime * 1000) / 1000, true); });
  video.addEventListener('seeked', () => { if (record && mediaAvailable) seek(clipStart() + Math.round(video.currentTime * 1000) / 1000, true); });
  video.addEventListener('play', () => { $('annotation-play').textContent = 'Pause video'; $('annotation-play').setAttribute('aria-pressed', 'true'); });
  video.addEventListener('pause', () => { $('annotation-play').textContent = mediaAvailable ? 'Play video' : 'Play annotations'; $('annotation-play').setAttribute('aria-pressed', 'false'); });
  video.addEventListener('error', () => { if (mediaAvailable && video.getAttribute('src')) { mediaAvailable = false; $('annotation-play').textContent = 'Play annotations'; $('media-error').textContent = 'Video unavailable. Annotation-only playback remains available; reload to retry video.'; } });
  $('playback-speed').addEventListener('change', () => { video.playbackRate = Number($('playback-speed').value); });
  const readJSON = url => fetch(url).then(r => { if (!r.ok) throw Error('Preview unavailable'); return r.json(); });
  Promise.all([readJSON('examples.json'), readJSON('media.json'), readJSON('featured-examples.json')]).then(([data, media, featured]) => {
    mediaCatalog = media;
    cases = featured.map(id => data.find(c => c.id === id));
    if (cases.length !== 5 || cases.some(c => !c || !media[c.id]) || new Set(featured).size !== 5) throw Error('Invalid featured examples');
    const select = $('example-select'); select.replaceChildren();
    cases.forEach(x => select.append(new Option(`${x.id} · ${x.goal}`, x.id)));
    document.querySelectorAll('#examples [disabled]').forEach(b => b.disabled = false);
    const first = cases.find(x => mediaCatalog[x.id]) || cases[0];
    select.value = first.id; loadCase(first.id);
  }).catch(() => { $('explorer-status').textContent = 'Examples could not load. Reload or open the dataset preview.'; });
})();
