const status = document.querySelector('#ranking-status');
const content = document.querySelector('#ranking-content');
const rows = document.querySelector('#ranking-rows');
const percent = value => `${(value * 100).toFixed(1)}%`;

fetch('/api/ranking').then(response => response.json()).then(data => {
  if (!data.candidates.length) {
    status.textContent = 'No screening results yet. Upload resumes and run screening first.';
    return;
  }
  data.candidates.forEach((candidate, index) => {
    const row = document.createElement('tr');
    row.innerHTML = `<td>#${candidate.rank || index + 1}</td><td></td><td></td><td>${percent(candidate.match_score)}</td><td>${percent(candidate.skill_score)}</td><td>${candidate.project_score === null ? 'N/A' : percent(candidate.project_score)}</td><td>${percent(candidate.semantic_similarity)}</td><td>${percent(candidate.critical_coverage)}</td><td></td>`;
    row.children[1].textContent = candidate.candidate;
    row.children[2].textContent = candidate.suggested_role;
    row.children[8].textContent = candidate.recommendation;
    rows.append(row);
  });
  status.hidden = true;
  content.hidden = false;
}).catch(() => { status.textContent = 'Could not load the ranking. Run a screening first.'; });
