const form = document.querySelector('#analysis-form');
const fileInput = document.querySelector('#resumes');
const csvInput = document.querySelector('#resume-csv');
const csvName = document.querySelector('#csv-name');
const selectedFiles = document.querySelector('#selected-files');
const fileActions = document.querySelector('#file-actions');
const fileCount = document.querySelector('#file-count');
const clearFiles = document.querySelector('#clear-files');
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const cards = document.querySelector('#cards');
const template = document.querySelector('#card-template');
const detailsPanel = document.querySelector('#details-panel');
const detailsBackdrop = document.querySelector('#details-backdrop');
const detailsName = document.querySelector('#details-name');
const detailsContact = document.querySelector('#details-contact');
const detailsSummary = document.querySelector('#details-summary');
const detailsExplanation = document.querySelector('#details-explanation');
const detailsSkills = document.querySelector('#details-skills');
const detailsProjectSummary = document.querySelector('#details-project-summary');
const detailsProjects = document.querySelector('#details-projects');
const detailsKeywords = document.querySelector('#details-keywords');
const detailsMissing = document.querySelector('#details-missing');
const detailsResume = document.querySelector('#details-resume');
const percent = value => `${(value * 100).toFixed(1)}%`;

let chosenFiles = [];

function syncInputFiles() {
  const transfer = new DataTransfer();
  chosenFiles.forEach(file => transfer.items.add(file));
  fileInput.files = transfer.files;
}

function showSelectedFiles() {
  selectedFiles.replaceChildren();
  chosenFiles.forEach((file, index) => {
    const item = document.createElement('li');
    const name = document.createElement('span');
    name.textContent = file.name;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'remove-file';
    remove.textContent = 'Remove';
    remove.addEventListener('click', () => {
      chosenFiles.splice(index, 1);
      syncInputFiles();
      showSelectedFiles();
    });
    item.append(name, remove);
    selectedFiles.append(item);
  });
  const hasFiles = chosenFiles.length > 0;
  selectedFiles.hidden = !hasFiles;
  fileActions.hidden = !hasFiles;
  fileCount.textContent = `${chosenFiles.length} resume${chosenFiles.length === 1 ? '' : 's'} selected`;
}

fileInput.addEventListener('change', () => {
  const newFiles = [...fileInput.files];
  newFiles.forEach(file => {
    if (!chosenFiles.some(existing => existing.name === file.name && existing.size === file.size)) chosenFiles.push(file);
  });
  syncInputFiles();
  showSelectedFiles();
});

clearFiles.addEventListener('click', () => {
  chosenFiles = [];
  syncInputFiles();
  showSelectedFiles();
});

csvInput.addEventListener('change', () => {
  csvName.textContent = csvInput.files.length ? `CSV selected: ${csvInput.files[0].name}` : '';
});

function closeDetails() {
  detailsPanel.hidden = true;
  detailsBackdrop.hidden = true;
}

document.querySelector('#close-details').addEventListener('click', closeDetails);
detailsBackdrop.addEventListener('click', closeDetails);

async function openDetails(detailId) {
  detailsPanel.hidden = false;
  detailsBackdrop.hidden = false;
  detailsName.textContent = 'Loading resume details…';
  detailsContact.textContent = '';
  detailsSummary.textContent = '';
  detailsExplanation.textContent = '';
  detailsSkills.textContent = '';
  detailsProjectSummary.textContent = '';
  detailsProjects.replaceChildren();
  detailsKeywords.textContent = '';
  detailsMissing.textContent = '';
  detailsResume.textContent = '';
  try {
    const response = await fetch(`/api/resume-details/${detailId}`);
    const details = await response.json();
    if (!response.ok) throw new Error(details.error || 'Could not load resume details.');
    detailsName.textContent = details.candidate;
    detailsContact.textContent = details.email || 'Email not found in the uploaded resume.';
    const match = details.matching;
    const projectText = match.project_score === null ? 'Project relevance: N/A (no projects detected; available components were reweighted).'
      : `Project relevance: ${percent(details.project_score)}.`;
    const criticalCoverage = match.critical_coverage === null ? 'N/A (no critical requirements)' : percent(match.critical_coverage);
    detailsSummary.textContent = `Suggested role: ${details.suggested_role}. Overall match: ${percent(details.match_score)}. Skill capability match: ${percent(match.skill_score)}. ${projectText} Semantic similarity: ${percent(match.semantic_similarity)}. Critical requirement coverage: ${criticalCoverage}. Evidence strength: ${percent(match.evidence_strength)}.`;
    const matched = match.matched_skills.length ? match.matched_skills.join(', ') : 'no direct required skills';
    const strongest = details.project_analysis[0];
    const missingCritical = match.requirements
      .filter(requirement => requirement.critical && !requirement.match_strength)
      .map(requirement => requirement.skill);
    const criticalNote = missingCritical.length ? ` Missing critical requirement: ${missingCritical.join(', ')}.` : '';
    detailsExplanation.textContent = strongest
      ? `Strongest project: ${strongest.title} (${percent(strongest.relevance_score)}). The strongest direct evidence is ${matched}.${criticalNote}`
      : `The strongest direct evidence is ${matched}.${criticalNote} Add project details to demonstrate these skills.`;
    const partial = match.related_skills.length
      ? match.related_skills.map(item => `${item.resume_skills.join(', ')} → ${item.required_skill}`).join('; ')
      : 'None';
    detailsSkills.textContent = `Matched skills: ${matched}. Strongly/partially related skills: ${partial}. Missing skills: ${match.missing_skills.length ? match.missing_skills.join(', ') : 'None'}.`;
    if (!details.project_analysis.length) {
      detailsProjectSummary.textContent = 'No project section was detected in this resume.';
    } else {
      detailsProjectSummary.textContent = `Most relevant project: ${strongest.title} — ${percent(strongest.relevance_score)}.`;
      details.project_analysis.forEach(project => {
        const row = document.createElement('tr');
        [project.title, project.relevant_skills.length ? project.relevant_skills.join(', ') : 'No direct match',
          percent(project.relevance_score), project.reason].forEach(value => {
          const cell = document.createElement('td');
          cell.textContent = value;
          row.append(cell);
        });
        detailsProjects.append(row);
      });
    }
    detailsKeywords.textContent = details.shared_keywords.length
      ? `The job description and resume both mention: ${details.shared_keywords.join(', ')}.`
      : 'No important job keywords were found in both documents.';
    detailsMissing.textContent = details.missing_job_keywords.length
      ? `Important job terms not found in this resume: ${details.missing_job_keywords.join(', ')}.`
      : 'The important job terms found by the model are present in this resume.';
    detailsResume.textContent = details.resume_text;
  } catch (error) { detailsName.textContent = error.message; }
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!chosenFiles.length && !csvInput.files.length) {
    status.textContent = 'Please add resume files or one CSV file.';
    return;
  }
  status.textContent = csvInput.files.length ? 'Reading the CSV and screening resumes…' : `Screening ${chosenFiles.length} resume(s)…`;
  results.hidden = true;
  const button = form.querySelector('button');
  button.disabled = true;
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: new FormData(form) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    cards.replaceChildren();
    data.candidates.forEach(candidate => {
      const node = template.content.cloneNode(true);
      node.querySelector('h3').textContent = candidate.candidate;
      node.querySelector('.rank').textContent = `Rank #${candidate.rank} · ${candidate.matching.recommendation}`;
      node.querySelector('.category').textContent = `Suggested role: ${candidate.suggested_role}`;
      const projectText = candidate.matching.project_score === null ? 'No projects detected.'
        : `Project relevance: ${percent(candidate.matching.project_score)}.`;
      const criticalCoverage = candidate.matching.critical_coverage === null ? 'N/A' : percent(candidate.matching.critical_coverage);
      node.querySelector('.similarity').textContent = `Overall match: ${percent(candidate.matching.overall_score)} · Skill capability: ${percent(candidate.matching.skill_score)} · ${projectText} Semantic similarity: ${percent(candidate.matching.semantic_similarity)} · Critical coverage: ${criticalCoverage}.`;
      const matchedSkills = candidate.matching.matched_skills;
      const missingCritical = candidate.matching.requirements
        .filter(requirement => requirement.critical && !requirement.match_strength)
        .map(requirement => requirement.skill);
      const evidenceSummary = matchedSkills.length
        ? `demonstrates ${matchedSkills.join(', ')}`
        : 'does not demonstrate direct required skills';
      const criticalSummary = missingCritical.length ? ` Missing critical requirement: ${missingCritical.join(', ')}.` : '';
      node.querySelector('.relevance').textContent = `${candidate.matching.recommendation}: ${evidenceSummary}.${criticalSummary}`;
      node.querySelector('.details-button').addEventListener('click', () => openDetails(candidate.detail_id));
      cards.append(node);
    });
    results.hidden = false;
    const failures = data.failed_resumes || [];
    status.textContent = `${data.recommended_count} of ${data.candidates.length} resume(s) are recommended.`
      + (failures.length ? ` ${failures.length} resume(s) could not be analyzed: ${failures.map(item => item.candidate).join(', ')}.` : '');
  } catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; }
});
