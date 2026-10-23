// ==========================================================================
// ML PORTFOLIO INTERACTION ENGINE & REAL-TIME INFERENCE CONTROLLERS
// ==========================================================================

// --- TAB SWITCHER ---
function switchDemo(demoId) {
  // Update tabs
  const tabBtns = document.querySelectorAll('.demo-tab-btn');
  tabBtns.forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');

  // Update panels
  const panels = document.querySelectorAll('.demo-panel');
  panels.forEach(p => p.classList.remove('active'));
  
  const targetPanel = document.getElementById('demo-' + demoId);
  if (targetPanel) {
    targetPanel.classList.add('active');
  }
}

// --- DEBOUNCE UTILITY ---
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

// ==========================================================================
// 1. BOSTON HOUSING CONTROLLER
// ==========================================================================
const updateHousing = debounce(async function() {
  const rm = parseFloat(document.getElementById('slider-rm').value);
  const lstat = parseFloat(document.getElementById('slider-lstat').value);
  const ptratio = parseFloat(document.getElementById('slider-ptratio').value);

  // Update labels
  document.getElementById('val-rm').textContent = rm.toFixed(2) + ' rooms';
  document.getElementById('val-lstat').textContent = lstat.toFixed(1) + '%';
  document.getElementById('val-ptratio').textContent = ptratio.toFixed(1);

  try {
    const res = await fetch('/api/predict/housing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rm, lstat, ptratio })
    });
    const data = await res.json();
    if (data.success) {
      document.getElementById('housing-price').textContent = data.price_formatted;
      document.getElementById('housing-tier').textContent = data.tier;
      
      const diffEl = document.getElementById('housing-diff');
      const sign = data.diff_pct >= 0 ? '+' : '';
      diffEl.textContent = sign + data.diff_pct + '%';
      if (data.diff_pct >= 0) {
        diffEl.className = 'text-emerald';
      } else {
        diffEl.className = 'text-rose';
      }
    }
  } catch (err) {
    console.error('Housing inference error:', err);
  }
}, 150);

// ==========================================================================
// 2. CHARITY ML DONOR CONTROLLER
// ==========================================================================
async function updateDonor() {
  const age = parseFloat(document.getElementById('donor-age').value) || 35;
  const education_num = parseFloat(document.getElementById('donor-education').value) || 12;
  const hours_per_week = parseFloat(document.getElementById('donor-hours').value) || 40;
  const marital_status = document.getElementById('donor-marital').value;
  const capital_gain = parseFloat(document.getElementById('donor-gain').value) || 0;
  const capital_loss = parseFloat(document.getElementById('donor-loss').value) || 0;

  try {
    const res = await fetch('/api/predict/donor', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        age, education_num, hours_per_week, marital_status, capital_gain, capital_loss
      })
    });
    const data = await res.json();
    if (data.success) {
      document.getElementById('donor-prob').textContent = data.probability + '%';
      document.getElementById('donor-decision').textContent = data.decision;
      document.getElementById('donor-conf').textContent = data.confidence;
      document.getElementById('donor-meter').style.width = data.probability + '%';

      const decisionEl = document.getElementById('donor-decision');
      if (data.is_donor) {
        decisionEl.className = 'tier-pill badge-success';
      } else {
        decisionEl.className = 'tier-pill badge-cyan';
      }
    }
  } catch (err) {
    console.error('Donor inference error:', err);
  }
}

// ==========================================================================
// 3. CUSTOMER SEGMENTS CONTROLLER
// ==========================================================================
const updateCustomer = debounce(async function() {
  const fresh = parseFloat(document.getElementById('slider-fresh').value);
  const grocery = parseFloat(document.getElementById('slider-grocery').value);
  const frozen = parseFloat(document.getElementById('slider-frozen').value);
  const detergents = parseFloat(document.getElementById('slider-detergents').value);

  // Update labels
  document.getElementById('val-fresh').textContent = fresh.toLocaleString();
  document.getElementById('val-grocery').textContent = grocery.toLocaleString();
  document.getElementById('val-frozen').textContent = frozen.toLocaleString();
  document.getElementById('val-detergents').textContent = detergents.toLocaleString();

  try {
    const res = await fetch('/api/predict/customer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        fresh,
        milk: grocery * 0.75, // proportional estimate
        grocery,
        frozen,
        detergents,
        delicatessen: fresh * 0.12
      })
    });
    const data = await res.json();
    if (data.success) {
      document.getElementById('cust-segment').textContent = data.segment_name;
      document.getElementById('cust-desc').textContent = data.description;
      document.getElementById('cust-confidence').textContent = 'Cluster Confidence: ' + data.confidence + '%';
      document.getElementById('pca-x').textContent = data.pca_x;
      document.getElementById('pca-y').textContent = data.pca_y;
    }
  } catch (err) {
    console.error('Customer segmentation error:', err);
  }
}, 150);

// ==========================================================================
// 4. SMARTCAB SIMULATOR CONTROLLER
// ==========================================================================
async function updateSmartcab() {
  const light = document.getElementById('cab-light').value;
  const waypoint = document.getElementById('cab-waypoint').value;
  const oncoming = document.getElementById('cab-oncoming').value;
  const left = document.getElementById('cab-left').value;

  try {
    const res = await fetch('/api/simulate/smartcab', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ light, waypoint, oncoming, left })
    });
    const data = await res.json();
    if (data.success) {
      document.getElementById('cab-action').textContent = data.action_label;
      document.getElementById('cab-safety').textContent = data.safety_assessment;
      document.getElementById('cab-status').textContent = data.status;

      const statusEl = document.getElementById('cab-status');
      if (data.status === 'PROCEED_LEGAL') {
        statusEl.className = 'tier-pill badge-success';
      } else if (data.status === 'YIELD_SAFE') {
        statusEl.className = 'tier-pill badge-cyan';
      } else {
        statusEl.className = 'tier-pill';
      }
    }
  } catch (err) {
    console.error('Smartcab simulation error:', err);
  }
}

// ==========================================================================
// 5. FAANG INTERVIEW SEARCH & FILTER ENGINE
// ==========================================================================
function filterFaang() {
  const query = document.getElementById('faangSearch').value.toLowerCase();
  const headings = document.querySelectorAll('.faang-body h3, .faang-body h2');

  headings.forEach(h => {
    // Traverse forward until next heading of same or higher level
    let content = h.textContent.toLowerCase();
    let sibling = h.nextElementSibling;
    let sectionText = content;

    while (sibling && !['H2', 'H3'].includes(sibling.tagName)) {
      sectionText += ' ' + sibling.textContent.toLowerCase();
      sibling = sibling.nextElementSibling;
    }

    const matches = sectionText.includes(query);
    
    // Toggle visibility
    h.style.display = matches ? '' : 'none';
    let s = h.nextElementSibling;
    while (s && !['H2', 'H3'].includes(s.tagName)) {
      s.style.display = matches ? '' : 'none';
      s = s.nextElementSibling;
    }
  });
}

function filterByCompany(company) {
  // Update button active state
  const pills = document.querySelectorAll('.company-pills .pill-btn');
  pills.forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');

  const searchInput = document.getElementById('faangSearch');
  if (company === 'ALL') {
    searchInput.value = '';
    filterFaang();
  } else {
    searchInput.value = company;
    filterFaang();
  }
}

// Initial trigger on demo page load
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('slider-rm')) {
    updateHousing();
  }
});
