/**
 * Personal Financial Digital Twin — Web Client Application
 * Integrated End-to-End Digital Twin Workflow over D10 REST API Backend Layer.
 */

let lastBaselineState = null;
let lastScenarioState = null;

document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initHealthCheck();
    initSchemaDisplay();
    initDerivedPreviewCalculations();
    initPresets();
    initFormActionHandlers();
    initSavingsGoalSimulator();
    initTooltips();
});

// Authoritative Primary Inputs
const PRIMARY_FIELDS = [
    'ending_balance_t',
    'income_credit_t',
    'debit_count_t',
    'spending_hh_t',
    'spending_st_t',
    'spending_in_t',
    'spending_lo_t',
    'spending_io_t',
    'spending_other_t',
    'spending_t_minus_1',
    'spending_t_minus_2'
];

/**
 * 1. Theme Switcher (Dark/Light Mode with localStorage persistence)
 */
function initThemeToggle() {
    const toggleBtn = document.getElementById('theme-toggle');
    if (!toggleBtn) return;

    const savedTheme = localStorage.getItem('pfd_twin_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);

    toggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('pfd_twin_theme', newTheme);
    });
}

/**
 * 2. Health Status Polling (/health)
 */
async function initHealthCheck() {
    const badge = document.getElementById('health-badge');
    const statusText = document.getElementById('health-status-text');
    if (!badge || !statusText) return;

    try {
        const response = await fetch('/health');
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        const data = await response.json();

        if (data.status === 'healthy' && data.model_artifact_exists) {
            badge.className = 'status-badge badge-online';
            statusText.textContent = 'System Ready';
        } else {
            badge.className = 'status-badge badge-pending';
            statusText.textContent = 'System Degrading';
        }
    } catch (err) {
        badge.className = 'status-badge badge-error';
        statusText.textContent = 'System Offline';
        console.error('Health check failed:', err);
    }
}

/**
 * 3. Authoritative Schema Specifications Display (/api/schema)
 */
async function initSchemaDisplay() {
    const featureContainer = document.getElementById('schema-feature-list');
    const rulesContainer = document.getElementById('schema-math-rules');
    if (!featureContainer || !rulesContainer) return;

    try {
        const response = await fetch('/api/schema');
        if (!response.ok) return;
        const data = await response.json();

        if (data.exact_14_feature_order) {
            featureContainer.innerHTML = '';
            data.exact_14_feature_order.forEach((feat, idx) => {
                const tag = document.createElement('span');
                const isDerived = data.derived_features && data.derived_features.includes(feat);
                tag.className = `schema-tag ${isDerived ? 'highlight' : ''}`;
                tag.textContent = `${idx + 1}. ${feat}${isDerived ? ' (derived)' : ''}`;
                featureContainer.appendChild(tag);
            });
        }

        if (data.derived_feature_rules) {
            rulesContainer.innerHTML = '';
            Object.entries(data.derived_feature_rules).forEach(([feat, rule]) => {
                const item = document.createElement('div');
                item.className = 'math-rule-item';
                item.innerHTML = `<strong>${feat}:</strong> ${rule}`;
                rulesContainer.appendChild(item);
            });
        }
    } catch (err) {
        console.warn('Could not load schema details:', err);
    }
}

/**
 * 4. Client-Side Derived Feature Preview Calculation (ddof=0 formula)
 */
function initDerivedPreviewCalculations() {
    const updatePreview = () => {
        const hh = parseFloat(document.getElementById('spending_hh_t').value) || 0;
        const st = parseFloat(document.getElementById('spending_st_t').value) || 0;
        const in_val = parseFloat(document.getElementById('spending_in_t').value) || 0;
        const lo = parseFloat(document.getElementById('spending_lo_t').value) || 0;
        const io = parseFloat(document.getElementById('spending_io_t').value) || 0;
        const other = parseFloat(document.getElementById('spending_other_t').value) || 0;
        const tm1 = parseFloat(document.getElementById('spending_t_minus_1').value) || 0;
        const tm2 = parseFloat(document.getElementById('spending_t_minus_2').value) || 0;

        const spending_t = hh + st + in_val + lo + io + other;
        const mean_3m = (spending_t + tm1 + tm2) / 3.0;

        // Population standard deviation ddof=0 formula
        const variance = ((spending_t - mean_3m) ** 2 + (tm1 - mean_3m) ** 2 + (tm2 - mean_3m) ** 2) / 3.0;
        const std_3m = Math.sqrt(variance);

        document.getElementById('preview-spending-t').textContent = formatCurrency(spending_t);
        document.getElementById('preview-3m-mean').textContent = formatCurrency(mean_3m);
        document.getElementById('preview-3m-std').textContent = formatCurrency(std_3m);
    };

    const inputs = document.querySelectorAll('.category-input, .lag-input');
    inputs.forEach(input => input.addEventListener('input', updatePreview));
    updatePreview();
}

/**
 * 5. Profile Presets
 */
function initPresets() {
    const presets = {
        default: {
            ending_balance_t: 15000, income_credit_t: 25000, debit_count_t: 12,
            spending_hh_t: 3000, spending_st_t: 500, spending_in_t: 1200,
            spending_lo_t: 2000, spending_io_t: 300, spending_other_t: 1500,
            spending_t_minus_1: 8000, spending_t_minus_2: 7500
        },
        high: {
            ending_balance_t: 45000, income_credit_t: 60000, debit_count_t: 28,
            spending_hh_t: 12000, spending_st_t: 1500, spending_in_t: 3500,
            spending_lo_t: 8000, spending_io_t: 1200, spending_other_t: 5000,
            spending_t_minus_1: 28000, spending_t_minus_2: 26000
        },
        saver: {
            ending_balance_t: 25000, income_credit_t: 30000, debit_count_t: 6,
            spending_hh_t: 1500, spending_st_t: 200, spending_in_t: 800,
            spending_lo_t: 0, spending_io_t: 0, spending_other_t: 500,
            spending_t_minus_1: 3200, spending_t_minus_2: 3000
        }
    };

    const loadProfile = (data) => {
        Object.keys(data).forEach(key => {
            const el = document.getElementById(key);
            if (el) el.value = data[key];
        });
        const hhInput = document.getElementById('spending_hh_t');
        if (hhInput) hhInput.dispatchEvent(new Event('input'));
        showToast('Financial profile preset loaded.', 'success');
    };

    const btnDef = document.getElementById('btn-preset-default');
    const btnHigh = document.getElementById('btn-preset-high');
    const btnSaver = document.getElementById('btn-preset-saver');

    if (btnDef) btnDef.addEventListener('click', () => loadProfile(presets.default));
    if (btnHigh) btnHigh.addEventListener('click', () => loadProfile(presets.high));
    if (btnSaver) btnSaver.addEventListener('click', () => loadProfile(presets.saver));
}

/**
 * 6. Form Actions & Integration Workflow Controls
 */
function initFormActionHandlers() {
    const btnPredict = document.getElementById('btn-predict');
    const btnSimulate = document.getElementById('btn-simulate');
    const btnExplainBase = document.getElementById('btn-explain-base');
    const btnExplainScen = document.getElementById('btn-explain-scenario');

    if (btnPredict) btnPredict.addEventListener('click', runPrediction);
    if (btnSimulate) btnSimulate.addEventListener('click', runSimulation);
    if (btnExplainBase) {
        btnExplainBase.addEventListener('click', () => {
            runSHAPExplanation(getFormPayload(), 'Current Profile');
        });
    }
    if (btnExplainScen) {
        btnExplainScen.addEventListener('click', () => {
            if (lastScenarioState) {
                runSHAPExplanation(lastScenarioState, 'Scenario Profile');
            } else {
                showToast('Run a scenario simulation first.', 'error');
            }
        });
    }
}

/**
 * Read current baseline financial state input payload
 */
function getFormPayload() {
    const payload = {};
    PRIMARY_FIELDS.forEach(field => {
        const el = document.getElementById(field);
        if (el) {
            payload[field] = field === 'debit_count_t' ? parseInt(el.value, 10) : parseFloat(el.value);
        }
    });
    return payload;
}

/**
 * POST /api/predict
 */
async function runPrediction() {
    const btn = document.getElementById('btn-predict');
    const forecastVal = document.getElementById('forecast-value');
    const forecastFooter = document.getElementById('forecast-footer');

    btn.disabled = true;
    forecastVal.textContent = '...';

    try {
        const payload = getFormPayload();
        lastBaselineState = payload;

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Prediction request failed');

        forecastVal.textContent = formatCurrency(data.prediction);
        forecastFooter.textContent = 'Forecast generated based on your current financial information.';
        updateSavingsContextIfActive();
        showToast('Spending forecast calculated successfully.', 'success');
    } catch (err) {
        forecastVal.textContent = 'Error';
        forecastFooter.textContent = `API Error: ${err.message}`;
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

/**
 * POST /api/simulate — Counterfactual Simulation & Visualizer
 */
async function runSimulation() {
    const btn = document.getElementById('btn-simulate');
    const fieldSelect = document.getElementById('sim-field-select').value;
    const newVal = parseFloat(document.getElementById('sim-new-value').value);

    if (isNaN(newVal)) {
        showToast('Please enter a valid numeric value.', 'error');
        return;
    }

    btn.disabled = true;

    try {
        const baselineState = getFormPayload();
        lastBaselineState = baselineState;

        const scenarioChanges = {};
        scenarioChanges[fieldSelect] = newVal;

        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                baseline_state: baselineState,
                scenario_changes: scenarioChanges
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Simulation request failed');

        lastScenarioState = data.scenario_state;

        // Render Numeric Results
        document.getElementById('sim-results-container').classList.remove('hidden');
        document.getElementById('btn-explain-scenario').classList.remove('hidden');

        const basePred = data.baseline_prediction;
        const scenPred = data.scenario_prediction;

        document.getElementById('sim-base-pred').textContent = formatCurrency(basePred);
        document.getElementById('sim-scen-pred').textContent = formatCurrency(scenPred);

        const absDiffEl = document.getElementById('sim-abs-diff');
        const pctDiffEl = document.getElementById('sim-pct-diff');

        const absDiff = data.absolute_difference;
        absDiffEl.textContent = `${absDiff >= 0 ? '+' : ''}${formatCurrency(absDiff)}`;
        absDiffEl.className = `sim-val ${absDiff >= 0 ? 'delta-tag-pos' : 'delta-tag-neg'}`;

        if (data.percentage_difference !== null) {
            const pctDiff = data.percentage_difference;
            pctDiffEl.textContent = `${pctDiff >= 0 ? '+' : ''}${pctDiff.toFixed(2)} %`;
            pctDiffEl.className = `sim-val ${pctDiff >= 0 ? 'delta-tag-pos' : 'delta-tag-neg'}`;
        } else {
            pctDiffEl.textContent = 'N/A';
        }

        // Render Side-by-Side Comparison Bars
        renderComparisonChart(basePred, scenPred);

        // Render Changed Primary Inputs Table
        renderChangedInputs(data.baseline_state, data.scenario_state);

        // Render Recalculated Derived Features Table
        renderRecalculatedDerived(data.baseline_state, data.scenario_state);

        showToast('Scenario analysis complete.', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

/**
 * Visual Comparison Side-by-Side Bar Chart Renderer
 */
function renderComparisonChart(baseVal, scenVal) {
    const maxVal = Math.max(baseVal, scenVal, 1.0);
    const basePct = Math.min(100, Math.max(5, (baseVal / maxVal) * 100));
    const scenPct = Math.min(100, Math.max(5, (scenVal / maxVal) * 100));

    const barBase = document.getElementById('chart-bar-baseline');
    const barScen = document.getElementById('chart-bar-scenario');
    const valBase = document.getElementById('chart-val-baseline');
    const valScen = document.getElementById('chart-val-scenario');

    if (barBase) barBase.style.width = `${basePct.toFixed(1)}%`;
    if (barScen) barScen.style.width = `${scenPct.toFixed(1)}%`;

    if (valBase) valBase.textContent = formatCurrency(baseVal);
    if (valScen) valScen.textContent = formatCurrency(scenVal);
}

/**
 * Render Changed Primary Inputs Breakdown
 */
function renderChangedInputs(baselineDict, scenarioDict) {
    const container = document.getElementById('changed-inputs-list');
    if (!container) return;
    container.innerHTML = '';

    const changedKeys = Object.keys(scenarioDict).filter(k => {
        return PRIMARY_FIELDS.includes(k) && baselineDict[k] !== scenarioDict[k];
    });

    if (changedKeys.length === 0) {
        container.innerHTML = '<div class="delta-row"><span class="delta-key">No changes detected.</span></div>';
        return;
    }

    // Friendly name mapping
    const friendlyNames = {
        'ending_balance_t': 'Current Balance',
        'income_credit_t': 'Money Coming In',
        'debit_count_t': 'Number of Payments',
        'spending_hh_t': 'Household',
        'spending_st_t': 'Everyday Spending',
        'spending_in_t': 'Insurance',
        'spending_lo_t': 'Loans / Repayments',
        'spending_io_t': 'Financial Payments',
        'spending_other_t': 'Other Spending',
        'spending_t_minus_1': 'Last Month',
        'spending_t_minus_2': 'Two Months Ago'
    };

    changedKeys.forEach(k => {
        const baseV = baselineDict[k];
        const scenV = scenarioDict[k];
        const delta = scenV - baseV;
        const isPos = delta >= 0;
        const friendlyName = friendlyNames[k] || k;

        const row = document.createElement('div');
        row.className = 'delta-row';
        row.innerHTML = `
            <span class="delta-key" data-tooltip="${k}">${friendlyName}</span>
            <div class="delta-vals">
                <span>${formatCurrency(baseV)}</span>
                <span class="delta-arrow">→</span>
                <span>${formatCurrency(scenV)}</span>
                <span class="${isPos ? 'delta-tag-pos' : 'delta-tag-neg'}">(${isPos ? '+' : ''}${formatCurrency(delta)})</span>
            </div>
        `;
        container.appendChild(row);
    });
}

/**
 * Render Backend Recalculated Derived Features Breakdown
 */
function renderRecalculatedDerived(baselineDict, scenarioDict) {
    const container = document.getElementById('recalculated-derived-list');
    if (!container) return;
    container.innerHTML = '';

    // Calculate derived values for both baseline and scenario
    const b_hh = baselineDict.spending_hh_t || 0, b_st = baselineDict.spending_st_t || 0, b_in = baselineDict.spending_in_t || 0;
    const b_lo = baselineDict.spending_lo_t || 0, b_io = baselineDict.spending_io_t || 0, b_ot = baselineDict.spending_other_t || 0;
    const b_tm1 = baselineDict.spending_t_minus_1 || 0, b_tm2 = baselineDict.spending_t_minus_2 || 0;

    const b_s_t = b_hh + b_st + b_in + b_lo + b_io + b_ot;
    const b_mean = (b_s_t + b_tm1 + b_tm2) / 3.0;
    const b_std = Math.sqrt(((b_s_t - b_mean)**2 + (b_tm1 - b_mean)**2 + (b_tm2 - b_mean)**2) / 3.0);

    const s_hh = scenarioDict.spending_hh_t || 0, s_st = scenarioDict.spending_st_t || 0, s_in = scenarioDict.spending_in_t || 0;
    const s_lo = scenarioDict.spending_lo_t || 0, s_io = scenarioDict.spending_io_t || 0, s_ot = scenarioDict.spending_other_t || 0;
    const s_tm1 = scenarioDict.spending_t_minus_1 || 0, s_tm2 = scenarioDict.spending_t_minus_2 || 0;

    const s_s_t = s_hh + s_st + s_in + s_lo + s_io + s_ot;
    const s_mean = (s_s_t + s_tm1 + s_tm2) / 3.0;
    const s_std = Math.sqrt(((s_s_t - s_mean)**2 + (s_tm1 - s_mean)**2 + (s_tm2 - s_mean)**2) / 3.0);

    const derivedMetrics = [
        { name: 'spending_t', friendly: 'Current Spending', base: b_s_t, scen: s_s_t },
        { name: 'spending_3m_mean', friendly: '3-Month Average', base: b_mean, scen: s_mean },
        { name: 'spending_3m_std (ddof=0)', friendly: 'Spending Variation', base: b_std, scen: s_std }
    ];

    derivedMetrics.forEach(m => {
        const delta = m.scen - m.base;
        const isPos = delta >= 0;

        const row = document.createElement('div');
        row.className = 'delta-row';
        row.innerHTML = `
            <span class="delta-key" data-tooltip="${m.name}">${m.friendly}</span>
            <div class="delta-vals">
                <span>${formatCurrency(m.base)}</span>
                <span class="delta-arrow">→</span>
                <span>${formatCurrency(m.scen)}</span>
                <span class="${isPos ? 'delta-tag-pos' : 'delta-tag-neg'}">(${isPos ? '+' : ''}${formatCurrency(delta)})</span>
            </div>
        `;
        container.appendChild(row);
    });
}

/**
 * POST /api/explain
 */
async function runSHAPExplanation(statePayload, targetLabel = 'Baseline Profile') {
    const btnBase = document.getElementById('btn-explain-base');
    const btnScen = document.getElementById('btn-explain-scenario');
    const container = document.getElementById('shap-bars-container');
    const metaBar = document.getElementById('shap-metadata');
    const tag = document.getElementById('shap-target-tag');

    if (btnBase) btnBase.disabled = true;
    if (btnScen) btnScen.disabled = true;
    if (tag) tag.textContent = targetLabel;
    if (container) container.innerHTML = '<div class="empty-state-text">Computing explanation...</div>';

    try {
        const response = await fetch('/api/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(statePayload)
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Explanation request failed');

        // Update Metadata
        if (metaBar) metaBar.classList.remove('hidden');
        document.getElementById('shap-base-val').textContent = formatCurrency(data.base_value);
        document.getElementById('shap-recon-val').textContent = formatCurrency(data.reconstructed_prediction);
        document.getElementById('shap-delta-val').textContent = `${data.additivity_delta.toFixed(6)}`;

        // Render SHAP Bars
        container.innerHTML = '';
        const maxShap = Math.max(...data.features.map(f => Math.abs(f.shap_value)), 1.0);

        // Friendly name mapping for SHAP features
        const friendlyNames = {
            'spending_t': 'Current Spending',
            'spending_t_minus_1': 'Last Month',
            'spending_t_minus_2': 'Two Months Ago',
            'spending_3m_mean': '3-Month Average',
            'spending_3m_std': 'Spending Variation',
            'debit_count_t': 'Number of Payments',
            'income_credit_t': 'Money Coming In',
            'ending_balance_t': 'Current Balance',
            'spending_hh_t': 'Household',
            'spending_st_t': 'Everyday Spending',
            'spending_in_t': 'Insurance',
            'spending_lo_t': 'Loans / Repayments',
            'spending_io_t': 'Financial Payments',
            'spending_other_t': 'Other Spending'
        };

        data.features.forEach(f => {
            const isPos = f.shap_value >= 0;
            const pctWidth = Math.min(100, Math.max(3, (Math.abs(f.shap_value) / maxShap) * 100));
            const friendlyName = friendlyNames[f.feature] || f.feature;

            const row = document.createElement('div');
            row.className = 'shap-row';
            row.innerHTML = `
                <span class="shap-feat-name" data-tooltip="${f.feature}">${friendlyName}</span>
                <span class="shap-feat-val">${typeof f.value === 'number' ? f.value.toLocaleString() : f.value}</span>
                <div class="shap-bar-track">
                    <div class="shap-bar-fill ${isPos ? 'positive' : 'negative'}" style="width: ${pctWidth.toFixed(1)}%;"></div>
                </div>
                <span class="shap-value-num ${isPos ? 'positive' : 'negative'}">${isPos ? '+' : ''}${f.shap_value.toFixed(4)}</span>
            `;
            container.appendChild(row);
        });

        // Re-initialize tooltips for dynamically added elements
        initTooltips();

        showToast(`Explanation loaded for ${targetLabel}.`, 'success');
    } catch (err) {
        if (container) container.innerHTML = `<div class="empty-state-text" style="color: var(--color-error);">Error loading SHAP attributions: ${err.message}</div>`;
        showToast(err.message, 'error');
    } finally {
        if (btnBase) btnBase.disabled = false;
        if (btnScen) btnScen.disabled = false;
    }
}

/**
 * Helper Utilities
 */
function formatCurrency(val) {
    return val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function showToast(msg, type = 'info') {
    const toast = document.getElementById('toast-notification');
    const toastMsg = document.getElementById('toast-message');
    if (!toast || !toastMsg) return;

    toastMsg.textContent = msg;
    toast.className = `toast toast-${type}`;

    setTimeout(() => {
        toast.className = 'toast hidden';
    }, 4000);
}

/**
 * 7. Tooltip System for Technical Feature Names
 */
function initTooltips() {
    const tooltip = document.getElementById('tooltip');
    if (!tooltip) return;

    const tooltipElements = document.querySelectorAll('[data-tooltip]');

    tooltipElements.forEach(element => {
        // Skip if already has tooltip listeners
        if (element.dataset.tooltipInitialized === 'true') return;

        element.classList.add('has-tooltip');
        element.dataset.tooltipInitialized = 'true';

        const showTooltip = () => {
            const technicalName = element.getAttribute('data-tooltip');
            if (!technicalName) return;

            tooltip.textContent = technicalName;
            tooltip.classList.remove('hidden');

            const rect = element.getBoundingClientRect();
            const tooltipRect = tooltip.getBoundingClientRect();

            let top = rect.bottom + 8;
            let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);

            // Keep tooltip within viewport
            if (left < 10) left = 10;
            if (left + tooltipRect.width > window.innerWidth - 10) {
                left = window.innerWidth - tooltipRect.width - 10;
            }
            if (top + tooltipRect.height > window.innerHeight - 10) {
                top = rect.top - tooltipRect.height - 8;
            }

            tooltip.style.top = `${top}px`;
            tooltip.style.left = `${left}px`;
        };

        const hideTooltip = () => {
            tooltip.classList.add('hidden');
        };

        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
        element.addEventListener('focus', showTooltip);
        element.addEventListener('blur', hideTooltip);
    });
}

/**
 * 8. Dimension 14 — Savings Goal Simulator
 */
let lastSavingsPlanState = null;

function escapeHTML(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function initSavingsGoalSimulator() {
    const btnCalc = document.getElementById('btn-calculate-savings');
    if (btnCalc) {
        btnCalc.addEventListener('click', calculateSavingsPlan);
    }
}

function calculateSavingsPlan() {
    const nameInput = document.getElementById('goal-name-input');
    const amountInput = document.getElementById('goal-amount-input');
    const monthsInput = document.getElementById('goal-months-input');

    const rawName = nameInput ? nameInput.value.trim() : '';
    const rawAmount = amountInput ? amountInput.value.trim() : '';
    const rawMonths = monthsInput ? monthsInput.value.trim() : '';

    if (!rawAmount || isNaN(rawAmount)) {
        showToast('Please enter a valid goal amount.', 'error');
        return;
    }

    const amount = parseFloat(rawAmount);
    if (!isFinite(amount) || amount <= 0) {
        showToast('Please enter a valid goal amount greater than 0.', 'error');
        return;
    }

    if (!rawMonths || isNaN(rawMonths)) {
        showToast('Please enter a whole number of months.', 'error');
        return;
    }

    const monthsNum = Number(rawMonths);
    if (!isFinite(monthsNum) || !Number.isInteger(monthsNum) || monthsNum <= 0) {
        showToast('Please enter a whole number of months greater than 0.', 'error');
        return;
    }

    if (monthsNum > 60) {
        showToast('Timeframe cannot exceed 60 months.', 'error');
        return;
    }

    const escapedGoalName = escapeHTML(rawName);

    // Exact penny rounding mathematics ensuring sum(monthly_savings) == goal_amount
    const schedule = [];
    let cumulativeSaved = 0;
    const baseMonthly = Math.round((amount / monthsNum) * 100) / 100;

    for (let m = 1; m <= monthsNum; m++) {
        let monthlyPlan;
        if (m === monthsNum) {
            monthlyPlan = Math.max(0, Math.round((amount - cumulativeSaved) * 100) / 100);
        } else {
            monthlyPlan = Math.min(baseMonthly, Math.max(0, Math.round((amount - cumulativeSaved) * 100) / 100));
        }
        cumulativeSaved += monthlyPlan;
        let remaining = Math.max(0, Math.round((amount - cumulativeSaved) * 100) / 100);
        if (m === monthsNum) remaining = 0;

        schedule.push({
            month: m,
            plannedSaving: monthlyPlan,
            remainingGoal: remaining
        });
    }

    const requiredMonthlyAverage = amount / monthsNum;

    lastSavingsPlanState = {
        goalName: escapedGoalName,
        rawName: rawName,
        amount: amount,
        months: monthsNum,
        requiredMonthly: requiredMonthlyAverage,
        schedule: schedule
    };

    renderSavingsPlan();
}

function renderSavingsPlan() {
    if (!lastSavingsPlanState) return;

    const { goalName, rawName, amount, months, requiredMonthly, schedule } = lastSavingsPlanState;
    const container = document.getElementById('savings-results-container');
    if (!container) return;

    container.classList.remove('hidden');

    document.getElementById('savings-target-val').textContent = formatCurrency(amount);
    document.getElementById('savings-time-val').textContent = `${months} ${months === 1 ? 'month' : 'months'}`;
    document.getElementById('savings-monthly-val').textContent = formatCurrency(requiredMonthly);

    const titleEl = document.getElementById('schedule-table-title');
    if (titleEl) {
        titleEl.textContent = rawName ? `Saving Plan: ${rawName}` : 'Your Saving Plan';
    }

    // Render ML Forecast Context & Feasibility Indicator
    renderSavingsContext(requiredMonthly);

    // Render Schedule Table Rows
    const tbody = document.getElementById('savings-schedule-body');
    if (tbody) {
        tbody.innerHTML = '';
        schedule.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>Month ${row.month}</td>
                <td>${formatCurrency(row.plannedSaving)}</td>
                <td>${formatCurrency(row.remainingGoal)}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    showToast('Saving plan generated successfully.', 'success');
}

function renderSavingsContext(requiredMonthly) {
    const contextBody = document.getElementById('savings-context-body');
    if (!contextBody) return;
    contextBody.innerHTML = '';

    const titleEl = document.getElementById('schedule-table-title');
    const rawName = lastSavingsPlanState ? lastSavingsPlanState.rawName : '';

    const payload = getFormPayload();
    const income = payload && typeof payload.income_credit_t === 'number' ? payload.income_credit_t : null;

    const forecastEl = document.getElementById('forecast-value');
    let forecastVal = null;
    if (forecastEl && forecastEl.textContent && forecastEl.textContent !== '--' && forecastEl.textContent !== '...' && forecastEl.textContent !== 'Error') {
        const textClean = forecastEl.textContent.replace(/,/g, '');
        const parsed = parseFloat(textClean);
        if (!isNaN(parsed)) forecastVal = parsed;
    }

    if (income !== null && forecastVal !== null) {
        const estimatedAvailable = income - forecastVal;
        const isNonPositiveAvailable = estimatedAvailable <= 0;
        const isExceedingAvailable = !isNonPositiveAvailable && requiredMonthly > estimatedAvailable;
        const isWithinEstimate = !isNonPositiveAvailable && !isExceedingAvailable;

        const contextBox = document.createElement('div');
        contextBox.className = 'context-box';

        let feasibilityPillClass = 'badge-info';
        let feasibilityTitle = '';
        let feasibilityMessage = '';

        if (isNonPositiveAvailable) {
            feasibilityPillClass = 'badge-warning';
            feasibilityTitle = estimatedAvailable < 0 ? 'Spending Exceeds Income' : 'Not Currently Feasible';
            feasibilityMessage = `Goal requires ${formatCurrency(requiredMonthly)} per month, but the estimated amount available after forecasted spending is ${formatCurrency(estimatedAvailable)}. This goal is not currently feasible based on the available estimate. Consider a longer timeline or explore a spending scenario.`;
            if (titleEl) {
                titleEl.textContent = rawName ? `Target Saving Schedule: ${rawName} (Illustrative)` : 'Target Saving Schedule (Illustrative)';
            }
        } else if (isExceedingAvailable) {
            feasibilityPillClass = 'badge-warning';
            feasibilityTitle = 'Above Estimated Amount';
            feasibilityMessage = `Goal requires ${formatCurrency(requiredMonthly)} per month, but the estimated amount available after forecasted spending is ${formatCurrency(estimatedAvailable)}. The required monthly saving is above the current estimated amount available after forecasted spending. Consider adjusting the goal timeline or exploring a spending scenario.`;
            if (titleEl) {
                titleEl.textContent = rawName ? `Target Saving Schedule: ${rawName} (Illustrative)` : 'Target Saving Schedule (Illustrative)';
            }
        } else {
            feasibilityPillClass = 'badge-success';
            feasibilityTitle = 'Within Estimated Amount';
            feasibilityMessage = `Goal requires ${formatCurrency(requiredMonthly)} per month, which is within the estimated amount available after forecasted spending (${formatCurrency(estimatedAvailable)}). Note: This is an illustrative target based on model estimates and does not guarantee affordability.`;
            if (titleEl) {
                titleEl.textContent = rawName ? `Saving Plan: ${rawName}` : 'Your Saving Plan';
            }
        }

        let htmlContent = `
            <div class="context-metrics-grid">
                <div class="context-metric">
                    <span class="context-label">Money Coming In</span>
                    <span class="context-val">${formatCurrency(income)}</span>
                </div>
                <div class="context-metric">
                    <span class="context-label">Current Spending Estimate</span>
                    <span class="context-val">${formatCurrency(forecastVal)}</span>
                </div>
                <div class="context-metric highlight">
                    <span class="context-label">Estimated Amount Available After Forecasted Spending</span>
                    <span class="context-val ${estimatedAvailable <= 0 ? 'text-error' : ''}">${formatCurrency(estimatedAvailable)}</span>
                </div>
            </div>
            <div class="feasibility-status-banner">
                <div class="status-header">
                    <span class="badge ${feasibilityPillClass}">${feasibilityTitle}</span>
                </div>
                <p class="status-text">${feasibilityMessage}</p>
        `;

        if (!isWithinEstimate) {
            htmlContent += `
                <button type="button" class="btn btn-secondary btn-sm btn-goto-sim" id="btn-savings-goto-sim">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m16 3 4 4-4 4"/><path d="M20 7H4"/><path d="m8 21-4-4 4-4"/><path d="M4 17h16"/></svg>
                    Explore a Spending Scenario
                </button>
            `;
        }

        htmlContent += `</div>`;
        contextBox.innerHTML = htmlContent;
        contextBody.appendChild(contextBox);

        const btnSimShortcut = document.getElementById('btn-savings-goto-sim');
        if (btnSimShortcut) {
            btnSimShortcut.addEventListener('click', () => {
                const simSection = document.querySelector('.simulator-card');
                if (simSection) {
                    simSection.scrollIntoView({ behavior: 'smooth' });
                    const selectEl = document.getElementById('sim-field-select');
                    if (selectEl) selectEl.focus();
                }
            });
        }
    } else {
        if (titleEl) {
            titleEl.textContent = rawName ? `Target Saving Schedule: ${rawName} (Illustrative)` : 'Target Saving Schedule (Illustrative)';
        }
        contextBody.innerHTML = `
            <div class="context-empty-note">
                <p>Forecast your next month's spending to compare your monthly saving target against your estimated available monthly amount.</p>
            </div>
        `;
    }
}

function updateSavingsContextIfActive() {
    if (lastSavingsPlanState) {
        renderSavingsContext(lastSavingsPlanState.requiredMonthly);
    }
}

