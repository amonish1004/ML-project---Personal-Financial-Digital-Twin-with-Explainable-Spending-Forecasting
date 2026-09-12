/**
 * Personal Financial Digital Twin — Web Frontend Client Application
 * Communicates exclusively with the D10 FastAPI REST API layer.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI Elements
    initHealthCheck();
    initDerivedPreviewCalculations();
    initPresets();
    initFormActionHandlers();
});

// Primary Input Field List
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
 * 1. Health Status Polling (/health)
 */
async function initHealthCheck() {
    const badge = document.getElementById('health-badge');
    const statusText = document.getElementById('health-status-text');

    try {
        const response = await fetch('/health');
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        const data = await response.json();

        if (data.status === 'healthy' && data.model_artifact_exists) {
            badge.className = 'status-badge badge-online';
            statusText.textContent = 'API Online • Model Ready';
        } else {
            badge.className = 'status-badge badge-pending';
            statusText.textContent = 'API Degrading • Check Artifacts';
        }
    } catch (err) {
        badge.className = 'status-badge badge-error';
        statusText.textContent = 'API Offline • Check Server';
        console.error('Health check failed:', err);
    }
}

/**
 * 2. Client-Side Derived Feature Preview Calculation (Fidelity to backend ddof=0 formula)
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

        document.getElementById('preview-spending-t').textContent = `${formatCurrency(spending_t)} CZK`;
        document.getElementById('preview-3m-mean').textContent = `${formatCurrency(mean_3m)} CZK`;
        document.getElementById('preview-3m-std').textContent = `${formatCurrency(std_3m)} CZK`;
    };

    const inputs = document.querySelectorAll('.category-input, .lag-input');
    inputs.forEach(input => input.addEventListener('input', updatePreview));
    updatePreview();
}

/**
 * 3. Profile Presets
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
        // Trigger input event to update previews
        document.getElementById('spending_hh_t').dispatchEvent(new Event('input'));
        showToast('Financial profile preset loaded.', 'success');
    };

    document.getElementById('btn-preset-default').addEventListener('click', () => loadProfile(presets.default));
    document.getElementById('btn-preset-high').addEventListener('click', () => loadProfile(presets.high));
    document.getElementById('btn-preset-saver').addEventListener('click', () => loadProfile(presets.saver));
}

/**
 * 4. Form Actions & API Integration
 */
function initFormActionHandlers() {
    document.getElementById('btn-predict').addEventListener('click', runPrediction);
    document.getElementById('btn-simulate').addEventListener('click', runSimulation);
    document.getElementById('btn-explain').addEventListener('click', runSHAPExplanation);
}

/**
 * Read current financial state input payload
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
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Prediction request failed');

        forecastVal.textContent = formatCurrency(data.prediction);
        forecastFooter.textContent = `Forecast generated via predict_spending() gateway (${data.currency}).`;
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
 * POST /api/simulate
 */
async function runSimulation() {
    const btn = document.getElementById('btn-simulate');
    const fieldSelect = document.getElementById('sim-field-select').value;
    const newVal = parseFloat(document.getElementById('sim-new-value').value);

    if (isNaN(newVal)) {
        showToast('Please enter a valid numeric counterfactual value.', 'error');
        return;
    }

    btn.disabled = true;

    try {
        const baselineState = getFormPayload();
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

        document.getElementById('sim-results-container').classList.remove('hidden');
        document.getElementById('sim-base-pred').textContent = `${formatCurrency(data.baseline_prediction)} CZK`;
        document.getElementById('sim-scen-pred').textContent = `${formatCurrency(data.scenario_prediction)} CZK`;
        
        const absDiffEl = document.getElementById('sim-abs-diff');
        const pctDiffEl = document.getElementById('sim-pct-diff');

        const absDiff = data.absolute_difference;
        absDiffEl.textContent = `${absDiff >= 0 ? '+' : ''}${formatCurrency(absDiff)} CZK`;
        absDiffEl.style.color = absDiff >= 0 ? '#f87171' : '#38bdf8';

        if (data.percentage_difference !== null) {
            const pctDiff = data.percentage_difference;
            pctDiffEl.textContent = `${pctDiff >= 0 ? '+' : ''}${pctDiff.toFixed(2)} %`;
            pctDiffEl.style.color = pctDiff >= 0 ? '#f87171' : '#38bdf8';
        } else {
            pctDiffEl.textContent = 'N/A';
        }

        showToast('Counterfactual scenario simulation complete.', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

/**
 * POST /api/explain
 */
async function runSHAPExplanation() {
    const btn = document.getElementById('btn-explain');
    const container = document.getElementById('shap-bars-container');
    const metaBar = document.getElementById('shap-metadata');

    btn.disabled = true;
    container.innerHTML = '<div class="empty-state-text">Computing TreeSHAP feature attributions...</div>';

    try {
        const payload = getFormPayload();
        const response = await fetch('/api/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Explanation request failed');

        // Update Metadata
        metaBar.classList.remove('hidden');
        document.getElementById('shap-base-val').textContent = formatCurrency(data.base_value);
        document.getElementById('shap-recon-val').textContent = formatCurrency(data.reconstructed_prediction);
        document.getElementById('shap-delta-val').textContent = `${data.additivity_delta.toFixed(6)}`;

        // Render SHAP Bars
        container.innerHTML = '';
        const maxShap = Math.max(...data.features.map(f => Math.abs(f.shap_value)), 1.0);

        data.features.forEach(f => {
            const isPos = f.shap_value >= 0;
            const pctWidth = Math.min(100, Math.max(4, (Math.abs(f.shap_value) / maxShap) * 100));

            const row = document.createElement('div');
            row.className = 'shap-row';
            row.innerHTML = `
                <span class="shap-feat-name" title="${f.feature}">${f.feature}</span>
                <span class="shap-feat-val">${typeof f.value === 'number' ? f.value.toLocaleString() : f.value}</span>
                <div class="shap-bar-track">
                    <div class="shap-bar-fill ${isPos ? 'positive' : 'negative'}" style="width: ${pctWidth.toFixed(1)}%;"></div>
                </div>
                <span class="shap-value-num ${isPos ? 'positive' : 'negative'}">${isPos ? '+' : ''}${f.shap_value.toFixed(4)}</span>
            `;
            container.appendChild(row);
        });

        showToast('TreeSHAP feature attributions loaded.', 'success');
    } catch (err) {
        container.innerHTML = `<div class="empty-state-text" style="color: #f87171;">Error loading SHAP attributions: ${err.message}</div>`;
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
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

    toastMsg.textContent = msg;
    toast.className = `toast toast-${type}`;

    setTimeout(() => {
        toast.className = 'toast hidden';
    }, 4000);
}
