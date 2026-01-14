/**
 * LyricalLattice Frontend Application
 * 21 Wordplay Device Detection Interface
 */

// Configuration
const CONFIG = {
    API_BASE: 'http://localhost:8000/api/v1',
    DEFAULT_TOLERANCE: 0.9,
    DEFAULT_MAX_RESULTS: 50,
    DEBOUNCE_MS: 300
};

// Device type display names and descriptions
const DEVICE_INFO = {
    homophone: { name: 'Homophone', desc: 'Words that sound identical' },
    oronym: { name: 'Oronym', desc: 'Phrases with different word boundaries' },
    perfect_rhyme: { name: 'Perfect Rhyme', desc: 'Identical rhyme sounds' },
    slant_rhyme: { name: 'Slant Rhyme', desc: 'Near-rhyme with similar sounds' },
    assonance: { name: 'Assonance', desc: 'Repeated vowel sounds' },
    consonance: { name: 'Consonance', desc: 'Repeated consonant sounds' },
    alliteration: { name: 'Alliteration', desc: 'Repeated initial sounds' },
    internal_rhyme: { name: 'Internal Rhyme', desc: 'Rhymes within a line' },
    multisyllabic_rhyme: { name: 'Multisyllabic', desc: 'Multi-syllable rhyme patterns' },
    compound_rhyme: { name: 'Compound', desc: 'Multi-word rhyme combinations' },
    onomatopoeia: { name: 'Onomatopoeia', desc: 'Sound-imitating words' },
    euphony: { name: 'Euphony', desc: 'Pleasant sound quality' },
    cacophony: { name: 'Cacophony', desc: 'Harsh sound quality' },
    stacked_rhyme: { name: 'Stacked Rhyme', desc: 'Multiple rhymes in sequence' },
    pun: { name: 'Pun', desc: 'Wordplay exploiting multiple meanings' },
    double_entendre: { name: 'Double Entendre', desc: 'Hidden secondary meaning' },
    malapropism: { name: 'Malapropism', desc: 'Incorrect word substitution' },
    mondegreen: { name: 'Mondegreen', desc: 'Misheard phrase' },
    polyrhythmic_rhyme: { name: 'Polyrhythmic', desc: 'Rhymes on varying beats' },
    breath_rhyme: { name: 'Breath Rhyme', desc: 'Rhymes at pause points' },
    melisma: { name: 'Melisma', desc: 'Stretched syllable patterns' },
    sample_flip: { name: 'Sample Flip', desc: 'Repurposed sample phonemes' }
};

// State management
const state = {
    currentFilter: 'all',
    tolerance: CONFIG.DEFAULT_TOLERANCE,
    dialect: 'en-US',
    genre: 'general',
    lastResults: null,
    isLoading: false
};

// DOM Elements
const elements = {
    searchInput: document.getElementById('searchInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    dialectSelector: document.getElementById('dialectSelector'),
    toleranceSlider: document.getElementById('toleranceSlider'),
    toleranceValue: document.getElementById('toleranceValue'),
    genreSelector: document.getElementById('genreSelector'),
    phonemeDisplay: document.getElementById('phonemeDisplay'),
    phonemeOutput: document.getElementById('phonemeOutput'),
    loadingState: document.getElementById('loadingState'),
    resultsSection: document.getElementById('resultsSection'),
    resultsGrid: document.getElementById('resultsGrid'),
    emptyState: document.getElementById('emptyState'),
    errorState: document.getElementById('errorState'),
    errorMessage: document.getElementById('errorMessage'),
    totalMatches: document.getElementById('totalMatches'),
    deviceCount: document.getElementById('deviceCount'),
    processingTime: document.getElementById('processingTime'),
    tabButtons: document.querySelectorAll('.tab-btn')
};

// Initialize application
function init() {
    setupEventListeners();
    setupAccessibility();
}

// Event listeners setup
function setupEventListeners() {
    // Search functionality
    elements.analyzeBtn.addEventListener('click', handleAnalyze);
    elements.searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleAnalyze();
    });

    // Controls
    elements.toleranceSlider.addEventListener('input', handleToleranceChange);
    elements.dialectSelector.addEventListener('change', handleDialectChange);
    elements.genreSelector.addEventListener('change', handleGenreChange);

    // Tab filtering
    elements.tabButtons.forEach(btn => {
        btn.addEventListener('click', handleTabClick);
    });
}

// Accessibility setup
function setupAccessibility() {
    // Keyboard navigation for tabs
    const tabList = document.querySelector('.pattern-tabs');
    tabList.addEventListener('keydown', (e) => {
        const tabs = Array.from(elements.tabButtons);
        const currentIndex = tabs.findIndex(tab => tab === document.activeElement);

        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
            e.preventDefault();
            const nextIndex = (currentIndex + 1) % tabs.length;
            tabs[nextIndex].focus();
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
            e.preventDefault();
            const prevIndex = (currentIndex - 1 + tabs.length) % tabs.length;
            tabs[prevIndex].focus();
        }
    });
}

// Control handlers
function handleToleranceChange(e) {
    state.tolerance = parseFloat(e.target.value);
    elements.toleranceValue.textContent = state.tolerance.toFixed(2);
}

function handleDialectChange(e) {
    state.dialect = e.target.value;
}

function handleGenreChange(e) {
    state.genre = e.target.value;
}

// Tab click handler
function handleTabClick(e) {
    const btn = e.currentTarget;

    // Update active state
    elements.tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');

    // Update filter
    if (btn.dataset.category === 'all') {
        state.currentFilter = 'all';
    } else {
        state.currentFilter = btn.dataset.device;
    }

    // Re-render results if we have them
    if (state.lastResults) {
        renderResults(state.lastResults);
    }
}

// Main analysis handler
async function handleAnalyze() {
    const text = elements.searchInput.value.trim();
    if (!text || state.isLoading) return;

    state.isLoading = true;
    showLoading();

    try {
        // Call the detect_wordplay API
        const response = await fetch(`${CONFIG.API_BASE}/detect_wordplay`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                categories: ['all'],
                min_confidence: state.tolerance,
                max_results: CONFIG.DEFAULT_MAX_RESULTS
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `API error: ${response.status}`);
        }

        const data = await response.json();
        state.lastResults = data;

        // Display phonemes
        if (data.input_phonemes && data.input_phonemes.length > 0) {
            displayPhonemes(data.input_phonemes);
        }

        // Render results
        renderResults(data);

    } catch (err) {
        showError(err.message);
    } finally {
        state.isLoading = false;
        hideLoading();
    }
}

// Display phonemes
function displayPhonemes(phonemes) {
    const phonemeStr = Array.isArray(phonemes[0])
        ? phonemes.map(p => p.join(' ')).join(' | ')
        : phonemes.join(' ');

    elements.phonemeOutput.textContent = phonemeStr;
    elements.phonemeDisplay.classList.remove('hidden');
}

// Render results
function renderResults(data) {
    const matches = data.matches || [];

    // Filter matches based on current filter
    const filteredMatches = state.currentFilter === 'all'
        ? matches
        : matches.filter(m => normalizeDeviceType(m.device_type) === state.currentFilter);

    // Update summary stats
    elements.totalMatches.textContent = filteredMatches.length;
    elements.deviceCount.textContent = new Set(matches.map(m => m.device_type)).size;
    elements.processingTime.textContent = data.processing_time_ms
        ? `${data.processing_time_ms.toFixed(0)}ms`
        : '';

    // Clear and render
    elements.resultsGrid.innerHTML = '';

    if (filteredMatches.length === 0) {
        elements.emptyState.classList.remove('hidden');
        elements.resultsGrid.classList.add('hidden');
    } else {
        elements.emptyState.classList.add('hidden');
        elements.resultsGrid.classList.remove('hidden');

        filteredMatches.forEach(match => {
            const matchEl = createMatchElement(match);
            elements.resultsGrid.appendChild(matchEl);
        });
    }

    elements.resultsSection.classList.remove('hidden');
    elements.errorState.classList.add('hidden');
}

// Normalize device type to match our data attributes
function normalizeDeviceType(deviceType) {
    return deviceType.toLowerCase().replace(/\s+/g, '_');
}

// Create match element
function createMatchElement(match) {
    const deviceKey = normalizeDeviceType(match.device_type);
    const deviceInfo = DEVICE_INFO[deviceKey] || { name: match.device_type, desc: '' };

    const div = document.createElement('div');
    div.className = 'match-item';
    div.setAttribute('data-device', deviceKey);
    div.setAttribute('role', 'listitem');

    // Extract display text from match details
    const displayText = getMatchDisplayText(match);
    const phonemeText = getMatchPhonemes(match);
    const explanation = getMatchExplanation(match);

    div.innerHTML = `
        <div class="match-header">
            <span class="match-device">${deviceInfo.name}</span>
            <span class="match-confidence">${Math.round(match.confidence * 100)}%</span>
        </div>
        <div class="match-content">
            <div class="match-text">${escapeHtml(displayText)}</div>
            ${phonemeText ? `<div class="match-phonemes">${escapeHtml(phonemeText)}</div>` : ''}
        </div>
        ${explanation ? `<div class="match-explanation">${escapeHtml(explanation)}</div>` : ''}
    `;

    return div;
}

// Extract display text from match details
function getMatchDisplayText(match) {
    const details = match.details || {};

    // Try various common fields
    if (details.words && Array.isArray(details.words)) {
        return details.words.join(' ');
    }
    if (details.phrase) {
        return Array.isArray(details.phrase) ? details.phrase.join(' ') : details.phrase;
    }
    if (details.word1 && details.word2) {
        return `${details.word1} / ${details.word2}`;
    }
    if (details.matched_words) {
        return Array.isArray(details.matched_words)
            ? details.matched_words.join(', ')
            : details.matched_words;
    }
    if (details.text) {
        return details.text;
    }
    if (details.alternative) {
        return details.alternative;
    }

    // Fallback
    return match.device_type;
}

// Extract phonemes from match details
function getMatchPhonemes(match) {
    const details = match.details || {};

    if (details.phonemes) {
        return Array.isArray(details.phonemes)
            ? details.phonemes.join(' ')
            : details.phonemes;
    }
    if (details.ipa) {
        return details.ipa;
    }
    if (details.pronunciation) {
        return details.pronunciation;
    }

    return null;
}

// Extract explanation from match details
function getMatchExplanation(match) {
    const details = match.details || {};

    if (details.explanation) {
        return details.explanation;
    }
    if (details.description) {
        return details.description;
    }

    // Generate explanation based on device type
    const deviceKey = normalizeDeviceType(match.device_type);
    const deviceInfo = DEVICE_INFO[deviceKey];

    if (deviceInfo && deviceInfo.desc) {
        return deviceInfo.desc;
    }

    return null;
}

// UI State functions
function showLoading() {
    elements.loadingState.classList.remove('hidden');
    elements.resultsSection.classList.add('hidden');
    elements.errorState.classList.add('hidden');
    elements.analyzeBtn.disabled = true;
}

function hideLoading() {
    elements.loadingState.classList.add('hidden');
    elements.analyzeBtn.disabled = false;
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorState.classList.remove('hidden');
    elements.resultsSection.classList.add('hidden');
    elements.phonemeDisplay.classList.add('hidden');
}

// Utility: Escape HTML to prevent XSS
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
