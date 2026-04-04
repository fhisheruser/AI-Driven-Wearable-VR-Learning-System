/**
 * Main application logic for the AI-Driven VR Learning System frontend.
 * Handles user input, API communication, and UI updates.
 */

const API_BASE = '';
let renderer = null;
let currentStepIndex = 0;
let currentSteps = [];

// ---- Initialization ----

document.addEventListener('DOMContentLoaded', () => {
    renderer = new SceneRenderer('viz-canvas', 'viz-container');
    checkHealth();
    setupEventListeners();
});

function setupEventListeners() {
    // Send button & Enter key
    document.getElementById('btn-send').addEventListener('click', sendQuery);
    document.getElementById('query-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    // Quick query chips
    document.querySelectorAll('.chip[data-query]').forEach(chip => {
        chip.addEventListener('click', () => {
            document.getElementById('query-input').value = chip.dataset.query;
            sendQuery();
        });
    });

    // Reset button
    document.getElementById('btn-reset').addEventListener('click', resetSession);

    // Viz controls
    document.getElementById('btn-rotate').addEventListener('click', () => {
        const active = renderer.toggleAutoRotate();
        document.getElementById('btn-rotate').classList.toggle('active', active);
    });
    document.getElementById('btn-labels').addEventListener('click', () => {
        const active = renderer.toggleLabels();
        document.getElementById('btn-labels').classList.toggle('active', active);
    });
    document.getElementById('btn-fullscreen').addEventListener('click', () => {
        const container = document.getElementById('viz-container');
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            container.requestFullscreen();
        }
    });

    // Step controls
    document.getElementById('btn-prev-step').addEventListener('click', () => navigateStep(-1));
    document.getElementById('btn-next-step').addEventListener('click', () => navigateStep(1));

    // Feedback stars
    document.querySelectorAll('.star').forEach(star => {
        star.addEventListener('click', () => submitFeedback(parseInt(star.dataset.rating)));
        star.addEventListener('mouseenter', () => highlightStars(parseInt(star.dataset.rating)));
    });
    document.getElementById('feedback-stars').addEventListener('mouseleave', () => highlightStars(0));
}

// ---- API Communication ----

async function checkHealth() {
    const badge = document.getElementById('status-badge');
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        if (data.pipeline_ready) {
            badge.textContent = 'Ready';
            badge.className = 'status-badge ready';
        } else {
            badge.textContent = 'Loading...';
        }
    } catch {
        badge.textContent = 'Offline';
        badge.className = 'status-badge error';
    }
}

async function sendQuery() {
    const input = document.getElementById('query-input');
    const text = input.value.trim();
    if (!text) return;

    // Add user message to chat
    addChatMessage(text, 'user');
    input.value = '';
    input.focus();

    // Disable send button
    const btn = document.getElementById('btn-send');
    btn.disabled = true;
    btn.textContent = '...';

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, learner_id: 'default' }),
        });
        const data = await res.json();
        handleResponse(data);
    } catch (err) {
        addChatMessage('Failed to get response. Is the server running?', 'system');
        console.error('Query error:', err);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Send';
    }
}

function handleResponse(data) {
    // Chat message with explanation
    const explanation = data.knowledge?.explanation || 'No explanation available.';
    addChatMessage(explanation, 'assistant');

    // Update visualization
    if (data.scene) {
        document.getElementById('viz-title').textContent = data.scene.title || '3D Visualization';
        renderer.renderScene(data.scene);

        // Handle steps
        if (data.scene.steps && data.scene.steps.length > 0) {
            currentSteps = data.scene.steps;
            currentStepIndex = 0;
            showStepControls(true);
            updateStepDisplay();
        } else {
            showStepControls(false);
        }
    }

    // Update explanation panel
    document.getElementById('explanation-content').innerHTML =
        `<p>${explanation}</p>` +
        (data.knowledge?.steps?.length
            ? '<ol>' + data.knowledge.steps.map(s => `<li>${s}</li>`).join('') + '</ol>'
            : '');

    // Update learning sequence
    const seqList = document.getElementById('sequence-list');
    seqList.innerHTML = '';
    if (data.knowledge?.pedagogical_sequence) {
        data.knowledge.pedagogical_sequence.forEach((step, i) => {
            const li = document.createElement('li');
            li.textContent = step;
            if (i === 0) li.className = 'active';
            seqList.appendChild(li);
        });
    }

    // Update related topics
    const relatedChips = document.getElementById('related-chips');
    relatedChips.innerHTML = '';
    if (data.knowledge?.related_topics) {
        data.knowledge.related_topics.forEach(topic => {
            const chip = document.createElement('button');
            chip.className = 'chip';
            chip.textContent = topic;
            chip.addEventListener('click', () => {
                document.getElementById('query-input').value = `Explain ${topic}`;
                sendQuery();
            });
            relatedChips.appendChild(chip);
        });
    }

    // Update adaptation metrics
    if (data.adaptation) {
        document.getElementById('metric-difficulty').textContent =
            data.adaptation.content_depth || '--';
        document.getElementById('metric-depth').textContent =
            data.adaptation.visualization_complexity || '--';
        document.getElementById('metric-pacing').textContent =
            data.adaptation.pacing || '--';
    }
    if (data.latency) {
        document.getElementById('metric-latency').textContent =
            Math.round(data.latency.total_ms) + ' ms';
    }

    // Reset feedback stars
    highlightStars(0);
}

async function submitFeedback(rating) {
    highlightStars(rating, true);
    try {
        await fetch(`${API_BASE}/api/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                learner_id: 'default',
                rating: rating,
                engagement: rating / 5.0,
                comprehension_change: (rating - 3) * 0.2,
            }),
        });
    } catch (err) {
        console.error('Feedback error:', err);
    }
}

async function resetSession() {
    try {
        await fetch(`${API_BASE}/api/session/reset?learner_id=default`, { method: 'POST' });
    } catch {}
    // Clear chat
    const chat = document.getElementById('chat-container');
    chat.innerHTML = '';
    addChatMessage('Session reset. Ask a new question!', 'system');
    // Reset viz
    document.getElementById('viz-placeholder').style.display = 'flex';
    document.getElementById('viz-title').textContent = '3D Visualization';
    showStepControls(false);
    document.getElementById('explanation-content').innerHTML =
        '<p class="placeholder-text">Explanation will appear here.</p>';
    document.getElementById('sequence-list').innerHTML = '';
    document.getElementById('related-chips').innerHTML = '';
    ['metric-difficulty', 'metric-depth', 'metric-pacing', 'metric-latency'].forEach(id => {
        document.getElementById(id).textContent = '--';
    });
}

// ---- UI Helpers ----

function addChatMessage(text, role) {
    const chat = document.getElementById('chat-container');
    const div = document.createElement('div');
    div.className = `chat-message ${role}`;
    div.innerHTML = `<p>${text}</p>`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function showStepControls(show) {
    document.getElementById('step-controls').style.display = show ? 'flex' : 'none';
}

function updateStepDisplay() {
    const indicator = document.getElementById('step-indicator');
    indicator.textContent = `Step ${currentStepIndex + 1} / ${currentSteps.length}`;

    const step = currentSteps[currentStepIndex];
    if (step) {
        addChatMessage(`Step ${step.step}: ${step.title} - ${step.description}`, 'system');
    }
}

function navigateStep(direction) {
    const newIndex = currentStepIndex + direction;
    if (newIndex >= 0 && newIndex < currentSteps.length) {
        currentStepIndex = newIndex;
        updateStepDisplay();
    }
}

function highlightStars(upTo, lock = false) {
    document.querySelectorAll('.star').forEach(star => {
        const rating = parseInt(star.dataset.rating);
        star.classList.toggle('active', rating <= upTo);
    });
}
