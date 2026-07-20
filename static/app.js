// Active simulation settings
let simModel = 'ml'; // Default simulation model
let simInterval = null;

// Initialize events when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // 1. Set up form sliders to update their label indicators dynamically
    setupSliders();

    // 2. Set up form submit handler for custom inference
    const predictorForm = document.getElementById('predictor-form');
    if (predictorForm) {
        predictorForm.addEventListener('submit', handleInferenceSubmit);
    }

    // 3. Set up simulation switch listener
    const simSwitch = document.getElementById('sim-switch');
    if (simSwitch) {
        simSwitch.addEventListener('change', (e) => {
            if (e.target.checked) {
                startSimulation();
            } else {
                stopSimulation();
            }
        });
    }

    // Start simulation initially
    startSimulation();
});

// Setup range sliders to update their values on the screen
function setupSliders() {
    const sliders = [
        { id: 'input-density', valId: 'val-density' },
        { id: 'input-speed', valId: 'val-speed' },
        { id: 'input-flow', valId: 'val-flow' },
        { id: 'input-visibility', valId: 'val-visibility' },
        { id: 'input-temp', valId: 'val-temp' },
        { id: 'input-latency', valId: 'val-latency' },
        { id: 'input-packet_loss', valId: 'val-packet_loss' },
        { id: 'input-hour', valId: 'val-hour' }
    ];

    sliders.forEach(sliderDef => {
        const slider = document.getElementById(sliderDef.id);
        const valLabel = document.getElementById(sliderDef.valId);
        if (slider && valLabel) {
            slider.addEventListener('input', (e) => {
                valLabel.textContent = e.target.value;
            });
        }
    });
}

// Handle tab switching in analytics section
function openTab(evt, tabId) {
    // Hide all tab contents
    const tabContents = document.getElementsByClassName('tab-content');
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }

    // Remove active class from all tab links
    const tabLinks = document.getElementsByClassName('tab-link');
    for (let i = 0; i < tabLinks.length; i++) {
        tabLinks[i].classList.remove('active');
    }

    // Show current tab content, add active class to clicked tab link
    document.getElementById(tabId).classList.add('active');
    evt.currentTarget.classList.add('active');
}

// Start simulation looping
function startSimulation() {
    // Fetch initial tick immediately
    fetchSimulationTick();

    // Set interval loop
    if (simInterval) clearInterval(simInterval);
    simInterval = setInterval(fetchSimulationTick, 4000);
    console.log("Simulation stream active");
}

// Stop simulation looping
function stopSimulation() {
    if (simInterval) {
        clearInterval(simInterval);
        simInterval = null;
    }
    console.log("Simulation stream paused");
}

// Switch active simulation engine between ML and DL
function setSimModel(modelType) {
    simModel = modelType;
    document.getElementById('toggle-sim-ml').classList.toggle('active', modelType === 'ml');
    document.getElementById('toggle-sim-dl').classList.toggle('active', modelType === 'dl');
    
    // Trigger tick immediately to show update matching the new model
    fetchSimulationTick();
}

// Perform a single simulation step
async function fetchSimulationTick() {
    try {
        const response = await fetch('/api/simulation-data');
        if (!response.ok) throw new Error("Simulation data retrieval failed");
        
        const data = await response.json();
        
        // Update dashboard KPI fields
        updateKPI('live-density', data.density, 'density-status', getDensityStatus(data.density));
        updateKPI('live-speed', data.speed, 'speed-status', getSpeedStatus(data.speed));
        updateKPI('live-flow', data.flow, 'flow-status', getFlowStatus(data.flow));
        updateKPI('live-latency', data.latency, 'latency-status', getLatencyStatus(data.latency));
        
        // Update telemetry side panel
        document.getElementById('telemetry-weather').textContent = data.weather;
        document.getElementById('telemetry-weather').className = `badge ${data.weather.toLowerCase()}`;
        document.getElementById('telemetry-temp').textContent = `${data.temp} °C`;
        document.getElementById('telemetry-visibility').textContent = `${data.visibility} m`;
        document.getElementById('telemetry-time').textContent = `${String(data.hour).padStart(2, '0')}:00 (${data.day})`;
        document.getElementById('telemetry-peak').textContent = data.peak_hour ? 'YES' : 'NO';
        document.getElementById('telemetry-peak').className = `badge ${data.peak_hour ? 'rainy' : 'clear'}`;
        document.getElementById('telemetry-loss').textContent = `${data.packet_loss} %`;

        // Request prediction using the simulation data
        requestSimulationPrediction(data);

    } catch (err) {
        console.error("Simulation error:", err);
    }
}

// Update KPI text and subtext status
function updateKPI(valId, val, statusId, statusText) {
    const valElem = document.getElementById(valId);
    const statusElem = document.getElementById(statusId);
    if (valElem) valElem.textContent = val;
    if (statusElem) statusElem.textContent = statusText;
}

// Helper threshold texts
function getDensityStatus(density) {
    if (density < 50) return "Free flow density";
    if (density < 130) return "Moderate traffic density";
    return "High density congestion";
}

function getSpeedStatus(speed) {
    if (speed > 70) return "High speed circulation";
    if (speed > 35) return "Normal driving speed";
    return "Slow traffic speed";
}

function getFlowStatus(flow) {
    if (flow < 200) return "Light vehicle volume";
    if (flow < 800) return "Moderate vehicle flow";
    return "Heavy traffic volume";
}

function getLatencyStatus(latency) {
    if (latency < 30) return "Low response time";
    if (latency < 90) return "Normal response delay";
    return "High connection latency";
}

// Call inference endpoint for simulation feed
async function requestSimulationPrediction(simData) {
    try {
        const payload = {
            model_type: simModel,
            density: simData.density,
            speed: simData.speed,
            flow: simData.flow,
            weather: simData.weather,
            temp: simData.temp,
            visibility: simData.visibility,
            signal: simData.signal,
            latency: simData.latency,
            packet_loss: simData.packet_loss,
            hour: simData.hour,
            day: simData.day,
            peak_hour: simData.peak_hour
        };

        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Inference failed");
        const out = await res.json();
        
        // Update decision indicator state
        const pulse = document.getElementById('decision-pulse');
        const textLabel = document.getElementById('decision-text');
        
        if (pulse && textLabel) {
            pulse.className = 'pulse-circle'; // reset
            
            const predLower = out.prediction.toLowerCase();
            pulse.classList.add(predLower);
            
            textLabel.textContent = out.prediction;
            // set style color of text to match prediction status
            if (predLower === 'low') textLabel.style.color = 'var(--color-success)';
            else if (predLower === 'medium') textLabel.style.color = 'var(--color-warning)';
            else if (predLower === 'high') textLabel.style.color = 'var(--color-danger)';
        }

        // Update probability bars in simulation section
        updateProbBars({
            low: out.probabilities['Low'] || 0,
            med: out.probabilities['Medium'] || 0,
            high: out.probabilities['High'] || 0
        }, 'sim');

    } catch (err) {
        console.error("Decision engine API error:", err);
    }
}

// Update probability visualizer rows
function updateProbBars(probs, prefix) {
    const lowBar = document.getElementById(`${prefix}-prob-low`);
    const medBar = document.getElementById(`${prefix}-prob-med`);
    const highBar = document.getElementById(`${prefix}-prob-high`);

    const lowLabel = document.getElementById(`${prefix}-val-low`);
    const medLabel = document.getElementById(`${prefix}-val-med`);
    const highLabel = document.getElementById(`${prefix}-val-high`);

    if (lowBar) lowBar.style.width = `${probs.low}%`;
    if (medBar) medBar.style.width = `${probs.med}%`;
    if (highBar) highBar.style.width = `${probs.high}%`;

    if (lowLabel) lowLabel.textContent = `${probs.low}%`;
    if (medLabel) medLabel.textContent = `${probs.med}%`;
    if (highLabel) highLabel.textContent = `${probs.high}%`;
}

// Handle Form Submission for custom inference playground
async function handleInferenceSubmit(e) {
    e.preventDefault();
    
    const startTime = performance.now();

    // Get variables from UI
    const modelType = document.getElementById('pred-model-type').value;
    const density = parseFloat(document.getElementById('input-density').value);
    const speed = parseFloat(document.getElementById('input-speed').value);
    const flow = parseFloat(document.getElementById('input-flow').value);
    const visibility = parseFloat(document.getElementById('input-visibility').value);
    const temp = parseFloat(document.getElementById('input-temp').value);
    const latency = parseFloat(document.getElementById('input-latency').value);
    const packet_loss = parseFloat(document.getElementById('input-packet_loss').value);
    const hour = parseInt(document.getElementById('input-hour').value);
    const weather = document.getElementById('input-weather').value;
    const day = document.getElementById('input-day').value;
    const signal = parseFloat(document.getElementById('input-signal').value);
    const peak_hour = document.getElementById('input-peak').checked ? 1 : 0;

    const payload = {
        model_type: modelType,
        density,
        speed,
        flow,
        weather,
        temp,
        visibility,
        signal,
        latency,
        packet_loss,
        hour,
        day,
        peak_hour
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Inference server error");
        }

        const data = await response.json();
        
        // Show results box and hide placeholder
        document.getElementById('playground-result-empty').classList.add('hidden');
        document.getElementById('playground-result-active').classList.remove('hidden');

        // Update predictions labels
        const playPred = document.getElementById('play-prediction');
        playPred.textContent = data.prediction;
        playPred.className = data.prediction.toLowerCase(); // add color styling class

        document.getElementById('play-confidence').textContent = `${data.confidence}%`;
        document.getElementById('play-engine').textContent = data.model_used;

        // Latency timing calculation
        const duration = (performance.now() - startTime).toFixed(1);
        document.getElementById('play-latency-ms').textContent = `~${duration} ms`;

        // Update probability progress bars
        updateProbBars({
            low: data.probabilities['Low'] || 0,
            med: data.probabilities['Medium'] || 0,
            high: data.probabilities['High'] || 0
        }, 'play');

    } catch (err) {
        alert("Inference Error: " + err.message);
    }
}
