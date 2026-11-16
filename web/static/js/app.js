/**
 * K3 Smart Helmet Surveillance Frontend Application (Open Design + Mobile Responsive Edition)
 * Features:
 * - Accessible tab navigation with URL hash sync (no Bootstrap JS dependency)
 * - 30-60 FPS Decoupled Smooth Camera Viewfinder (Zero Stutter / Lag)
 * - Dynamic Front & Rear Camera Switching (facingMode: 'environment' vs 'user')
 * - Ultra-lightweight Async Vector AI Loop (~15KB per frame, ~95% bandwidth reduction)
 * - Web Audio dual-tone alarm synthesizer
 * - Deep Image Processing Diagnostic breakdown (Slide BSI P01-P11)
 * - Live Audit log & CSV export
 */

// Global Camera & Inference State
let isWebcamRunning = false;
let webcamStream = null;
let videoElement = null;
let outputCanvas = null;
let canvasCtx = null;
let processingCanvas = null;
let procCtx = null;

// Camera Facing Mode: 'environment' (Belakang) or 'user' (Depan)
let currentFacingMode = 'environment';

// Real-time AI Detections Cache
let activeDetections = [];
let lastDetectionTime = 0;
let isProcessingFrame = false;
let renderRafId = null;

// Sound & Alarm
let soundEnabled = true;
let lastAlertTime = 0;
let audioCtx = null;

// Dual FPS Monitoring (Viewfinder Display FPS & AI Detection FPS)
let displayFps = 0;
let aiFps = 0;
let displayFrameCount = 0;
let aiFrameCount = 0;
let lastFpsUpdateTime = performance.now();

// Map K3 label or color_hex to Open Design semantic status pill class
function getStatusPillClass(label, colorHex, badge) {
    const lbl = String(label || '').toUpperCase();
    const hex = String(colorHex || '').toLowerCase();
    const bdg = String(badge || '').toUpperCase();

    if (lbl === 'REAL_HELMET' || hex === '#10b981' || bdg === 'COMPLIANT') {
        return 'status-pill--compliant';
    }
    if (lbl === 'HELM_MOTOR' || hex === '#06b6d4') {
        return 'status-pill--cyan';
    }
    if (lbl === 'KERUDUNG' || hex === '#8b5cf6') {
        return 'status-pill--purple';
    }
    if (lbl === 'NO_HELMET' || hex === '#ef4444' || bdg === 'DANGER' || bdg === 'BAHAYA') {
        return 'status-pill--danger';
    }
    return 'status-pill--warning';
}

// Accessible Tab Switching + URL Hash Persistence
function setupTabNavigation() {
    const tabButtons = document.querySelectorAll('#mainTab [data-bs-toggle="pill"]');
    const tabPanes = document.querySelectorAll('#mainTabContent .tab-pane');
    const brandLink = document.querySelector('.brand');

    function activateTab(targetSelector, updateUrl = true) {
        const targetPane = document.querySelector(targetSelector);
        if (!targetPane) return;

        tabButtons.forEach(btn => {
            const isTarget = btn.getAttribute('data-bs-target') === targetSelector;
            btn.classList.toggle('active', isTarget);
            btn.setAttribute('aria-selected', isTarget ? 'true' : 'false');
        });

        tabPanes.forEach(pane => {
            const isTarget = `#${pane.id}` === targetSelector;
            pane.classList.toggle('active', isTarget);
        });

        if (updateUrl && window.location.hash !== targetSelector) {
            try {
                history.replaceState(null, '', targetSelector);
            } catch (_) {
                // Ignore history restriction on file:// or sandboxed frames
            }
        }
    }

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-bs-target');
            if (target) activateTab(target, true);
        });
    });

    if (brandLink) {
        brandLink.addEventListener('click', (e) => {
            e.preventDefault();
            activateTab('#tab-webcam', true);
        });
    }

    const initialHash = window.location.hash;
    if (initialHash && document.querySelector(`#mainTab [data-bs-target="${initialHash}"]`)) {
        activateTab(initialHash, false);
    }
}

// Audio Synthesizer Initialization
function initAudio() {
    if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
            audioCtx = new AudioContextClass();
        }
    }
}

function playHazardChime() {
    if (!soundEnabled) return;
    const now = Date.now();
    // Throttle alarm sound to at most once every 3.5 seconds
    if (now - lastAlertTime < 3500) return;
    lastAlertTime = now;

    try {
        initAudio();
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        if (!audioCtx) return;

        // Dual-tone warning beep (880Hz then 660Hz)
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime);
        osc.frequency.setValueAtTime(660, audioCtx.currentTime + 0.15);

        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);

        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.35);
    } catch (e) {
        console.warn('Audio play failed:', e);
    }
}

// Clock Display
function updateClock() {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    const clockEl = document.getElementById('clock-display');
    if (clockEl) clockEl.innerText = timeStr;
}
setInterval(updateClock, 1000);
updateClock();

// DOM Initialization
document.addEventListener('DOMContentLoaded', () => {
    videoElement = document.getElementById('webcam-video');
    outputCanvas = document.getElementById('output-canvas');
    if (outputCanvas) {
        canvasCtx = outputCanvas.getContext('2d', { alpha: false });
    }

    // Hidden lightweight processing canvas for AI downscaling
    processingCanvas = document.createElement('canvas');
    procCtx = processingCanvas.getContext('2d');

    // Initialize tab navigation
    setupTabNavigation();

    // Button event listeners
    const btnStartCam = document.getElementById('btn-start-cam');
    const btnStopCam = document.getElementById('btn-stop-cam');
    const btnResumeCam = document.getElementById('btn-resume-cam');
    const btnSwitchCam = document.getElementById('btn-switch-cam');
    const btnSwitchCamInit = document.getElementById('btn-switch-cam-init');
    const btnSnapshotCam = document.getElementById('btn-snapshot-cam');
    const btnToggleSound = document.getElementById('btn-toggle-sound');
    const btnClearLogs = document.getElementById('btn-clear-logs');
    const btnExportCsv = document.getElementById('btn-export-csv');

    if (btnStartCam) btnStartCam.addEventListener('click', startWebcam);
    if (btnStopCam) btnStopCam.addEventListener('click', stopWebcam);
    if (btnResumeCam) btnResumeCam.addEventListener('click', startWebcam);

    // Camera Switch Event Handlers
    if (btnSwitchCam) btnSwitchCam.addEventListener('click', toggleCamera);
    if (btnSwitchCamInit) btnSwitchCamInit.addEventListener('click', toggleCamera);

    // Snapshot
    if (btnSnapshotCam) btnSnapshotCam.addEventListener('click', takeWebcamSnapshot);

    // Sound Toggle
    if (btnToggleSound) {
        btnToggleSound.setAttribute('aria-pressed', 'true');
        btnToggleSound.addEventListener('click', () => {
            soundEnabled = !soundEnabled;
            btnToggleSound.setAttribute('aria-pressed', soundEnabled ? 'true' : 'false');
            if (soundEnabled) {
                btnToggleSound.innerHTML = '<i class="bi bi-volume-up-fill" id="sound-icon" aria-hidden="true"></i><span>Alarm suara: ON</span>';
                initAudio();
            } else {
                btnToggleSound.innerHTML = '<i class="bi bi-volume-mute-fill" id="sound-icon" aria-hidden="true"></i><span>Alarm suara: OFF</span>';
            }
        });
    }

    // Audit Log Controls
    if (btnClearLogs) {
        btnClearLogs.addEventListener('click', async () => {
            await fetch('/api/audit/clear', { method: 'POST' });
            loadAuditLogs();
        });
    }

    if (btnExportCsv) {
        btnExportCsv.addEventListener('click', exportAuditCsv);
    }

    // Setup initial camera labels
    updateCameraLabels();

    // File Upload Setup
    setupUploadHandlers();

    // Initial Load of Audit Logs
    loadAuditLogs();
});

// Update Camera Mode UI Badges & Buttons
function updateCameraLabels() {
    const isRear = currentFacingMode === 'environment';
    const modeName = isRear ? 'Kamera Belakang' : 'Kamera Depan';
    const nextModeName = isRear ? 'Kamera Depan' : 'Kamera Belakang';

    const textInit = document.getElementById('cam-mode-text-init');
    if (textInit) textInit.innerText = `Mode: ${modeName}`;

    const textActive = document.getElementById('cam-mode-text');
    if (textActive) textActive.innerText = `Ganti: ${nextModeName}`;

    const badgeFacing = document.getElementById('cam-facing-badge');
    if (badgeFacing) {
        badgeFacing.innerText = isRear ? 'Belakang' : 'Depan';
        badgeFacing.className = isRear
            ? 'status-pill status-pill--stage'
            : 'status-pill status-pill--stage-info';
    }
}

// Toggle Front/Back Camera
async function toggleCamera() {
    currentFacingMode = (currentFacingMode === 'environment') ? 'user' : 'environment';
    updateCameraLabels();

    if (isWebcamRunning) {
        // Stop current stream tracks only
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        await startWebcam();
    }
}

// Start Webcam Stream
async function startWebcam() {
    initAudio();
    try {
        const constraints = {
            video: {
                facingMode: { ideal: currentFacingMode },
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        };

        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = webcamStream;
        await videoElement.play();

        isWebcamRunning = true;
        isProcessingFrame = false;
        activeDetections = [];

        // UI State Updates
        document.getElementById('cam-placeholder').classList.add('d-none');
        if (outputCanvas) outputCanvas.classList.remove('d-none');
        document.getElementById('btn-start-cam').classList.add('d-none');
        document.getElementById('btn-stop-cam').classList.remove('d-none');
        document.getElementById('btn-resume-cam').classList.add('d-none');
        document.getElementById('btn-switch-cam').classList.remove('d-none');
        document.getElementById('btn-snapshot-cam').classList.remove('d-none');

        updateCameraLabels();

        // Launch decoupled loops:
        // 1. High-frequency 60 FPS Viewfinder Canvas Render Loop
        if (renderRafId) cancelAnimationFrame(renderRafId);
        renderRafId = requestAnimationFrame(renderCanvasLoop);

        // 2. Async Lightweight Background AI Inference Loop
        sendInferenceFrame();

    } catch (err) {
        alert('Gagal mengakses kamera: ' + err.message + '\nPastikan izin akses kamera telah diizinkan pada browser Anda.');
    }
}

// Stop Webcam Stream
function stopWebcam() {
    isWebcamRunning = false;
    isProcessingFrame = false;
    if (renderRafId) {
        cancelAnimationFrame(renderRafId);
        renderRafId = null;
    }

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    activeDetections = [];

    // Clear output canvas and hide until camera starts again
    if (canvasCtx && outputCanvas) {
        canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        outputCanvas.classList.add('d-none');
    }

    // UI Updates
    document.getElementById('cam-placeholder').classList.remove('d-none');
    document.getElementById('btn-start-cam').classList.remove('d-none');
    document.getElementById('btn-stop-cam').classList.add('d-none');
    document.getElementById('btn-resume-cam').classList.add('d-none');
    document.getElementById('btn-switch-cam').classList.add('d-none');
    document.getElementById('btn-snapshot-cam').classList.add('d-none');
    document.getElementById('live-alert-banner').classList.add('d-none');

    const fpsEl = document.getElementById('cam-fps');
    if (fpsEl) fpsEl.innerText = '0 FPS';
}

// Take Snapshot from Canvas
function takeWebcamSnapshot() {
    if (!outputCanvas || !isWebcamRunning) return;
    try {
        const link = document.createElement('a');
        link.download = `k3_snapshot_${Date.now()}.jpg`;
        link.href = outputCanvas.toDataURL('image/jpeg', 0.92);
        link.click();
    } catch (e) {
        console.error('Snapshot failed:', e);
    }
}

// -------------------------------------------------------------
// LOOP 1: High-Frequency Smooth Viewfinder (30 - 60 FPS)
// -------------------------------------------------------------
function renderCanvasLoop() {
    if (!isWebcamRunning) return;

    if (videoElement && videoElement.readyState >= 2) {
        const vw = videoElement.videoWidth;
        const vh = videoElement.videoHeight;

        if (vw > 0 && vh > 0) {
            if (outputCanvas.width !== vw || outputCanvas.height !== vh) {
                outputCanvas.width = vw;
                outputCanvas.height = vh;
            }

            canvasCtx.save();
            // If Front/Selfie Camera, mirror horizontally for natural preview
            if (currentFacingMode === 'user') {
                canvasCtx.translate(vw, 0);
                canvasCtx.scale(-1, 1);
            }
            canvasCtx.drawImage(videoElement, 0, 0, vw, vh);
            canvasCtx.restore();

            // Overlay active AI bounding boxes on top of live video
            drawActiveDetections(canvasCtx, vw, vh);
        }
    }

    displayFrameCount++;
    updateFpsDisplay();

    renderRafId = requestAnimationFrame(renderCanvasLoop);
}

// Render AI Bounding Boxes & Badges over Live Canvas
function drawActiveDetections(ctx, vw, vh) {
    if (!activeDetections || activeDetections.length === 0) return;

    // Fade out detection bounding boxes if no update received in 1.8 seconds
    const timeSinceLast = performance.now() - lastDetectionTime;
    if (timeSinceLast > 1800) return;

    ctx.save();
    activeDetections.forEach(det => {
        let [x1, y1, x2, y2] = det.scaledBbox || [0, 0, 0, 0];

        // If front camera, invert X coordinates to match mirrored canvas video
        if (currentFacingMode === 'user') {
            const origX1 = x1;
            const origX2 = x2;
            x1 = vw - origX2;
            x2 = vw - origX1;
        }

        const bw = x2 - x1;
        const bh = y2 - y1;
        const color = det.color_hex || '#10b981';

        // 1. Draw Bounding Box
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(3, Math.round(vw / 180));
        ctx.strokeRect(x1, y1, bw, bh);

        // 2. Draw Top Label Pill
        const labelText = `${det.title || det.label} (${det.confidence}%)`;
        ctx.font = '600 13px "Red Hat Text", sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        const badgeH = 24;
        const badgeW = textWidth + 18;
        const badgeY = Math.max(4, y1 - badgeH - 4);

        // Badge Background Pill
        ctx.fillStyle = color;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x1, badgeY, badgeW, badgeH, 6);
        } else {
            ctx.rect(x1, badgeY, badgeW, badgeH);
        }
        ctx.fill();

        // Badge Text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x1 + 9, badgeY + 16);
    });
    ctx.restore();
}

// -------------------------------------------------------------
// LOOP 2: Ultra-Fast Async AI Inference (~15KB per request)
// -------------------------------------------------------------
async function sendInferenceFrame() {
    if (!isWebcamRunning) return;
    if (isProcessingFrame) return;

    if (!videoElement || videoElement.readyState < 2) {
        setTimeout(sendInferenceFrame, 60);
        return;
    }

    isProcessingFrame = true;

    // Downscale frame to ~384px width for blazing fast mobile bandwidth & YOLO inference
    const targetW = 384;
    const aspect = (videoElement.videoHeight || 480) / (videoElement.videoWidth || 640);
    const targetH = Math.round(targetW * aspect);

    if (processingCanvas.width !== targetW || processingCanvas.height !== targetH) {
        processingCanvas.width = targetW;
        processingCanvas.height = targetH;
    }

    // Capture small frame
    procCtx.drawImage(videoElement, 0, 0, targetW, targetH);
    const dataUrl = processingCanvas.toDataURL('image/jpeg', 0.55);

    try {
        const resp = await fetch('/api/detect/frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: dataUrl,
                conf: 0.35,
                send_annotated: false // Saves ~80KB response payload per frame!
            })
        });

        if (resp.ok) {
            const res = await resp.json();
            if (res.status === 'success') {
                const fW = res.frame_size ? res.frame_size[0] : targetW;
                const fH = res.frame_size ? res.frame_size[1] : targetH;
                const scaleX = (outputCanvas.width || targetW) / fW;
                const scaleY = (outputCanvas.height || targetH) / fH;

                // Scale bounding boxes accurately to current canvas viewport
                activeDetections = (res.detections || []).map(d => {
                    const [bx1, by1, bx2, by2] = d.bbox;
                    return {
                        ...d,
                        scaledBbox: [
                            bx1 * scaleX,
                            by1 * scaleY,
                            bx2 * scaleX,
                            by2 * scaleY
                        ]
                    };
                });

                lastDetectionTime = performance.now();
                updateStatsDisplay(res.stats);

                // Hazard Alert Check
                const banner = document.getElementById('live-alert-banner');
                if (res.has_violation) {
                    banner.classList.remove('d-none');
                    const vTypes = (res.violation_types || []).join(', ');
                    document.getElementById('live-alert-text').innerText = `Peringatan: ${vTypes}`;
                    playHazardChime();
                } else {
                    banner.classList.add('d-none');
                }

                aiFrameCount++;
            }
        }
    } catch (err) {
        console.warn('Inference error:', err);
    } finally {
        isProcessingFrame = false;
        // Schedule next AI frame with minimal delay
        if (isWebcamRunning) {
            requestAnimationFrame(sendInferenceFrame);
        }
    }
}

// Dual FPS Monitoring Display
function updateFpsDisplay() {
    const now = performance.now();
    if (now - lastFpsUpdateTime >= 1000) {
        displayFps = displayFrameCount;
        aiFps = aiFrameCount;
        displayFrameCount = 0;
        aiFrameCount = 0;
        lastFpsUpdateTime = now;

        const fpsEl = document.getElementById('cam-fps');
        if (fpsEl) {
            fpsEl.innerText = `${displayFps} FPS (AI: ${aiFps}/s)`;
        }
    }
}

// Update Dashboard Statistics Cards
function updateStatsDisplay(stats) {
    if (!stats) return;
    const elWorkers = document.getElementById('stat-total-workers');
    const elCompliant = document.getElementById('stat-compliant');
    const elViolations = document.getElementById('stat-violations');
    const elRate = document.getElementById('stat-compliance-rate');

    if (elWorkers) elWorkers.innerText = stats.total_workers || 0;
    if (elCompliant) elCompliant.innerText = stats.compliant || 0;
    if (elViolations) elViolations.innerText = stats.violations || 0;
    if (elRate) elRate.innerText = `${stats.compliance_rate || 100}%`;
}

// -------------------------------------------------------------
// SECTION 2: File Upload & Deep Image Processing Inspection
// -------------------------------------------------------------
function setupUploadHandlers() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('image-file-input');
    const btnChoose = document.getElementById('btn-choose-file');

    if (!dropZone || !fileInput) return;

    if (btnChoose) {
        btnChoose.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    dropZone.addEventListener('click', (e) => {
        if (e.target !== btnChoose) {
            fileInput.click();
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageUpload(e.target.files[0]);
        }
    });
}

async function handleImageUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf', '0.35');

    const resultWrapper = document.getElementById('image-result-wrapper');
    const summaryText = document.getElementById('detection-summary-text');
    const btnChoose = document.getElementById('btn-choose-file');

    summaryText.innerText = 'Memproses citra dengan YOLO11n dan pipeline pengolahan citra...';
    resultWrapper.classList.remove('d-none');
    if (btnChoose) btnChoose.disabled = true;

    try {
        const resp = await fetch('/api/detect/image', {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            throw new Error(`Server returned ${resp.status}`);
        }

        const data = await resp.json();
        if (data.status === 'success') {
            document.getElementById('annotated-image-result').src = data.annotated_image;
            updateStatsDisplay(data.stats);
            summaryText.innerText = `Terdeteksi: ${data.stats.total_workers} pekerja (${data.stats.compliant} aman, ${data.stats.violations} pelanggaran)`;

            // Render Deep Image Processing Step-by-Step Breakdown
            renderDIPBreakdown(data.detections);

            // Refresh Audit Logs
            loadAuditLogs();
        }
    } catch (err) {
        summaryText.innerText = 'Gagal memproses gambar. Periksa format file lalu coba lagi.';
        alert('Gagal memproses gambar: ' + err.message);
    } finally {
        if (btnChoose) btnChoose.disabled = false;
    }
}

// Render Diagnostic DIP Cards (Slide P02, P03, P04, P07, P11)
function renderDIPBreakdown(detections) {
    const emptyState = document.getElementById('dip-empty-state');
    const activeContent = document.getElementById('dip-active-content');

    if (!detections || detections.length === 0) {
        emptyState.classList.remove('d-none');
        activeContent.classList.add('d-none');
        return;
    }

    emptyState.classList.add('d-none');
    activeContent.classList.remove('d-none');
    activeContent.innerHTML = '';

    detections.forEach((det, idx) => {
        const d = det.diagnostics || {};
        const pillClass = getStatusPillClass(det.label, det.color_hex, det.badge);
        const hairRatio = det.hair_ratio !== undefined ? det.hair_ratio : 0;

        const cardHtml = `
            <article class="entry">
                <header class="entry-head">
                    <span class="entry-head-title">Pekerja #${idx + 1}: ${det.title}</span>
                    <div class="entry-head-meta">
                        <span class="status-pill ${pillClass}">${det.badge}</span>
                        <span class="status-pill status-pill--stage num">${det.confidence}%</span>
                    </div>
                </header>
                <div class="notebook">
                    <div class="diag-steps-grid">
                        <div class="diag-card">
                            <img src="${d.original_crop || ''}" class="diag-thumb" width="120" height="96" alt="Potongan kepala pekerja ${idx + 1}">
                            <div class="diag-caption">1. Crop Kepala</div>
                        </div>
                        <div class="diag-card">
                            <img src="${d.enhanced_roi || ''}" class="diag-thumb" width="120" height="96" alt="Peningkatan kontras CLAHE pekerja ${idx + 1}">
                            <div class="diag-caption">2. CLAHE (P03)</div>
                        </div>
                        <div class="diag-card">
                            <img src="${d.color_mask || ''}" class="diag-thumb" width="120" height="96" alt="Masker kubah pekerja ${idx + 1}">
                            <div class="diag-caption">3. Mask Kubah (P09)</div>
                        </div>
                        <div class="diag-card">
                            <img src="${d.canny_edges || ''}" class="diag-thumb" width="120" height="96" alt="Deteksi tepi Canny pekerja ${idx + 1}">
                            <div class="diag-caption">4. Tepi Canny (P07)</div>
                        </div>
                    </div>

                    <table class="dip-metrics-table">
                        <tbody>
                            <tr>
                                <td class="dip-metrics-label">Densitas Tepi Interior</td>
                                <td class="dip-metrics-value"><span class="num">${det.edge_density}%</span> ${det.edge_density < 20 ? '(Permukaan Halus Plastik)' : '(Tekstur Kain Berlipat)'}</td>
                            </tr>
                            <tr>
                                <td class="dip-metrics-label">Rasio Lipatan Drapery</td>
                                <td class="dip-metrics-value"><span class="num">${det.drapery_ratio}</span> ${det.drapery_ratio > 1.25 ? '(Lipatan Vertikal Hijab/Kain)' : '(Bentuk Kubah Rigit)'}</td>
                            </tr>
                            <tr>
                                <td class="dip-metrics-label">Simetri Geometri</td>
                                <td class="dip-metrics-value"><span class="num">${det.symmetry_score}</span> ${det.symmetry_score < 0.68 ? '(Asimetris Visor Topi)' : '(Simetris Melingkar)'}</td>
                            </tr>
                            <tr>
                                <td class="dip-metrics-label">Bayangan Lidah Topi</td>
                                <td class="dip-metrics-value"><span class="num">${det.visor_shadow}%</span> ${det.visor_shadow > 16 ? '(Ciri Khas Topi Berlidah)' : '(Tanpa Bayangan Visor)'}</td>
                            </tr>
                            <tr>
                                <td class="dip-metrics-label">Pori Ventilasi Sepeda</td>
                                <td class="dip-metrics-value"><span class="num">${det.vent_count}</span> celah ${det.vent_count >= 3 ? '(Helm Sepeda Aerodinamis)' : '(Kubah Padat K3)'}</td>
                            </tr>
                            <tr>
                                <td class="dip-metrics-label">Kaca Visor Helm Motor</td>
                                <td class="dip-metrics-value"><span class="num">${det.dark_visor_ratio}%</span> ${det.dark_visor_ratio > 25 ? '(Kaca Visor Helm Motor)' : '(Wajah Terbuka)'}</td>
                            </tr>
                            <tr>
                                <td class="dip-metrics-label">Rambut Asli (Ubun-Ubun)</td>
                                <td class="dip-metrics-value"><span class="num">${hairRatio}%</span> ${hairRatio >= 15 ? '(Rambut Terbuka / Tanpa Helm)' : '(Tertutup Pelindung)'}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </article>
        `;
        activeContent.insertAdjacentHTML('beforeend', cardHtml);
    });
}

// Load and Render Audit Logs
async function loadAuditLogs() {
    try {
        const resp = await fetch('/api/audit/logs');
        if (resp.ok) {
            const data = await resp.json();
            const tbody = document.getElementById('audit-table-body');
            if (!tbody) return;

            if (!data.logs || data.logs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="audit-empty-cell">Belum ada catatan pelanggaran. Area proyek terpantau aman.</td></tr>';
                return;
            }

            tbody.innerHTML = '';
            data.logs.forEach(log => {
                const pillClass = getStatusPillClass('', log.color_hex, log.badge);
                const tr = `
                    <tr>
                        <td class="num" data-label="Waktu">${log.timestamp}</td>
                        <td data-label="Bukti snapshot">
                            <img src="${log.thumbnail}" class="audit-thumb" width="48" height="48" alt="Bukti pelanggaran ${log.violation_type}">
                        </td>
                        <td data-label="Jenis pelanggaran"><strong>${log.violation_type}</strong></td>
                        <td data-label="Status K3"><span class="status-pill ${pillClass}">${log.badge}</span></td>
                        <td data-label="Analisis fisik citra">${log.details}</td>
                    </tr>
                `;
                tbody.insertAdjacentHTML('beforeend', tr);
            });
        }
    } catch (e) {
        console.error('Failed to load audit logs:', e);
    }
}

// Export Audit Logs to CSV
async function exportAuditCsv() {
    try {
        const resp = await fetch('/api/audit/logs');
        const data = await resp.json();
        if (!data.logs || data.logs.length === 0) {
            alert('Tidak ada data log untuk diekspor.');
            return;
        }

        let csv = 'Timestamp,Violation Type,Status,Details\n';
        data.logs.forEach(l => {
            csv += `"${l.timestamp}","${l.violation_type}","${l.badge}","${l.details}"\n`;
        });

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `k3_safety_audit_${Date.now()}.csv`;
        a.click();
    } catch (e) {
        alert('Gagal mengekspor CSV: ' + e.message);
    }
}
