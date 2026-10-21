/**
 * K3 Smart Helmet Surveillance Frontend Application
 * Handles Real-time Webcam streaming, Web Audio hazard alarm,
 * Image upload with DIP visualizer, and live audit logging.
 */

// Global State
let isWebcamRunning = false;
let webcamStream = null;
let videoElement = null;
let outputCanvas = null;
let canvasCtx = null;
let processingCanvas = null;
let procCtx = null;
let soundEnabled = true;
let lastAlertTime = 0;
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;
let isProcessingFrame = false;

// Audio Context for Alert Synthesizer
let audioCtx = null;

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
        canvasCtx = outputCanvas.getContext('2d');
    }

    // Hidden processing canvas
    processingCanvas = document.createElement('canvas');
    procCtx = processingCanvas.getContext('2d');

    // Button event listeners
    const btnStartCam = document.getElementById('btn-start-cam');
    const btnStopCam = document.getElementById('btn-stop-cam');
    const btnResumeCam = document.getElementById('btn-resume-cam');
    const btnToggleSound = document.getElementById('btn-toggle-sound');
    const btnClearLogs = document.getElementById('btn-clear-logs');
    const btnExportCsv = document.getElementById('btn-export-csv');

    if (btnStartCam) btnStartCam.addEventListener('click', startWebcam);
    if (btnStopCam) btnStopCam.addEventListener('click', stopWebcam);
    if (btnResumeCam) btnResumeCam.addEventListener('click', startWebcam);

    if (btnToggleSound) {
        btnToggleSound.addEventListener('click', () => {
            soundEnabled = !soundEnabled;
            const soundIcon = document.getElementById('sound-icon');
            if (soundEnabled) {
                btnToggleSound.innerHTML = '<i class="bi bi-volume-up-fill" id="sound-icon"></i> Alarm Suara: ON';
                btnToggleSound.className = 'btn btn-sm btn-outline-warning';
                initAudio();
            } else {
                btnToggleSound.innerHTML = '<i class="bi bi-volume-mute-fill" id="sound-icon"></i> Alarm Suara: OFF';
                btnToggleSound.className = 'btn btn-sm btn-outline-secondary';
            }
        });
    }

    if (btnClearLogs) {
        btnClearLogs.addEventListener('click', async () => {
            await fetch('/api/audit/clear', { method: 'POST' });
            loadAuditLogs();
        });
    }

    if (btnExportCsv) {
        btnExportCsv.addEventListener('click', exportAuditCsv);
    }

    // File Upload Setup
    setupUploadHandlers();
    
    // Initial Load
    loadAuditLogs();
});

// Webcam Controls
async function startWebcam() {
    initAudio();
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });

        videoElement.srcObject = webcamStream;
        await videoElement.play();

        isWebcamRunning = true;
        document.getElementById('cam-placeholder').classList.add('d-none');
        document.getElementById('btn-start-cam').classList.add('d-none');
        document.getElementById('btn-stop-cam').classList.remove('d-none');
        document.getElementById('btn-resume-cam').classList.add('d-none');
        document.getElementById('btn-snapshot-cam').classList.remove('d-none');

        // Set canvas dimensions
        outputCanvas.width = videoElement.videoWidth || 640;
        outputCanvas.height = videoElement.videoHeight || 480;
        processingCanvas.width = 640;
        processingCanvas.height = 480;

        requestAnimationFrame(processWebcamLoop);
    } catch (err) {
        alert('Gagal mengakses kamera: ' + err.message + '\nPastikan izin kamera telah diberikan pada browser.');
    }
}

function stopWebcam() {
    isWebcamRunning = false;
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    document.getElementById('cam-placeholder').classList.remove('d-none');
    document.getElementById('btn-start-cam').classList.remove('d-none');
    document.getElementById('btn-stop-cam').classList.add('d-none');
    document.getElementById('btn-resume-cam').classList.add('d-none');
    document.getElementById('btn-snapshot-cam').classList.add('d-none');
    document.getElementById('live-alert-banner').classList.add('d-none');
}

// Webcam Inference Loop
async function processWebcamLoop() {
    if (!isWebcamRunning) return;

    if (!isProcessingFrame && videoElement.readyState >= 2) {
        isProcessingFrame = true;

        // Draw video frame to processing canvas
        procCtx.drawImage(videoElement, 0, 0, processingCanvas.width, processingCanvas.height);
        const dataUrl = processingCanvas.toDataURL('image/jpeg', 0.70);

        try {
            const resp = await fetch('/api/detect/frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl, conf: 0.35 })
            });

            if (resp.ok) {
                const res = await resp.json();
                if (res.status === 'success' && res.annotated_frame) {
                    const img = new Image();
                    img.onload = () => {
                        outputCanvas.width = img.width;
                        outputCanvas.height = img.height;
                        canvasCtx.drawImage(img, 0, 0);
                    };
                    img.src = res.annotated_frame;

                    // Update Top Stats
                    updateStatsDisplay(res.stats);

                    // Hazard alert check
                    const banner = document.getElementById('live-alert-banner');
                    if (res.has_violation) {
                        banner.classList.remove('d-none');
                        const vTypes = res.violation_types.join(', ');
                        document.getElementById('live-alert-text').innerText = `PERINGATAN: ${vTypes.toUpperCase()}`;
                        playHazardChime();
                    } else {
                        banner.classList.add('d-none');
                    }
                }
            }
        } catch (err) {
            console.error('Frame inference error:', err);
        } finally {
            isProcessingFrame = false;
        }

        // FPS calculation
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {
            fps = frameCount;
            frameCount = 0;
            lastFrameTime = now;
            const fpsEl = document.getElementById('cam-fps');
            if (fpsEl) fpsEl.innerText = `${fps} FPS`;
        }
    }

    if (isWebcamRunning) {
        requestAnimationFrame(processWebcamLoop);
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

// File Upload & Deep Image Processing Inspection
function setupUploadHandlers() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('image-file-input');

    if (!dropZone || !fileInput) return;

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

    // UI Loading state
    const resultWrapper = document.getElementById('image-result-wrapper');
    const summaryText = document.getElementById('detection-summary-text');
    summaryText.innerText = 'Memproses citra dengan YOLO11 & Pipeline Pengolahan Citra...';
    resultWrapper.classList.remove('d-none');

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
            summaryText.innerText = `Terdeteksi: ${data.stats.total_workers} pekerja (${data.stats.compliant} Aman, ${data.stats.violations} Pelanggaran)`;

            // Render Deep Image Processing Step-by-Step Breakdown
            renderDIPBreakdown(data.detections);

            // Refresh Audit Logs
            loadAuditLogs();
        }
    } catch (err) {
        alert('Gagal memproses gambar: ' + err.message);
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
        const d = det.diagnostics;
        const cardHtml = `
            <div class="card mb-3 bg-dark border border-secondary shadow-sm">
                <div class="card-header bg-subtle py-2 d-flex justify-content-between align-items-center">
                    <span class="badge" style="background-color: ${det.color_hex}">${det.badge}</span>
                    <span class="fw-bold small text-light">Pekerja #${idx + 1}: ${det.title}</span>
                    <span class="badge bg-secondary font-monospace">${det.confidence}%</span>
                </div>
                <div class="card-body p-3">
                    <div class="row g-2 mb-3">
                        <!-- Step 1: Original Crop -->
                        <div class="col-3">
                            <div class="diag-card p-1 text-center">
                                <img src="${d.original_crop}" class="diag-thumb rounded mb-1">
                                <div class="small fw-semibold text-secondary" style="font-size:0.7rem">1. Crop Kepala</div>
                            </div>
                        </div>
                        <!-- Step 2: CLAHE Enhancement (Slide P03) -->
                        <div class="col-3">
                            <div class="diag-card p-1 text-center">
                                <img src="${d.enhanced_roi}" class="diag-thumb rounded mb-1">
                                <div class="small fw-semibold text-info" style="font-size:0.7rem">2. CLAHE (P03)</div>
                            </div>
                        </div>
                        <!-- Step 3: HSV Color Mask (Slide P04) -->
                        <div class="col-3">
                            <div class="diag-card p-1 text-center">
                                <img src="${d.color_mask}" class="diag-thumb rounded mb-1">
                                <div class="small fw-semibold text-warning" style="font-size:0.7rem">3. Mask K3 (P04)</div>
                            </div>
                        </div>
                        <!-- Step 4: Canny Edge (Slide P07) -->
                        <div class="col-3">
                            <div class="diag-card p-1 text-center">
                                <img src="${d.canny_edges}" class="diag-thumb rounded mb-1">
                                <div class="small fw-semibold text-danger" style="font-size:0.7rem">4. Tepi Canny (P07)</div>
                            </div>
                        </div>
                    </div>

                    <!-- Metrics Table from DIP Feature Extraction -->
                    <div class="table-responsive">
                        <table class="table table-sm table-dark table-bordered mb-0 small" style="font-size: 0.75rem;">
                            <tbody>
                                <tr>
                                    <td class="text-secondary">Warna Standar K3:</td>
                                    <td class="fw-bold text-light">${det.detected_k3_color} (${det.k3_color_ratio}%)</td>
                                </tr>
                                <tr>
                                    <td class="text-secondary">Densitas Tepi Interior:</td>
                                    <td class="fw-bold text-light">${det.edge_density}% ${det.edge_density < 20 ? '(Permukaan Halus Plastik)' : '(Tekstur Kain Berlipat)'}</td>
                                </tr>
                                <tr>
                                    <td class="text-secondary">Rasio Lipatan Drapery:</td>
                                    <td class="fw-bold text-light">${det.drapery_ratio} ${det.drapery_ratio > 1.25 ? '(Lipatan Vertikal Hijab/Kain)' : '(Bentuk Kubah Rigit)'}</td>
                                </tr>
                                <tr>
                                    <td class="text-secondary">Simetri Geometri:</td>
                                    <td class="fw-bold text-light">${det.symmetry_score} ${det.symmetry_score < 0.68 ? '(Asimetris Visor Topi)' : '(Simetris Melingkar)'}</td>
                                </tr>
                                <tr>
                                    <td class="text-secondary">Bayangan Lidah Topi:</td>
                                    <td class="fw-bold text-light">${det.visor_shadow}% ${det.visor_shadow > 16 ? '(Ciri Khas Topi Berlidah)' : '(Tanpa Bayangan Visor)'}</td>
                                </tr>
                                <tr>
                                    <td class="text-secondary">Pori Ventilasi Sepeda:</td>
                                    <td class="fw-bold text-light">${det.vent_count} celah ${det.vent_count >= 3 ? '(Helm Sepeda Aerodinamis)' : '(Kubah Padat K3)'}</td>
                                </tr>
                                <tr>
                                    <td class="text-secondary">Kaca Visor Helm Motor:</td>
                                    <td class="fw-bold text-light">${det.dark_visor_ratio}% ${det.dark_visor_ratio > 25 ? '(Kaca Visor Helm Motor)' : '(Wajah Terbuka)'}</td>
                                </tr>
                                <tr>
                                    <td class="text-secondary">Rambut Asli (Ubun-Ubun):</td>
                                    <td class="fw-bold text-light">${det.hair_ratio !== undefined ? det.hair_ratio : 0}% ${(det.hair_ratio >= 15) ? '(Rambut Terbuka / Tanpa Helm)' : '(Tertutup Pelindung)'}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
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
                tbody.innerHTML = `<tr><td colspan="5" class="text-center py-4 text-secondary">Belum ada catatan pelanggaran. Area proyek terpantau aman!</td></tr>`;
                return;
            }

            tbody.innerHTML = '';
            data.logs.forEach(log => {
                const tr = `
                    <tr>
                        <td class="font-monospace small px-3">${log.timestamp}</td>
                        <td>
                            <img src="${log.thumbnail}" class="rounded border border-secondary" style="width: 48px; height: 48px; object-fit: cover;">
                        </td>
                        <td class="fw-bold text-light">${log.violation_type}</td>
                        <td><span class="badge" style="background-color: ${log.color_hex}">${log.badge}</span></td>
                        <td class="small text-secondary">${log.details}</td>
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
