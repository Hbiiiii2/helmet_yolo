/**
 * Client-Side YOLO11n (ONNX Runtime Web) + Headwear Classifier
 * Menjalankan deteksi YOLO11n + klasifikasi K3 (Helm Proyek, Hijab, Helm Motor, Helm Sepeda, Topi)
 * langsung di browser user (WebGPU / WASM) tanpa round-trip server.
 */

const BrowserYOLO = (() => {
    const MODEL_URL = '/static/models/best.onnx';
    const INPUT_SIZE = 640;
    const CLASS_NAMES = ['helmet', 'no-helmet', 'no-vest', 'person', 'vest'];

    const STATUS_META = {
        REAL_HELMET: {
            title: 'HELM PROYEK RESMI (K3)',
            badge: 'COMPLIANT',
            color_hex: '#10b981'
        },
        TOPI: {
            title: 'PERINGATAN: HANYA TOPI (BUKAN HELM K3)',
            badge: 'VIOLATION',
            color_hex: '#f59e0b'
        },
        HELM_MOTOR: {
            title: 'PERINGATAN: HELM MOTOR (BUKAN HELM K3)',
            badge: 'VIOLATION',
            color_hex: '#06b6d4'
        },
        HELM_SEPEDA: {
            title: 'PERINGATAN: HELM SEPEDA (BUKAN HELM K3)',
            badge: 'VIOLATION',
            color_hex: '#eab308'
        },
        KERUDUNG: {
            title: 'PERINGATAN: KERUDUNG / HIJAB (WAJIB PAKAI HELM)',
            badge: 'VIOLATION',
            color_hex: '#8b5cf6'
        },
        NO_HELMET: {
            title: 'BAHAYA: TANPA HELM KESELAMATAN',
            badge: 'DANGER',
            color_hex: '#ef4444'
        }
    };

    let session = null;
    let loadPromise = null;
    let activeProvider = 'wasm';
    let letterboxCanvas = null;
    let letterboxCtx = null;
    let inputBuffer = null;

    function ensureBuffers() {
        if (!letterboxCanvas) {
            letterboxCanvas = document.createElement('canvas');
            letterboxCanvas.width = INPUT_SIZE;
            letterboxCanvas.height = INPUT_SIZE;
            letterboxCtx = letterboxCanvas.getContext('2d', { willReadFrequently: true });
        }
        if (!inputBuffer) {
            inputBuffer = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
        }
    }

    async function init() {
        if (session) return session;
        if (loadPromise) return loadPromise;

        loadPromise = (async () => {
            if (typeof ort === 'undefined') {
                throw new Error('onnxruntime-web tidak tersedia');
            }
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
            if (navigator.hardwareConcurrency) {
                ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency);
            }

            // Coba WebGPU dulu untuk kecepatan maksimal, fallback ke WASM
            const providers = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
            for (const ep of providers) {
                try {
                    session = await ort.InferenceSession.create(MODEL_URL, {
                        executionProviders: [ep],
                        graphOptimizationLevel: 'all'
                    });
                    activeProvider = ep;
                    console.log(`[BrowserYOLO] Model dimuat dengan provider: ${ep}`);
                    return session;
                } catch (err) {
                    console.warn(`[BrowserYOLO] Gagal muat dengan ${ep}:`, err);
                }
            }
            throw new Error('Gagal memuat session ONNX di browser');
        })();

        return loadPromise;
    }

    function rgbToHsv(r, g, b) {
        const rn = r / 255, gn = g / 255, bn = b / 255;
        const max = Math.max(rn, gn, bn);
        const min = Math.min(rn, gn, bn);
        const d = max - min;
        let h = 0;
        if (d > 1e-6) {
            if (max === rn) h = ((gn - bn) / d) % 6;
            else if (max === gn) h = (bn - rn) / d + 2;
            else h = (rn - gn) / d + 4;
            h *= 30; // OpenCV scale [0, 180]
            if (h < 0) h += 180;
        }
        const s = max === 0 ? 0 : (d / max) * 255;
        const v = max * 255;
        return [h, s, v];
    }

    function computeIoU(a, b) {
        const x1 = Math.max(a[0], b[0]);
        const y1 = Math.max(a[1], b[1]);
        const x2 = Math.min(a[2], b[2]);
        const y2 = Math.min(a[3], b[3]);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
        const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
        return inter / (areaA + areaB - inter + 1e-6);
    }

    function nms(boxes, iouThreshold) {
        boxes.sort((a, b) => b.conf - a.conf);
        const kept = [];
        for (const box of boxes) {
            let suppress = false;
            for (const k of kept) {
                if (computeIoU(box.bbox, k.bbox) > iouThreshold) {
                    suppress = true;
                    break;
                }
            }
            if (!suppress) kept.push(box);
        }
        return kept;
    }

    function analyzeHeadwearFeatures(imgData, imgW, imgH, headBox) {
        const [hx1, hy1, hx2, hy2] = headBox;
        const hw = Math.max(1, hx2 - hx1);
        const hh = Math.max(1, hy2 - hy1);
        const data = imgData.data;

        const step = Math.max(1, Math.floor(Math.min(hw, hh) / 48));

        // 1. Cranial hair & skin (y: 5%-50%, x: 25%-75%)
        const cy1 = hy1 + Math.floor(hh * 0.05);
        const cy2 = hy1 + Math.floor(hh * 0.50);
        const cx1 = hx1 + Math.floor(hw * 0.25);
        const cx2 = hx1 + Math.floor(hw * 0.75);
        let cCount = 0, hairCount = 0, skinCount = 0;

        for (let y = cy1; y < cy2; y += step) {
            for (let x = cx1; x < cx2; x += step) {
                const idx = (y * imgW + x) * 4;
                const [h, s, v] = rgbToHsv(data[idx], data[idx + 1], data[idx + 2]);
                cCount++;
                if (v < 75 || (v < 95 && s < 80)) hairCount++;
                if (h <= 25 && s >= 35 && s <= 180 && v >= 80 && v <= 240) skinCount++;
            }
        }
        const hairRatio = cCount > 0 ? hairCount / cCount : 0;
        const skinRatio = cCount > 0 ? skinCount / cCount : 0;

        // 2. Crown mean HSV & Side continuity (untuk Hijab)
        let crH = 0, crS = 0, crV = 0, crN = 0;
        for (let y = hy1 + Math.floor(hh * 0.05); y < hy1 + Math.floor(hh * 0.45); y += step) {
            for (let x = hx1 + Math.floor(hw * 0.20); x < hx1 + Math.floor(hw * 0.80); x += step) {
                const idx = (y * imgW + x) * 4;
                const [h, s, v] = rgbToHsv(data[idx], data[idx + 1], data[idx + 2]);
                crH += h; crS += s; crV += v; crN++;
            }
        }
        if (crN > 0) { crH /= crN; crS /= crN; crV /= crN; }

        let sH = 0, sS = 0, sV = 0, sN = 0;
        const sy1 = hy1 + Math.floor(hh * 0.35);
        const sy2 = hy1 + Math.floor(hh * 0.85);
        for (let y = sy1; y < sy2; y += step) {
            for (let x = hx1; x < hx1 + Math.floor(hw * 0.22); x += step) {
                const idx = (y * imgW + x) * 4;
                const [h, s, v] = rgbToHsv(data[idx], data[idx + 1], data[idx + 2]);
                sH += h; sS += s; sV += v; sN++;
            }
            for (let x = hx1 + Math.floor(hw * 0.78); x < hx2; x += step) {
                const idx = (y * imgW + x) * 4;
                const [h, s, v] = rgbToHsv(data[idx], data[idx + 1], data[idx + 2]);
                sH += h; sS += s; sV += v; sN++;
            }
        }
        if (sN > 0) { sH /= sN; sS /= sN; sV /= sN; }
        const shDiff = Math.min(Math.abs(crH - sH), 180 - Math.abs(crH - sH));
        const sideAchrom = (crS < 55 && sS < 55) || (crV < 70 && sV < 70);
        const sideContinuity = sN > 0 && (sideAchrom
            ? (Math.abs(crV - sV) < 55 && Math.abs(crS - sS) < 55)
            : (shDiff < 22 && Math.abs(crS - sS) < 50 && Math.abs(crV - sV) < 55));

        // 3. Neck ROI tepat di bawah headBox
        const ny1 = Math.max(0, hy2 - Math.floor(hh * 0.08));
        const ny2 = Math.min(imgH, hy2 + Math.floor(hh * 0.85));
        const nx1 = Math.max(0, hx1 - Math.floor(hw * 0.15));
        const nx2 = Math.min(imgW, hx2 + Math.floor(hw * 0.15));
        let nH = 0, nS = 0, nV = 0, nN = 0, nSkin = 0;
        for (let y = ny1; y < ny2; y += step) {
            for (let x = nx1; x < nx2; x += step) {
                const idx = (y * imgW + x) * 4;
                const [h, s, v] = rgbToHsv(data[idx], data[idx + 1], data[idx + 2]);
                nH += h; nS += s; nV += v; nN++;
                if (h <= 25 && s >= 35 && s <= 180 && v >= 80 && v <= 240) nSkin++;
            }
        }
        let isHijab = false;
        if (nN > 10) {
            nH /= nN; nS /= nN; nV /= nN;
            const neckSkinRatio = nSkin / nN;
            const nhDiff = Math.min(Math.abs(crH - nH), 180 - Math.abs(crH - nH));
            const neckAchrom = (crS < 55 && nS < 55) || (crV < 70 && nV < 70);
            const colorCont = neckAchrom
                ? (Math.abs(crS - nS) < 55 && Math.abs(crV - nV) < 55)
                : (nhDiff < 22 && Math.abs(crS - nS) < 50 && Math.abs(crV - nV) < 55);
            isHijab = colorCont && neckSkinRatio < 0.25 && sideContinuity;
        }

        // 4. Visor Helm Motor (y: 35%-78%, x: 15%-85%)
        const my1 = hy1 + Math.floor(hh * 0.35);
        const my2 = hy1 + Math.floor(hh * 0.78);
        const mx1 = hx1 + Math.floor(hw * 0.15);
        const mx2 = hx1 + Math.floor(hw * 0.85);
        let mN = 0, mDark = 0, mHorizEdge = 0;
        for (let y = my1; y < my2 - step; y += step) {
            for (let x = mx1; x < mx2; x += step) {
                const idx = (y * imgW + x) * 4;
                const idxDown = ((y + step) * imgW + x) * 4;
                const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                const lumDown = 0.299 * data[idxDown] + 0.587 * data[idxDown + 1] + 0.114 * data[idxDown + 2];
                mN++;
                if (lum < 45) mDark++;
                mHorizEdge += Math.abs(lumDown - lum);
            }
        }
        const darkVisorRatio = mN > 0 ? mDark / mN : 0;
        const visorHorizEdge = mN > 0 ? mHorizEdge / mN : 0;

        // 5. Topi (visor shadow + warmth)
        let vUp = 0, vUpN = 0, vLow = 0, vLowN = 0, rMinusB = 0, allN = 0;
        for (let y = hy1; y < hy2; y += step) {
            for (let x = hx1; x < hx2; x += step) {
                const idx = (y * imgW + x) * 4;
                const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                const [, , v] = rgbToHsv(r, g, b);
                rMinusB += (r - b); allN++;
                if (y < hy1 + hh * 0.40) { vUp += v; vUpN++; }
                else if (y >= hy1 + hh * 0.60) { vLow += v; vLowN++; }
            }
        }
        vUp = vUpN > 0 ? vUp / vUpN : 0;
        vLow = vLowN > 0 ? vLow / vLowN : 0;
        rMinusB = allN > 0 ? rMinusB / allN : 0;
        const visorShadow = Math.max(0, (vUp - vLow) / Math.max(1, vUp + vLow));

        // 6. Helm Sepeda: dark pockets on upper dome (tidak dianggap kalau rambut gelap)
        let domeDarkPockets = 0, domeEdgeCount = 0, domeN = 0;
        const dy1 = hy1;
        const dy2 = hy1 + Math.floor(hh * 0.55);
        const dx1 = hx1 + Math.floor(hw * 0.10);
        const dx2 = hx1 + Math.floor(hw * 0.90);
        for (let y = dy1 + step; y < dy2 - step; y += step) {
            for (let x = dx1 + step; x < dx2 - step; x += step) {
                const idx = (y * imgW + x) * 4;
                const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                const idxR = (y * imgW + (x + step)) * 4;
                const lumR = 0.299 * data[idxR] + 0.587 * data[idxR + 1] + 0.114 * data[idxR + 2];
                domeN++;
                if (Math.abs(lumR - lum) > 25) domeEdgeCount++;
                if (lum < crV * 0.55 && lum < 75 && crV > 85) domeDarkPockets++;
            }
        }
        const domeEdgeDensity = domeN > 0 ? domeEdgeCount / domeN : 0;
        const darkPocketRatio = domeN > 0 ? domeDarkPockets / domeN : 0;

        return {
            hairRatio,
            skinRatio,
            isHijab,
            darkVisorRatio,
            visorHorizEdge,
            visorShadow,
            rMinusB,
            domeEdgeDensity,
            darkPocketRatio
        };
    }

    function classifyHeadwear(yoloCls, yoloConf, feat) {
        const isKerudung = feat.isHijab;
        const isHelmMotor = (feat.darkVisorRatio >= 0.28) || (feat.darkVisorRatio >= 0.20 && feat.visorHorizEdge > 15.0);
        const isHelmSepeda = (feat.hairRatio < 0.30) && (feat.darkPocketRatio >= 0.08 && feat.domeEdgeDensity > 0.12);
        const isTopi = feat.visorShadow >= 0.18 && (feat.rMinusB > 22.0 || feat.visorHorizEdge > 15.0);

        let finalLabel = 'NO_HELMET';
        let confidence = yoloConf;

        if (yoloCls === 'helmet') {
            if (yoloConf < 0.55 && feat.hairRatio >= 0.45 && !isKerudung && !isHelmMotor) {
                finalLabel = 'NO_HELMET';
                confidence = Math.min(0.88, Math.max(0.60, feat.hairRatio));
            } else if (isHelmSepeda) {
                finalLabel = 'HELM_SEPEDA';
                confidence = Math.min(0.95, Math.max(yoloConf, 0.72));
            } else if (isHelmMotor) {
                finalLabel = 'HELM_MOTOR';
                confidence = Math.min(0.95, Math.max(yoloConf, 0.74));
            } else if (isKerudung) {
                finalLabel = 'KERUDUNG';
                confidence = Math.min(0.95, Math.max(yoloConf, 0.75));
            } else if (isTopi) {
                finalLabel = 'TOPI';
                confidence = Math.min(0.94, Math.max(yoloConf, 0.70));
            } else {
                finalLabel = 'REAL_HELMET';
                confidence = Math.min(0.98, yoloConf + 0.08);
            }
        } else {
            if (feat.skinRatio >= 0.35) {
                finalLabel = 'NO_HELMET';
                confidence = Math.min(0.96, Math.max(yoloConf, 0.75));
            } else if (isKerudung) {
                finalLabel = 'KERUDUNG';
                confidence = Math.min(0.95, Math.max(yoloConf, 0.75));
            } else if (feat.hairRatio >= 0.15) {
                finalLabel = 'NO_HELMET';
                confidence = Math.min(0.96, Math.max(yoloConf, 0.75));
            } else if (isTopi) {
                finalLabel = 'TOPI';
                confidence = Math.min(0.92, Math.max(yoloConf, 0.68));
            } else if (isHelmSepeda && feat.hairRatio < 0.10 && feat.skinRatio < 0.20) {
                finalLabel = 'HELM_SEPEDA';
                confidence = Math.min(0.92, Math.max(yoloConf, 0.68));
            } else if (isHelmMotor && feat.hairRatio < 0.15 && feat.skinRatio < 0.25) {
                finalLabel = 'HELM_MOTOR';
                confidence = Math.min(0.92, Math.max(yoloConf, 0.68));
            } else {
                finalLabel = 'NO_HELMET';
                confidence = Math.min(0.95, yoloConf + 0.05);
            }
        }

        const meta = STATUS_META[finalLabel];
        return {
            label: finalLabel,
            title: meta.title,
            badge: meta.badge,
            color_hex: meta.color_hex,
            confidence: Math.round(confidence * 1000) / 10
        };
    }

    async function detect(sourceCanvas, confThreshold = 0.35) {
        const sess = await init();
        ensureBuffers();

        const srcW = sourceCanvas.width;
        const srcH = sourceCanvas.height;

        // Letterbox ke 640x640 (pad warna 114, 114, 114 sesuai standar Ultralytics)
        const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
        const newW = Math.round(srcW * scale);
        const newH = Math.round(srcH * scale);
        const padX = Math.floor((INPUT_SIZE - newW) / 2);
        const padY = Math.floor((INPUT_SIZE - newH) / 2);

        letterboxCtx.fillStyle = 'rgb(114, 114, 114)';
        letterboxCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
        letterboxCtx.drawImage(sourceCanvas, 0, 0, srcW, srcH, padX, padY, newW, newH);

        const lbData = letterboxCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
        const area = INPUT_SIZE * INPUT_SIZE;
        for (let i = 0; i < area; i++) {
            const p = i * 4;
            inputBuffer[i] = lbData[p] / 255.0;
            inputBuffer[area + i] = lbData[p + 1] / 255.0;
            inputBuffer[2 * area + i] = lbData[p + 2] / 255.0;
        }

        const inputName = sess.inputNames[0];
        const tensor = new ort.Tensor('float32', inputBuffer, [1, 3, INPUT_SIZE, INPUT_SIZE]);
        const outputs = await sess.run({ [inputName]: tensor });
        const outTensor = outputs[sess.outputNames[0]];
        const outData = outTensor.data; // shape [1, 9, 8400]
        const numAnchors = outTensor.dims[2] || 8400;

        const rawBoxes = [];
        for (let i = 0; i < numAnchors; i++) {
            let bestCls = 0;
            let bestScore = -1;
            for (let c = 0; c < CLASS_NAMES.length; c++) {
                const score = outData[(4 + c) * numAnchors + i];
                if (score > bestScore) {
                    bestScore = score;
                    bestCls = c;
                }
            }
            if (bestScore >= confThreshold) {
                const cx = outData[0 * numAnchors + i];
                const cy = outData[1 * numAnchors + i];
                const w = outData[2 * numAnchors + i];
                const h = outData[3 * numAnchors + i];

                const x1 = Math.max(0, Math.round((cx - w / 2 - padX) / scale));
                const y1 = Math.max(0, Math.round((cy - h / 2 - padY) / scale));
                const x2 = Math.min(srcW, Math.round((cx + w / 2 - padX) / scale));
                const y2 = Math.min(srcH, Math.round((cy + h / 2 - padY) / scale));

                if (x2 > x1 && y2 > y1) {
                    rawBoxes.push({
                        bbox: [x1, y1, x2, y2],
                        cls_name: CLASS_NAMES[bestCls],
                        conf: bestScore
                    });
                }
            }
        }

        const filtered = nms(rawBoxes, 0.45);
        const personBoxes = filtered.filter(b => b.cls_name === 'person');
        const headBoxes = filtered.filter(b => b.cls_name === 'helmet' || b.cls_name === 'no-helmet');

        const srcCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });
        const srcImgData = srcCtx.getImageData(0, 0, srcW, srcH);

        const processedHeads = [];
        for (const head of headBoxes) {
            const feat = analyzeHeadwearFeatures(srcImgData, srcW, srcH, head.bbox);
            const cls = classifyHeadwear(head.cls_name, head.conf, feat);
            processedHeads.push({ bbox: head.bbox, ...cls });
        }

        // Fallback: jika ada person tanpa head_box, inspeksi 32% bagian atas
        for (const p of personBoxes) {
            const [px1, py1, px2, py2] = p.bbox;
            const pw = px2 - px1;
            const ph = py2 - py1;
            let hasHead = false;
            for (const h of processedHeads) {
                const [hx1, hy1, hx2, hy2] = h.bbox;
                const hcx = (hx1 + hx2) / 2;
                const hcy = (hy1 + hy2) / 2;
                if (hcx >= px1 - pw * 0.2 && hcx <= px2 + pw * 0.2 && hcy >= py1 - ph * 0.2 && hcy <= py1 + ph * 0.55) {
                    hasHead = true;
                    break;
                }
            }
            if (!hasHead && ph > 40 && pw > 20) {
                const estHead = [
                    Math.max(0, px1 + Math.round(pw * 0.15)),
                    py1,
                    Math.min(srcW, px2 - Math.round(pw * 0.15)),
                    Math.min(srcH, py1 + Math.round(ph * 0.32))
                ];
                if (estHead[2] > estHead[0] && estHead[3] > estHead[1]) {
                    const feat = analyzeHeadwearFeatures(srcImgData, srcW, srcH, estHead);
                    const cls = classifyHeadwear('no-helmet', 0.75, feat);
                    processedHeads.push({ bbox: estHead, ...cls });
                }
            }
        }

        const breakdown = {
            real_helmet: 0,
            topi: 0,
            helm_motor: 0,
            helm_sepeda: 0,
            kerudung: 0,
            no_helmet: 0
        };
        const violationTypes = new Set();
        for (const d of processedHeads) {
            const key = d.label.toLowerCase();
            if (key in breakdown) breakdown[key]++;
            if (d.label !== 'REAL_HELMET') violationTypes.add(d.title);
        }

        const totalWorkers = processedHeads.length;
        const compliant = breakdown.real_helmet;
        const violations = totalWorkers - compliant;
        const complianceRate = totalWorkers > 0 ? Math.round((compliant / totalWorkers) * 1000) / 10 : 100.0;

        return {
            status: 'success',
            provider: activeProvider,
            detections: processedHeads,
            frame_size: [srcW, srcH],
            stats: {
                total_workers: totalWorkers,
                compliant,
                violations,
                compliance_rate: complianceRate,
                breakdown
            },
            has_violation: violations > 0,
            violation_types: Array.from(violationTypes)
        };
    }

    return {
        init,
        detect,
        getProvider: () => activeProvider
    };
})();
