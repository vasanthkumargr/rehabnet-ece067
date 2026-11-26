// MediaPipe pose estimation in the browser (runs on the device; video never leaves it).
// Only the 33 body landmarks are sent to the server.
import { FilesetResolver, PoseLandmarker } from "/vendor/mediapipe/vision_bundle.mjs";

let landmarker = null;
let loading = null;

export async function loadPose(model = "full") {
  if (landmarker) return landmarker;
  if (loading) return loading;
  loading = (async () => {
    const files = await FilesetResolver.forVisionTasks("/vendor/mediapipe/wasm");
    const make = (delegate) => PoseLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: `/vendor/mediapipe/pose_landmarker_${model}.task`, delegate },
      runningMode: "VIDEO", numPoses: 1,
      minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    try { landmarker = await make("GPU"); landmarker._delegate = "GPU"; }
    catch (e) { console.warn("GPU delegate failed, using CPU", e); landmarker = await make("CPU"); landmarker._delegate = "CPU"; }
    return landmarker;
  })();
  return loading;
}

let lastTs = 0;
export function detect(video) {
  let ts = performance.now();
  if (ts <= lastTs) ts = lastTs + 1;
  lastTs = ts;
  const r = landmarker.detectForVideo(video, ts);
  if (!r.landmarks || !r.landmarks.length) return null;
  return { img: r.landmarks[0], world: r.worldLandmarks[0] };
}

// Body yaw from the shoulder line in world coordinates: 0 deg = facing the camera, 90 deg = side-on.
export function cameraView(world) {
  const l = world[11], r = world[12];
  const dx = r.x - l.x, dz = r.z - l.z;
  const yaw = Math.atan2(Math.abs(dz), Math.abs(dx)) * 180 / Math.PI;
  return { yaw, view: yaw < 30 ? "front" : yaw < 60 ? "45°" : "side" };
}

// Is the whole body in frame and visible?
export function framing(img) {
  const need = [11, 12, 23, 24, 25, 26, 27, 28];
  const vis = need.filter((i) => (img[i].visibility ?? 1) > 0.5 && img[i].x > 0.02 && img[i].x < 0.98 && img[i].y > 0.02 && img[i].y < 0.98).length;
  const ys = img.map((p) => p.y), h = Math.max(...ys) - Math.min(...ys);
  if (vis < need.length) return { ok: false, msg: "Step back so your whole body, including your feet, is in the picture." };
  if (h < 0.35) return { ok: false, msg: "Come a little closer to the camera." };
  return { ok: true, msg: "" };
}

// Mean brightness (0-255) of a video frame, from a tiny 32x18 copy. In the Milestone 5 robustness test the person
// was found in only 12 % of frames of video darkened to a mean of ~35, so below ~45 the patient is asked for more light.
const _lum = document.createElement("canvas");
_lum.width = 32; _lum.height = 18;
export function brightness(video) {
  const c = _lum.getContext("2d", { willReadFrequently: true });
  c.drawImage(video, 0, 0, 32, 18);
  const d = c.getImageData(0, 0, 32, 18).data;
  let s = 0;
  for (let i = 0; i < d.length; i += 4) s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  return s / (d.length / 4);
}

const EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32]];

export function draw(ctx, img, w, h, highlight = null) {
  ctx.clearRect(0, 0, w, h);
  if (!img) return;
  ctx.lineWidth = Math.max(3, w / 240);
  ctx.lineCap = "round";
  for (const [a, b] of EDGES) {
    const p = img[a], q = img[b];
    if ((p.visibility ?? 1) < 0.3 || (q.visibility ?? 1) < 0.3) continue;
    ctx.strokeStyle = highlight && (highlight.includes(a) || highlight.includes(b)) ? "#F59E0B" : "rgba(45,212,191,.95)";
    ctx.beginPath(); ctx.moveTo(p.x * w, p.y * h); ctx.lineTo(q.x * w, q.y * h); ctx.stroke();
  }
  ctx.fillStyle = "#fff";
  for (const i of [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    const p = img[i];
    if ((p.visibility ?? 1) < 0.3) continue;
    ctx.beginPath(); ctx.arc(p.x * w, p.y * h, Math.max(3, w / 300), 0, Math.PI * 2); ctx.fill();
  }
}

// Landmarks per body-part word used in feedback text, to highlight the joint the message is about.
export const PART_JOINTS = {
  "upper body": [11, 12, 23, 24], chest: [11, 12, 23, 24], hips: [23, 24], "standing knee": [25, 26],
  "left knee": [25], "right knee": [26], "front knee": [25, 26], "back knee": [25, 26], knee: [25, 26],
  "left elbow": [13], "right elbow": [14], elbow: [13, 14], arm: [11, 12, 13, 14, 15, 16], "left arm": [11, 13, 15],
  "right arm": [12, 14, 16], leg: [23, 24, 25, 26, 27, 28],
};
