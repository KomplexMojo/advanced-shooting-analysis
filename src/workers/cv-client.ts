import * as Comlink from 'comlink';

import type { HoleWidth } from '@/lib/cv/backing-colour';
import type { ShotCandidate } from '@/lib/cv/holes';
import type { PointMm } from '@/lib/cv/split-cluster';
import type { DetectionRecord } from '@/lib/domain/analysis';
import type { BackingMode, ColourSignature } from '@/lib/domain/backing';
import type { TemplateId } from '@/lib/domain/enums';
import type { Calibration } from '@/lib/domain/photo';
import type { CappableShot } from '@/lib/scoring/cap-shots';

/** analysis-pipeline §6: what `reviewAndAlign` gives Stage A back. */
export interface ReviewAndAlignResult {
  detection: { calibration: Calibration; confidence: number; outsidePrior: boolean } | null;
  sharpness: number;
  templateHint: { template: TemplateId; confidence: number } | null;
}

/** backing-sheet.md §5 (REV-48): the Settings backing, as A5 and Re-analyze need it. */
export interface BackingInput {
  mode: BackingMode;
  /** The card's measured colour, when the session has one. */
  colour: ColourSignature | null;
}

/** analysis-pipeline §6: what `detectShots` gives Stage A back. */
export interface DetectShotsResult {
  /** backing-sheet.md §5.4: the colour path adds the blob's coloured area, which REV-28's cap ranks by. */
  shots: CappableShot[];
  /** backing-sheet.md §3, §5.6: which path ran, and why it fell back. */
  detection: DetectionRecord;
  /**
   * M21 step 1 (REV-40): the discarded candidates worth offering in Adjust, best first (`suggestShots`).
   * **Derived**: never written to IndexedDB, never part of `analysis.shots`, never scored or drawn on a
   * diagram. Stage A ignores it.
   */
  suggestions: ShotCandidate[];
  /** M21 step 3 (REV-41): each detected hole's measured width, read only by Adjust's double-punch prompt. */
  holeWidths: HoleWidth[];
}

export interface CvWorkerApi {
  ping(): Promise<{ loadedMs: number; hasMat: boolean }>;
  /** `templateHint` is `capture.overlayTemplate`; null (an import) searches for both anchor sizes. */
  reviewAndAlign(
    workingJpeg: ArrayBuffer,
    prior: Calibration | null,
    templateHint: TemplateId | null,
  ): Promise<ReviewAndAlignResult>;
  /**
   * M11 (A5). `calibration` is in the working image's pixel space; shots come back in mm.
   * `backing` (REV-38) picks the colour path, the standard path, or `Auto`'s per-photo decision.
   */
  detectShots(
    workingJpeg: ArrayBuffer,
    calibration: Calibration,
    template: TemplateId,
    holeDiameterMm: number,
    backing: BackingInput,
  ): Promise<DetectShotsResult>;
  /**
   * M11 step 6, for M13's Adjust screen: `k` centroids in mm for one cluster's points. Not listed in
   * analysis-pipeline §6 (see the M11 Open questions); the milestone's Files section asks for it here.
   */
  splitCluster(pointsMm: PointMm[], k: number): Promise<PointMm[]>;
}

let client: Comlink.Remote<CvWorkerApi> | undefined;
let rawWorker: Worker | undefined;

export function getCvClient(): Comlink.Remote<CvWorkerApi> {
  if (client === undefined) {
    rawWorker = new Worker(new URL('./cv.worker.ts', import.meta.url), { type: 'module' });
    client = Comlink.wrap<CvWorkerApi>(rawWorker);
  }
  return client;
}

/**
 * Owner report, 2026-09-26: a CV call can hang indefinitely on-device (never resolving or rejecting),
 * which wedges the single persistent worker for every job after it too, since Comlink's calls queue on
 * one thread. There is no way to interrupt a stuck call from the caller's side, so the only recovery is
 * to kill the worker outright and let the next `getCvClient()` create a fresh one. Callers pair this with
 * a timeout on the call they gave up waiting on (`stage-a.ts`).
 */
export function terminateCvClient(): void {
  rawWorker?.terminate();
  rawWorker = undefined;
  client = undefined;
}
