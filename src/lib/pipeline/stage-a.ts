// analysis-pipeline §2 (A3, A4, A5), §5, §8. Stage A for one photo: review the image, align it on
// the template, detect the shots, save the result.

import * as Comlink from 'comlink';

import { usedBackingFallback } from '@/lib/cv/backing-colour';
import { BLUR_THRESHOLD } from '@/lib/cv/constants';
import { SIGHTING_TEMPLATE } from '@/lib/defaults/templates';
import { INITIAL_DETECTION, type DetectionRecord, type Shot, type TargetAnalysis } from '@/lib/domain/analysis';
import { isTargetPhoto } from '@/lib/domain/backing';
import { isCategorizationComplete } from '@/lib/domain/categorization';
import type { TemplateId, Warning } from '@/lib/domain/enums';
import type { Calibration, Categorization, TargetPhoto } from '@/lib/domain/photo';
import { withBackingColour } from '@/lib/domain/backing';
import { backingInputFromSettings } from '@/lib/domain/settings';
import { photoStatus } from '@/lib/domain/status';
import { withoutArea } from '@/lib/scoring/cap-shots';
import { reconcileShots } from '@/lib/scoring/reconcile-shots';
import { scaleCalibration } from '@/lib/geometry/transform';
import { emitPipelineChanged } from '@/lib/pipeline/events';
import { AnalysisNotFoundError, PhotoNotFoundError } from '@/lib/services/photos';
import type { ServiceContext } from '@/lib/services/context';
import type { ImageTools } from '@/lib/services/ingest';
import { getAnalysisRecord, putAnalysisRecord } from '@/lib/store/analyses-repo';
import { photoWorkingKey } from '@/lib/store/blob-keys';
import { getBlob } from '@/lib/store/blobs-repo';
import { getPhotoRecord, putPhotoRecord } from '@/lib/store/photos-repo';
import { getSettings } from '@/lib/store/settings-repo';
import { terminateCvClient, type CvWorkerApi, type ReviewAndAlignResult } from '@/workers/cv-client';

import { chooseAlignment } from './alignment';
import { shouldRerunStageA } from './template-change';

/** Only the part of the worker Stage A needs, so tests can stub it. */
export type CvApi = Pick<CvWorkerApi, 'reviewAndAlign' | 'detectShots'>;

/** Owner report, 2026-09-26: a CV call has been observed to hang on-device rather than ever settling. */
export const CV_CALL_TIMEOUT_MS = 45_000;

/**
 * Races a CV worker call against a timeout. On timeout, kills the persistent worker (`cv-client.ts`) so
 * the *next* job gets a fresh one instead of queuing behind a wedged one forever, and rejects — which
 * `runStageA`'s existing catch records as `stageA: 'error'` (retryable), rather than an indefinite
 * "Reviewing…" spinner that previously needed the app force-quit and relaunched to clear.
 */
export function withCvTimeout<T>(promise: Promise<T>, label: string, timeoutMs = CV_CALL_TIMEOUT_MS): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      terminateCvClient();
      reject(new Error(`${label} did not respond within ${Math.round(timeoutMs / 1000)}s; the CV worker was restarted`));
    }, timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err: unknown) => {
        clearTimeout(timer);
        reject(err);
      },
    );
  });
}

/** backing-sheet.md §3: a backing-card photo is not a target, so Stage A never runs on it. */
export class NotATargetPhotoError extends Error {
  constructor(photoId: string) {
    super(`Not a target photo: ${photoId}`);
    this.name = 'NotATargetPhotoError';
  }
}

export class WorkingImageMissingError extends Error {
  constructor(photoId: string) {
    super(`Working image missing for photo: ${photoId}`);
    this.name = 'WorkingImageMissingError';
  }
}

/** capture-overlay §3.3: the prior is stored in FRAME px; Stage A scales it to working px. */
export function priorInWorkingPx(photo: TargetPhoto): Calibration | null {
  const capture = photo.capture;
  const prior = capture?.calibrationPriorFramePx ?? null;
  if (capture === null || prior === null) return null;
  const frameLongest = Math.max(capture.frameWidthPx, capture.frameHeightPx);
  const workingLongest = Math.max(photo.working.widthPx, photo.working.heightPx);
  if (frameLongest <= 0) return null;
  return scaleCalibration(prior, workingLongest / frameLongest);
}

/**
 * A5 has to know which sheet it is looking at, because the printed circles it erases (M11 step 4)
 * and the outer radius it crops to differ per template. Nothing in the spec names a source for it,
 * so this is the precedence Stage A uses — what the user (or the capture screen) said, then the
 * overlay, then A3's hint, and finally the anchor size the calibration was measured against. See
 * the M11 Open questions.
 */
export function shotTemplate(
  photo: TargetPhoto,
  hint: ReviewAndAlignResult['templateHint'],
  calibration: Calibration,
): TemplateId {
  return (
    photo.categorization.template ??
    photo.capture?.overlayTemplate ??
    hint?.template ??
    (calibration.anchorDiameterMm === SIGHTING_TEMPLATE.anchor.diameterMm ? 'sighting' : 'precision')
  );
}

/** One transaction: save the mutated analysis and the photo status it implies (data-model §7). */
async function commitAnalysis(
  ctx: ServiceContext,
  photoId: string,
  mutate: (analysis: TargetAnalysis) => TargetAnalysis,
): Promise<void> {
  const nowIso = ctx.now().toISOString();
  const tx = ctx.db.transaction(['photos', 'analyses'], 'readwrite');
  const photo = await getPhotoRecord(tx, photoId);
  const analysis = await getAnalysisRecord(tx, photoId);
  if (photo === null || analysis === null) {
    await tx.done;
    throw photo === null ? new PhotoNotFoundError(photoId) : new AnalysisNotFoundError(photoId);
  }

  const next: TargetAnalysis = { ...mutate(analysis), updatedAt: nowIso };
  const { status, reasons } = photoStatus({
    categorization: photo.categorization,
    analysis: next,
    result: next.computed?.result ?? null,
  });

  await putAnalysisRecord(tx, next);
  await putPhotoRecord(tx, { ...photo, status, reasons });
  await tx.done;
}

/**
 * analysis-pipeline §5. Runs A3, A4 and A5 for one photo and records the outcome in one transaction.
 * Never throws for a CV or decode failure: it records `stageA: 'error'` instead, so the runner does
 * not retry in a loop.
 */
export async function runStageA(
  ctx: ServiceContext,
  photoId: string,
  cvApi: CvApi,
  imageTools: ImageTools,
): Promise<void> {
  // The worker decodes the working JPEG itself; `imageTools` is part of the service signature
  // (data-model §7) and is used by later stages.
  void imageTools;

  const photo = await getPhotoRecord(ctx.db, photoId);
  if (photo === null) throw new PhotoNotFoundError(photoId);
  // backing-sheet.md §3: card photos are excluded from Stage A and B.
  if (!isTargetPhoto(photo)) throw new NotATargetPhotoError(photoId);
  const analysis = await getAnalysisRecord(ctx.db, photoId);
  if (analysis === null) throw new AnalysisNotFoundError(photoId);

  await commitAnalysis(ctx, photoId, (a) => ({
    ...a,
    pipeline: { ...a.pipeline, stageA: 'running', error: null },
  }));
  emitPipelineChanged({ sessionId: photo.sessionId, photoId });

  try {
    const workingBlob = await getBlob(ctx.db, photoWorkingKey(photoId));
    if (workingBlob === null) throw new WorkingImageMissingError(photoId);
    const bytes = await workingBlob.arrayBuffer();
    // A5 needs the same image again, and `Comlink.transfer` detaches the buffer it hands over.
    const bytesForShots = bytes.slice(0);
    // data-model §5: the hole diameter the touch rule and A5 use is a profile override.
    const settings = await getSettings(ctx.db);
    // backing-sheet.md §5 (REV-48): the Settings backing, as it is now, decides which A5 path runs.
    const backing = backingInputFromSettings(settings);

    // §8: a calibration the user saved in Adjust is never replaced, so A4 is skipped for it.
    const manual = analysis.calibration?.source === 'manual';
    const manualCalibration = manual ? analysis.calibration : null;
    const prior = manual ? null : priorInWorkingPx(photo);
    // REV-57: align against the OWNER's template when they have chosen one, else the capture overlay's; the
    // disc size (115 vs 112.4 mm) and the printed circles depend on it. Before, only the overlay was passed, so
    // an import with a template set was still aligned by guess.
    const usedTemplate = photo.categorization.template ?? photo.capture?.overlayTemplate ?? null;

    const review = await withCvTimeout(cvApi.reviewAndAlign(Comlink.transfer(bytes, [bytes]), prior, usedTemplate), 'reviewAndAlign');

    const choice =
      manualCalibration !== null
        ? {
            calibration: manualCalibration,
            method: 'manual' as const,
            confidence: analysis.pipeline.alignment.confidence,
            warnings: [] as Array<'alignment-uncertain'>,
          }
        : chooseAlignment({ prior, detection: review.detection });

    const warnings: Warning[] = [...choice.warnings];
    if (review.sharpness < BLUR_THRESHOLD) warnings.push('image-blurry');
    // Stage B owns `template-mismatch` (analysis-pipeline §2 B4); keep it if it was already there.
    if (analysis.pipeline.warnings.includes('template-mismatch')) warnings.push('template-mismatch');

    // A5 (§2, §8): with no alignment there is nothing to measure against, and shots the user placed
    // in Adjust are never overwritten. In both cases the stored shots are left exactly as they are.
    const hasManualShot = analysis.shots.some((shot) => shot.source === 'manual');
    const detected =
      choice.calibration === null || hasManualShot
        ? null
        : await withCvTimeout(
            cvApi.detectShots(
              Comlink.transfer(bytesForShots, [bytesForShots]),
              choice.calibration,
              shotTemplate(photo, review.templateHint, choice.calibration),
              settings.profileOverrides.holeDiameterMm,
              backing,
            ),
            'detectShots',
          );

    // backing-sheet.md §3, §5.6: record which path ran, and warn when the colour path found nothing.
    const detection: DetectionRecord = detected
      ? withBackingColour(detected.detection, settings.backing)
      : (analysis.pipeline.detection ?? INITIAL_DETECTION);
    if (detected !== null && usedBackingFallback(detection)) warnings.push('backing-colour-not-found');

    // A5 / REV-39 (M20): the declared rounds are fact. Stage A runs before metadata, so it can only
    // reconcile (reject / cap / double punches / misses) when the categorization is already complete;
    // Stage B reconciles again once it is. A stored `Shot` carries no area, so the worker's is dropped.
    let shots: Shot[] | null = detected === null ? null : withoutArea(detected.shots);
    if (shots !== null && isCategorizationComplete(photo.categorization)) {
      const reconciled = reconcileShots({
        shots,
        categorization: photo.categorization,
        method: detection.method,
        maxPlausibleHoles: settings.maxPlausibleHoles,
      });
      shots = reconciled.shots;
      warnings.push(...reconciled.warnings);
    }

    await commitAnalysis(ctx, photoId, (a) => ({
      ...a,
      calibration: choice.calibration,
      shots: shots === null ? a.shots : shots,
      pipeline: {
        ...a.pipeline,
        stageA: 'done',
        error: null,
        alignment: { method: choice.method, confidence: choice.confidence },
        templateHint: review.templateHint,
        sharpness: review.sharpness,
        warnings,
        detection,
      },
    }));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await commitAnalysis(ctx, photoId, (a) => ({
      ...a,
      pipeline: { ...a.pipeline, stageA: 'error', error: message.slice(0, 200) },
    }));
  }

  await requeueIfTemplateChanged(ctx, photoId, photo.categorization);
  emitPipelineChanged({ sessionId: photo.sessionId, photoId });
}

/**
 * REV-57: the owner can change the template while this run is in flight, and the stored photo then has a
 * template the run did not use. `updatePhotoMetadata` cannot queue Stage A for a running job, so the run
 * checks on the way out and queues itself once. It cannot loop: the next run reads the stored template.
 */
async function requeueIfTemplateChanged(ctx: ServiceContext, photoId: string, used: Categorization): Promise<void> {
  const now = await getPhotoRecord(ctx.db, photoId);
  const analysis = await getAnalysisRecord(ctx.db, photoId);
  if (now === null || analysis === null) return;
  if (!shouldRerunStageA(used, now.categorization, analysis)) return;
  await commitAnalysis(ctx, photoId, (a) => ({ ...a, pipeline: { ...a.pipeline, stageA: 'pending', stageB: 'pending' } }));
}
