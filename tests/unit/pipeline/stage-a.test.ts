import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { DetectionRecord, Shot } from '@/lib/domain/analysis';
import type { ColourSignature } from '@/lib/domain/backing';
import type { TemplateId } from '@/lib/domain/enums';
import type { Calibration } from '@/lib/domain/photo';
import { defaultAppSettings } from '@/lib/domain/settings';
import {
  NotATargetPhotoError,
  runStageA,
  priorInWorkingPx,
  shotTemplate,
  withCvTimeout,
  type CvApi,
} from '@/lib/pipeline/stage-a';
import type { ServiceContext } from '@/lib/services/context';
import { getAnalysisRecord, putAnalysisRecord } from '@/lib/store/analyses-repo';
import { photoWorkingKey } from '@/lib/store/blob-keys';
import { putBlob } from '@/lib/store/blobs-repo';
import { getPhotoRecord, putPhotoRecord } from '@/lib/store/photos-repo';
import { putSessionRecord } from '@/lib/store/sessions-repo';
import { putSettings } from '@/lib/store/settings-repo';
import type { BackingInput, ReviewAndAlignResult } from '@/workers/cv-client';

import { openTestDb } from '../../helpers/db';
import { makeTestContext } from '../../helpers/fixtures';
import { makeAnalysis, makeCapture, makePhoto, makePrior, makeSession } from '../../helpers/records';
import { stubImageTools } from '../../helpers/stub-image-tools';

/** What the CV worker would have measured: a disc at (620, 830) in the 1200x1600 working image. */
const MEASURED: Calibration = {
  cx: 620,
  cy: 830,
  radiusPx: 260,
  axisRatio: 0.93,
  angleDeg: 0,
  anchorDiameterMm: 112.4,
  source: 'auto',
  confidence: 0.96,
  perspective: null,
};

const MANUAL: Calibration = { ...MEASURED, cx: 500, cy: 500, source: 'manual', confidence: null };

/** What A5 would have detected. */
const DETECTED: Shot = {
  id: 'auto-1',
  xMm: 1.4,
  yMm: -0.8,
  multiplicity: 1,
  positionOverrides: null,
  source: 'auto',
  confidence: 0.92,
  cluster: false,
  possibleOverlap: false,
};

/** A shot the user placed in Adjust: A5 must never touch it (analysis-pipeline §8). */
const MANUAL_SHOT: Shot = {
  id: 'm-1',
  xMm: -7.6,
  yMm: -9.4,
  multiplicity: 1,
  positionOverrides: null,
  source: 'manual',
  confidence: null,
  cluster: false,
  possibleOverlap: false,
};

/**
 * A full set for the seeded 10-round prone target, so the alignment tests below see every round
 * accounted for and no REV-39 reconciliation warning.
 */
const TEN: Shot[] = Array.from({ length: 10 }, (_, i) => ({ ...DETECTED, id: `auto-${i + 1}`, xMm: i * 3, yMm: 0 }));

function review(over: Partial<ReviewAndAlignResult> = {}): ReviewAndAlignResult {
  return {
    detection: null,
    sharpness: 120,
    templateHint: { template: 'precision', confidence: 0.75 },
    ...over,
  };
}

/** The owner's orange backing card (backing-sheet.md §4, measured 15.9 degrees). */
const ORANGE: ColourSignature = { hueDeg: 15.9, hueSpreadDeg: 2.4, satP10: 0.73, valP10: 0.85, samples: 70610 };
const STANDARD_DETECTION: DetectionRecord = { method: 'standard', backing: 'off', fallbackReason: null };

function stubCv(result: ReviewAndAlignResult | Error, shots: Shot[] = TEN, detectionRecord = STANDARD_DETECTION) {
  const calls: Array<{ prior: Calibration | null; templateHint: TemplateId | null }> = [];
  const detectCalls: Array<{
    calibration: Calibration;
    template: TemplateId;
    holeDiameterMm: number;
    backing: BackingInput;
  }> = [];
  const api: CvApi = {
    async reviewAndAlign(workingJpeg, prior, templateHint) {
      void workingJpeg;
      calls.push({ prior, templateHint });
      if (result instanceof Error) throw result;
      return result;
    },
    async detectShots(workingJpeg, calibration, template, holeDiameterMm, backing) {
      void workingJpeg;
      detectCalls.push({ calibration, template, holeDiameterMm, backing });
      return { shots, detection: detectionRecord, suggestions: [], holeWidths: [] };
    },
  };
  return { api, calls, detectCalls };
}

interface Seeded {
  ctx: ServiceContext;
  photoId: string;
}

/** A stored photo with a 2400x3200 capture frame and a 1200x1600 working image (scale factor 0.5). */
async function seed(
  opts: {
    withPrior?: boolean;
    calibration?: Calibration | null;
    shots?: Shot[];
    /** `null` is an import whose template the user has not set yet. */
    template?: TemplateId | null;
  } = {},
): Promise<Seeded> {
  const db = await openTestDb();
  const ctx = makeTestContext(db);
  const session = makeSession();
  const photo = makePhoto({
    sessionId: session.id,
    capture:
      opts.withPrior === false
        ? null
        : makeCapture({ overlayTemplate: 'precision', calibrationPriorFramePx: makePrior() }),
    categorization: {
      template: opts.template === undefined ? 'precision' : opts.template,
      position: 'prone',
      roundsProne: 10,
      roundsStanding: null,
    },
  });
  const analysis = makeAnalysis(photo.id, {}, { calibration: opts.calibration ?? null, shots: opts.shots ?? [] });

  await putSessionRecord(db, { ...session, photoIds: [photo.id] });
  await putPhotoRecord(db, photo);
  await putAnalysisRecord(db, analysis);
  await putBlob(db, photoWorkingKey(photo.id), {
    bytes: new Uint8Array([0xff, 0xd8, 0xff, 0xe0]).buffer,
    contentType: 'image/jpeg',
    sizeBytes: 4,
    createdAt: '2026-09-05T23:40:00.000Z',
  });

  return { ctx, photoId: photo.id };
}

let imageTools: ReturnType<typeof stubImageTools>;

beforeEach(() => {
  imageTools = stubImageTools();
});

describe('priorInWorkingPx (capture-overlay §3.3)', () => {
  it('scales the frame-px prior by workingLongest / frameLongest', () => {
    const photo = makePhoto({ capture: makeCapture({ calibrationPriorFramePx: makePrior() }) });
    const prior = priorInWorkingPx(photo);

    expect(prior).toEqual({ ...makePrior(), cx: 600, cy: 800, radiusPx: 280 });
  });

  it('is null when the photo has no capture prior', () => {
    expect(priorInWorkingPx(makePhoto({ capture: null }))).toBeNull();
    expect(priorInWorkingPx(makePhoto({ capture: makeCapture() }))).toBeNull();
  });
});

describe('runStageA (analysis-pipeline §2 A3/A4, §5)', () => {
  it('uses a detection inside the prior gate with no warning', async () => {
    const { ctx, photoId } = await seed();
    const { api, calls } = stubCv(review({ detection: { calibration: MEASURED, confidence: 0.96, outsidePrior: false } }));

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.pipeline.stageA).toBe('done');
    expect(analysis?.pipeline.alignment).toEqual({ method: 'cv', confidence: 0.96 });
    expect(analysis?.calibration).toEqual({ ...MEASURED, source: 'auto', confidence: 0.96 });
    expect(analysis?.pipeline.warnings).toEqual([]);
    expect(analysis?.pipeline.sharpness).toBe(120);
    expect(analysis?.pipeline.templateHint).toEqual({ template: 'precision', confidence: 0.75 });
    // The scaled prior and the capture template reach the worker.
    expect(calls[0]?.prior?.radiusPx).toBe(280);
    expect(calls[0]?.templateHint).toBe('precision');

    const photo = await getPhotoRecord(ctx.db, photoId);
    expect(photo?.status).toBe('ready');
    expect(photo?.reasons).toEqual([]);
  });

  it('keeps a detection outside the prior gate and warns instead of falling back (REV-25)', async () => {
    const { ctx, photoId } = await seed();
    const { api } = stubCv(review({ detection: { calibration: MEASURED, confidence: 0.88, outsidePrior: true } }));

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.pipeline.alignment).toEqual({ method: 'cv', confidence: 0.88 });
    expect(analysis?.calibration?.cx).toBe(MEASURED.cx);
    expect(analysis?.calibration?.radiusPx).toBe(MEASURED.radiusPx);
    expect(analysis?.calibration?.source).toBe('auto');
    expect(analysis?.pipeline.warnings).toEqual(['alignment-uncertain']);

    const photo = await getPhotoRecord(ctx.db, photoId);
    expect(photo?.status).toBe('ready');
    expect(photo?.reasons).toEqual(['alignment-uncertain']);
  });

  it('falls back to the scaled overlay prior when nothing was detected', async () => {
    const { ctx, photoId } = await seed();
    const { api } = stubCv(review({ detection: null }));

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.pipeline.alignment).toEqual({ method: 'overlay', confidence: null });
    expect(analysis?.calibration).toEqual({ ...makePrior(), cx: 600, cy: 800, radiusPx: 280, source: 'overlay', confidence: null });
    expect(analysis?.pipeline.warnings).toEqual(['alignment-uncertain']);
  });

  it('reports no alignment at all when there is neither a prior nor a detection', async () => {
    // An import whose template the owner has not chosen yet (REV-57: once chosen, Stage A aligns against it).
    const { ctx, photoId } = await seed({ withPrior: false, template: null });
    const { api, calls } = stubCv(review({ detection: null }));

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.pipeline.alignment).toEqual({ method: 'none', confidence: null });
    expect(analysis?.calibration).toBeNull();
    expect(analysis?.pipeline.warnings).toEqual([]);
    // No overlay template and none chosen, so the worker searches both anchor sizes.
    expect(calls[0]?.templateHint).toBeNull();
    expect(calls[0]?.prior).toBeNull();
  });

  it('never re-aligns a manual calibration (analysis-pipeline §8)', async () => {
    const { ctx, photoId } = await seed({ calibration: MANUAL });
    const { api, calls } = stubCv(review({ detection: { calibration: MEASURED, confidence: 0.96, outsidePrior: false } }));

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.calibration).toEqual(MANUAL);
    expect(analysis?.pipeline.alignment.method).toBe('manual');
    expect(analysis?.pipeline.warnings).toEqual([]);
    // A4 is skipped: no prior is handed over and the detection is ignored.
    expect(calls[0]?.prior).toBeNull();
    // A3 still ran.
    expect(analysis?.pipeline.sharpness).toBe(120);
  });

  it('adds image-blurry below BLUR_THRESHOLD', async () => {
    const { ctx, photoId } = await seed();
    const { api } = stubCv(
      review({ sharpness: 12, detection: { calibration: MEASURED, confidence: 0.96, outsidePrior: false } }),
    );

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.pipeline.warnings).toEqual(['image-blurry']);
    const photo = await getPhotoRecord(ctx.db, photoId);
    expect(photo?.reasons).toEqual(['image-blurry']);
  });

  it('records stageA error, truncated to 200 chars, when the worker throws', async () => {
    const { ctx, photoId } = await seed();
    const { api } = stubCv(new Error('x'.repeat(300)));

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.pipeline.stageA).toBe('error');
    expect(analysis?.pipeline.error).toHaveLength(200);

    const photo = await getPhotoRecord(ctx.db, photoId);
    expect(photo?.status).toBe('failed');
  });
});

describe('runStageA A5: shot detection (analysis-pipeline §2 A5, §8)', () => {
  const detection = { calibration: MEASURED, confidence: 0.96, outsidePrior: false };

  it('replaces the shots, with the chosen calibration, template and hole diameter', async () => {
    const { ctx, photoId } = await seed();
    const { api, detectCalls } = stubCv(review({ detection }), [DETECTED]);

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.shots).toEqual([DETECTED]);
    expect(analysis?.pipeline.stageA).toBe('done');

    expect(detectCalls).toHaveLength(1);
    // A5 runs on the alignment A4 chose, not on the raw detection.
    expect(detectCalls[0]?.calibration).toEqual({ ...MEASURED, source: 'auto', confidence: 0.96 });
    expect(detectCalls[0]?.template).toBe('precision');
    // data-model §5: the default profile override, geometry-scoring §1.1's .22 LR hole.
    expect(detectCalls[0]?.holeDiameterMm).toBe(5.6);
  });

  it('uses the stored profile hole diameter (data-model §5)', async () => {
    const { ctx, photoId } = await seed();
    await putSettings(ctx.db, { ...defaultAppSettings(), profileOverrides: { holeDiameterMm: 4.5 } });
    const { api, detectCalls } = stubCv(review({ detection }), [DETECTED]);

    await runStageA(ctx, photoId, api, imageTools);

    expect(detectCalls[0]?.holeDiameterMm).toBe(4.5);
  });

  it('skips A5 when there is no calibration, leaving the shots empty', async () => {
    const { ctx, photoId } = await seed({ withPrior: false });
    const { api, detectCalls } = stubCv(review({ detection: null }), [DETECTED]);

    await runStageA(ctx, photoId, api, imageTools);

    expect(detectCalls).toHaveLength(0);
    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.calibration).toBeNull();
    expect(analysis?.shots).toEqual([]);
    expect(analysis?.pipeline.stageA).toBe('done');
  });

  it('never replaces shots the user placed by hand (§8)', async () => {
    const { ctx, photoId } = await seed({ shots: [MANUAL_SHOT] });
    const { api, detectCalls } = stubCv(review({ detection }), [DETECTED]);

    await runStageA(ctx, photoId, api, imageTools);

    expect(detectCalls).toHaveLength(0);
    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.shots).toEqual([MANUAL_SHOT]);
  });

  it('still detects over auto shots from a previous run', async () => {
    const stale: Shot = { ...DETECTED, id: 'auto-9', xMm: 40, yMm: 40 };
    const { ctx, photoId } = await seed({ shots: [stale] });
    const { api, detectCalls } = stubCv(review({ detection }), [DETECTED]);

    await runStageA(ctx, photoId, api, imageTools);

    expect(detectCalls).toHaveLength(1);
    expect((await getAnalysisRecord(ctx.db, photoId))?.shots).toEqual([DETECTED]);
  });

  it('caps the shots to the declared rounds and warns, within the maxPlausibleHoles safety net (REV-28)', async () => {
    const { ctx, photoId } = await seed(); // categorization: prone, 10 rounds
    // Owner instruction, 2026-09-26: maxPlausibleHoles defaults to 10, matching declared rounds here, so
    // the cap step (a few low-confidence extras) needs its own headroom to be exercised at all — a raw
    // count this far past declared is otherwise the maxPlausibleHoles reject case (see the test below).
    await putSettings(ctx.db, { ...defaultAppSettings(), maxPlausibleHoles: 20 });
    const twelve: Shot[] = Array.from({ length: 12 }, (_, i) => ({
      ...DETECTED,
      id: `auto-${i + 1}`,
      xMm: i,
      yMm: 0,
      confidence: 1 - i * 0.05,
    }));
    const { api } = stubCv(review({ detection }), twelve);

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.shots).toHaveLength(10);
    expect(analysis?.shots.map((shot) => shot.id)).not.toContain('auto-11');
    expect(analysis?.pipeline.warnings).toEqual(['extra-candidates-dropped']);

    // A capped photo is still for the owner to confirm (analysis-pipeline §4).
    const photo = await getPhotoRecord(ctx.db, photoId);
    expect(photo?.reasons).toEqual(['extra-candidates-dropped']);
  });

  it('rejects outright, ahead of the cap, when raw holes exceed maxPlausibleHoles (owner instruction, 2026-09-26)', async () => {
    const { ctx, photoId } = await seed(); // categorization: prone, 10 rounds; maxPlausibleHoles defaults to 10
    const twelve: Shot[] = Array.from({ length: 12 }, (_, i) => ({
      ...DETECTED,
      id: `auto-${i + 1}`,
      xMm: i,
      yMm: 0,
      confidence: 1 - i * 0.05,
    }));
    const { api } = stubCv(review({ detection }), twelve);

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.shots).toHaveLength(12);
    expect(analysis?.pipeline.warnings).toEqual(['too-many-holes']);
  });

  it('does not cap, or warn, when the shots already fit the declared rounds', async () => {
    const { ctx, photoId } = await seed();
    const { api } = stubCv(review({ detection }), TEN);

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.shots).toEqual(TEN);
    expect(analysis?.pipeline.warnings).toEqual([]);
  });

  it('reconciles fewer holes than rounds once the rounds are known: the rest are misses (REV-39)', async () => {
    const { ctx, photoId } = await seed();
    const { api } = stubCv(review({ detection }), [DETECTED]);

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.shots).toEqual([DETECTED]);
    expect(analysis?.pipeline.warnings).toEqual(['rounds-scored-as-miss']);
  });

  it('rejects clearly more holes than rounds on the colour path, keeping every shot (REV-39)', async () => {
    const { ctx, photoId } = await seed();
    const fifteen: Shot[] = Array.from({ length: 15 }, (_, i) => ({ ...DETECTED, id: `auto-${i + 1}`, xMm: i * 3, confidence: null }));
    const { api } = stubCv(review({ detection }), fifteen, { method: 'colour', backing: 'forced', fallbackReason: null });

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.shots).toHaveLength(15);
    expect(analysis?.pipeline.warnings).toEqual(['too-many-holes']);
  });

  it('leaves the shots uncapped while the categorization is incomplete (Stage A runs before metadata)', async () => {
    const { ctx, photoId } = await seed({ template: null });
    const twelve: Shot[] = Array.from({ length: 12 }, (_, i) => ({ ...DETECTED, id: `auto-${i + 1}`, xMm: i }));
    const { api } = stubCv(review({ detection }), twelve);

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.shots).toHaveLength(12);
    expect(analysis?.pipeline.warnings).toEqual([]);
  });

  it('passes the Settings backing to A5 and records the detection (backing-sheet.md §3, §5, REV-48)', async () => {
    const { ctx, photoId } = await seed();
    await putSettings(ctx.db, {
      ...defaultAppSettings(),
      backingMode: 'coloured',
      backing: { kind: 'coloured', source: 'card', cardPhotoId: null, colour: ORANGE },
    });
    const { api, detectCalls } = stubCv(review({ detection }), [DETECTED], {
      method: 'colour',
      backing: 'forced',
      fallbackReason: null,
    });

    await runStageA(ctx, photoId, api, imageTools);

    expect(detectCalls[0]?.backing).toEqual({ mode: 'coloured', colour: ORANGE });
    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.pipeline.detection).toEqual({
      method: 'colour',
      backing: 'forced',
      fallbackReason: null,
      // REV-82: the colour in force when the shots were found, for drawing them.
      backingColour: expect.stringMatching(/^#[0-9A-F]{6}$/),
    });
    expect(analysis?.pipeline.warnings).not.toContain('backing-colour-not-found');
  });

  it('warns backing-colour-not-found when the colour path fell back to the standard detector (§5.6)', async () => {
    const { ctx, photoId } = await seed();
    const { api } = stubCv(review({ detection }), [DETECTED], {
      method: 'standard',
      backing: 'forced',
      fallbackReason: 'no backing colour showed through',
    });

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.pipeline.warnings).toContain('backing-colour-not-found');
    const photo = await getPhotoRecord(ctx.db, photoId);
    expect(photo?.reasons).toContain('backing-colour-not-found');
  });

  it("does not warn when Auto simply decided the photo is not backed (backing 'not-detected')", async () => {
    const { ctx, photoId } = await seed();
    const { api } = stubCv(review({ detection }), [DETECTED], {
      method: 'standard',
      backing: 'not-detected',
      fallbackReason: 'no coloured spots',
    });

    await runStageA(ctx, photoId, api, imageTools);

    const analysis = await getAnalysisRecord(ctx.db, photoId);
    expect(analysis?.pipeline.warnings).not.toContain('backing-colour-not-found');
    expect(analysis?.pipeline.detection.backing).toBe('not-detected');
  });

  it('refuses to run on a backing-card photo (backing-sheet.md §3)', async () => {
    const { ctx, photoId } = await seed();
    const photo = (await getPhotoRecord(ctx.db, photoId))!;
    await putPhotoRecord(ctx.db, { ...photo, origin: 'backing-card' });
    const { api } = stubCv(review({ detection }), [DETECTED]);

    await expect(runStageA(ctx, photoId, api, imageTools)).rejects.toBeInstanceOf(NotATargetPhotoError);
  });

  it('falls back to the template hint when the photo has no template yet', async () => {
    const { ctx, photoId } = await seed({ withPrior: false, template: null });
    const sightingDisc: Calibration = { ...MEASURED, anchorDiameterMm: 115 };
    const { api, detectCalls } = stubCv(
      review({
        detection: { calibration: sightingDisc, confidence: 0.9, outsidePrior: false },
        templateHint: { template: 'sighting', confidence: 0.8 },
      }),
      [],
    );

    await runStageA(ctx, photoId, api, imageTools);

    expect(detectCalls[0]?.template).toBe('sighting');
  });

  it("detects with the user's template, not a confident hint that disagrees (M23 step 4)", async () => {
    const { ctx, photoId } = await seed({ withPrior: false, template: 'sighting' });
    const { api, detectCalls } = stubCv(
      review({ detection, templateHint: { template: 'precision', confidence: 1 } }),
      [],
    );

    await runStageA(ctx, photoId, api, imageTools);

    expect(detectCalls[0]?.template).toBe('sighting');
  });
});

describe('shotTemplate (M23 step 4: the user\'s choice wins)', () => {
  const precisionDisc: Calibration = { ...MEASURED, anchorDiameterMm: 112.4 };
  const sightingDisc: Calibration = { ...MEASURED, anchorDiameterMm: 115 };
  const confidentPrecision = { template: 'precision' as const, confidence: 1 };

  function photoWith(template: TemplateId | null, overlayTemplate: TemplateId | null) {
    return makePhoto({
      capture: overlayTemplate === null ? null : makeCapture({ overlayTemplate, calibrationPriorFramePx: makePrior() }),
      categorization: { template, position: null, roundsProne: null, roundsStanding: null },
    });
  }

  it("prefers the user's categorization over the overlay, a confident hint and the anchor size", () => {
    expect(shotTemplate(photoWith('sighting', 'precision'), confidentPrecision, precisionDisc)).toBe('sighting');
    expect(shotTemplate(photoWith('precision', 'sighting'), { template: 'sighting', confidence: 1 }, sightingDisc)).toBe(
      'precision',
    );
  });

  it('then the capture overlay, then the hint, then the anchor size the disc was measured at', () => {
    expect(shotTemplate(photoWith(null, 'sighting'), confidentPrecision, precisionDisc)).toBe('sighting');
    expect(shotTemplate(photoWith(null, null), confidentPrecision, sightingDisc)).toBe('precision');
    expect(shotTemplate(photoWith(null, null), null, sightingDisc)).toBe('sighting');
    expect(shotTemplate(photoWith(null, null), null, precisionDisc)).toBe('precision');
  });
});

describe('withCvTimeout (owner report, 2026-09-26: a CV call can hang indefinitely on-device)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('resolves with the call\'s own result when it settles before the timeout', async () => {
    const result = withCvTimeout(Promise.resolve('done'), 'reviewAndAlign', 1000);
    await vi.advanceTimersByTimeAsync(10);
    expect(await result).toBe('done');
  });

  it('rejects a hung call once the timeout elapses, instead of waiting forever', async () => {
    const never = new Promise<string>(() => {});
    const result = withCvTimeout(never, 'reviewAndAlign', 1000);
    const assertion = expect(result).rejects.toThrow(/reviewAndAlign did not respond within 1s/);
    await vi.advanceTimersByTimeAsync(1000);
    await assertion;
  });

  it('propagates a normal rejection unchanged, without waiting for the timeout', async () => {
    const result = withCvTimeout(Promise.reject(new Error('boom')), 'detectShots', 1000);
    await expect(result).rejects.toThrow('boom');
  });
});
