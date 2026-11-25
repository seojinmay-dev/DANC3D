import test from 'node:test';
import assert from 'node:assert/strict';
import {createScoreDisplay} from '../src/services/scoreDisplay.js';

test('first valid score is shown immediately; invalid scores keep it', () => {
  const display = createScoreDisplay();
  assert.equal(display.update(null, 0), null);
  assert.equal(display.update(.9, .1), 90);
  assert.equal(display.update(NaN, .2), 90);
});

test('smoothing follows seconds, not frame count', () => {
  const run = fps => {
    const display = createScoreDisplay({holdSec: 0});
    display.update(1, 0);
    for (let frame = 1; frame <= fps; frame++) display.update(0, frame / fps);
    return display.smoothed;
  };
  // One time constant (1s) moves ~63% toward the target at any frame rate.
  for (const fps of [10, 30, 60]) assert.ok(Math.abs(run(fps) - 100 * Math.exp(-1)) < .01, `${fps}fps`);
});

test('shown number changes at most every hold interval', () => {
  const display = createScoreDisplay();
  display.update(1, 0);
  const shown = [];
  for (let frame = 1; frame <= 30; frame++) shown.push(display.update(0, frame / 30));
  assert.equal(new Set(shown.slice(0, 14)).size, 1);
  assert.equal(shown[0], 100);
  assert.ok(shown[14] < 100);
  assert.equal(new Set(shown.slice(14, 29)).size, 1);
  assert.ok(shown[29] < shown[14]);
});

test('a long gap is capped and reset clears the score', () => {
  const display = createScoreDisplay({holdSec: 0});
  display.update(1, 0);
  display.update(0, 30);
  assert.ok(display.smoothed > 70);
  display.reset();
  assert.equal(display.update(.4, 31), 40);
});
