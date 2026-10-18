import test from "node:test";
import assert from "node:assert/strict";
import { selectExample, stateReply, validateSelection } from "../src/inputs.js";

test("switching examples replaces all paired assets", () => {
  const examples = [1, 2].map((n) => ({name: String(n), image: `${n}.png`, conditions: `${n}.zip`, preview: `${n}.mp4`, prompt: `scene ${n}`, max_chunks: 24}));
  assert.equal(selectExample(examples, "1").image, "1.png");
  assert.deepEqual(selectExample(examples, "2"), { image: "2.png", conditions: "2.zip", preview: "2.mp4", prompt: "scene 2", limit: 24 });
  assert.equal(selectExample(examples, "").image, null);
});
test("selection is explicit and bounded", () => {
  const good = {conditions: {}, image: {}, prompt: "road", seed: 0, limit: 24};
  validateSelection(good);
  for (const change of [{conditions: null}, {image: null}, {prompt: " "}, {seed: -1}, {limit: 1}, {limit: 481}]) {
    assert.throws(() => validateSelection({...good, ...change}));
  }
});
test("command refusal preserves the SDK error instead of replacing it with a missing ack", () => {
  const failure = Error("invalid_input: File is not a zip file");
  assert.throws(() => stateReply(undefined, failure), (error) => error === failure);
  assert.throws(() => stateReply(undefined), /state acknowledgement/);
  const state = {world_id: 3, active: true};
  assert.equal(stateReply({type: "state_update", data: state}), state);
});
