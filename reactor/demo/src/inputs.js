export function validateSelection({ conditions, image, prompt, seed, limit }) {
  if (!conditions || !image || !prompt.trim())
    throw Error("Select lossless conditions, a reference image and a prompt.");
  if (prompt.length > 32768) throw Error("Prompt exceeds 32768 characters.");
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647)
    throw Error("Seed must be an integer from 0 to 2147483647.");
  if (!Number.isInteger(limit) || limit < 0 || limit > 480 || (limit > 0 && limit < 6))
    throw Error("Chunk limit must be zero (all), or an integer from 6 to 480.");
}

export function stateReply(reply, error) {
  // The SDK reports command failures through lastError and resolves undefined.
  if (!Number.isInteger(reply?.data?.world_id))
    throw error ?? Error("The command did not return a state acknowledgement.");
  return reply.data;
}

export function selectExample(examples, name) {
  const item = examples.find((x) => x.name === name);
  return {
    conditions: item?.conditions ?? null,
    image: item?.image ?? null,
    preview: item?.preview ?? null,
    prompt: item?.prompt ?? "",
    limit: item?.max_chunks ?? 24,
  };
}

export async function sourceFile(source, kind) {
  let file = source;
  if (typeof source === "string") {
    if (!source.startsWith("/examples/media/") || source.includes(".."))
      throw Error("Unsupported example asset path.");
    const response = await fetch(source);
    if (!response.ok) throw Error(`Could not load ${kind}. Reinstall the examples.`);
    file = new File([await response.blob()], source.split("/").at(-1));
  }
  const limit = kind === "conditions" ? 512 : 16;
  if (!file?.size || file.size > limit * 1048576)
    throw Error(`${kind} must contain data and be at most ${limit} MiB.`);
  return file;
}
