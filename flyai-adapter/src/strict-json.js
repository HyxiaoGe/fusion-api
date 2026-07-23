export class StrictJsonError extends Error {
  constructor(code) {
    super(code);
    this.name = "StrictJsonError";
    this.code = code;
  }
}

export function parseStrictJson(
  text,
  { maxDepth = 20, maxNodes = 10_000, maxStringChars = 262_144 } = {},
) {
  if (typeof text !== "string") {
    throw new StrictJsonError("invalid_json");
  }

  let cursor = 0;
  let nodes = 0;

  function fail(code = "invalid_json") {
    throw new StrictJsonError(code);
  }

  function skipWhitespace() {
    while (cursor < text.length && /[\u0009\u000a\u000d\u0020]/u.test(text[cursor])) {
      cursor += 1;
    }
  }

  function consumeNode(depth) {
    nodes += 1;
    if (nodes > maxNodes) {
      fail("json_too_many_nodes");
    }
    if (depth > maxDepth) {
      fail("json_too_deep");
    }
  }

  function parseString() {
    const start = cursor;
    cursor += 1;
    let decodedLength = 0;
    while (cursor < text.length) {
      const character = text[cursor];
      if (character === '"') {
        cursor += 1;
        try {
          const value = JSON.parse(text.slice(start, cursor));
          if (value.length > maxStringChars) {
            fail("json_string_too_large");
          }
          return value;
        } catch (error) {
          if (error instanceof StrictJsonError) {
            throw error;
          }
          fail();
        }
      }
      if (character === "\\") {
        cursor += 1;
        if (cursor >= text.length) fail();
        const escape = text[cursor];
        if (escape === "u") {
          const unicode = text.slice(cursor + 1, cursor + 5);
          if (!/^[0-9a-fA-F]{4}$/u.test(unicode)) fail();
          cursor += 5;
          decodedLength += 1;
          continue;
        }
        if (!'"\\/bfnrt'.includes(escape)) fail();
        cursor += 1;
        decodedLength += 1;
        continue;
      }
      if (character.charCodeAt(0) < 0x20) fail();
      cursor += 1;
      decodedLength += 1;
      if (decodedLength > maxStringChars) fail("json_string_too_large");
    }
    fail();
  }

  function parseNumber() {
    const match = text.slice(cursor).match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/u);
    if (!match) fail();
    cursor += match[0].length;
    const value = Number(match[0]);
    if (!Number.isFinite(value)) fail();
    return value;
  }

  function parseArray(depth) {
    cursor += 1;
    const result = [];
    skipWhitespace();
    if (text[cursor] === "]") {
      cursor += 1;
      return result;
    }
    while (cursor < text.length) {
      result.push(parseValue(depth + 1));
      skipWhitespace();
      if (text[cursor] === "]") {
        cursor += 1;
        return result;
      }
      if (text[cursor] !== ",") fail();
      cursor += 1;
      skipWhitespace();
    }
    fail();
  }

  function parseObject(depth) {
    cursor += 1;
    const result = Object.create(null);
    const keys = new Set();
    skipWhitespace();
    if (text[cursor] === "}") {
      cursor += 1;
      return result;
    }
    while (cursor < text.length) {
      if (text[cursor] !== '"') fail();
      const key = parseString();
      if (keys.has(key)) fail("duplicate_key");
      keys.add(key);
      skipWhitespace();
      if (text[cursor] !== ":") fail();
      cursor += 1;
      skipWhitespace();
      result[key] = parseValue(depth + 1);
      skipWhitespace();
      if (text[cursor] === "}") {
        cursor += 1;
        return result;
      }
      if (text[cursor] !== ",") fail();
      cursor += 1;
      skipWhitespace();
    }
    fail();
  }

  function parseValue(depth) {
    skipWhitespace();
    consumeNode(depth);
    const character = text[cursor];
    if (character === "{") return parseObject(depth);
    if (character === "[") return parseArray(depth);
    if (character === '"') return parseString();
    if (text.startsWith("true", cursor)) {
      cursor += 4;
      return true;
    }
    if (text.startsWith("false", cursor)) {
      cursor += 5;
      return false;
    }
    if (text.startsWith("null", cursor)) {
      cursor += 4;
      return null;
    }
    if (character === "-" || /[0-9]/u.test(character ?? "")) return parseNumber();
    fail();
  }

  const result = parseValue(0);
  skipWhitespace();
  if (cursor !== text.length) fail();
  return result;
}
