const SORT_TYPES = Object.freeze({
  recommended: "2",
  price_asc: "3",
  duration_asc: "4",
  departure_asc: "6",
});

const COMMON_FIELDS = new Set([
  "origin",
  "destination",
  "departure_date",
  "max_price_yuan",
  "departure_hour_start",
  "departure_hour_end",
  "sort_by",
  "limit",
]);

const DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/u;
const SAFE_SCOPE_PATTERN = /^[A-Za-z0-9._:-]{1,128}$/u;

export class RequestValidationError extends Error {
  constructor() {
    super("invalid_request");
    this.name = "RequestValidationError";
    this.code = "invalid_request";
  }
}

function invalid() {
  throw new RequestValidationError();
}

function validateString(value, { maxLength = 80 } = {}) {
  if (typeof value !== "string") invalid();
  const normalized = value.trim();
  if (
    normalized.length < 1 ||
    normalized.length > maxLength ||
    normalized.startsWith("-") ||
    /[\u0000-\u001f\u007f]/u.test(normalized)
  ) {
    invalid();
  }
  return normalized;
}

function validateDate(value) {
  if (typeof value !== "string") invalid();
  const match = value.match(DATE_PATTERN);
  if (!match) invalid();
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) {
    invalid();
  }
  return value;
}

function optionalInteger(value, { minimum, maximum }) {
  if (value === undefined) return undefined;
  if (!Number.isInteger(value) || value < minimum || value > maximum) invalid();
  return value;
}

export function validateUserScope(value) {
  if (typeof value !== "string" || !SAFE_SCOPE_PATTERN.test(value)) invalid();
  return value;
}

export function validateSearchRequest(value, kind) {
  if (!value || typeof value !== "object" || Array.isArray(value)) invalid();
  if (kind !== "flight" && kind !== "train") invalid();

  const allowedFields = new Set(COMMON_FIELDS);
  allowedFields.add(kind === "flight" ? "cabin_class" : "seat_class");
  for (const key of Object.keys(value)) {
    if (!allowedFields.has(key)) invalid();
  }

  const normalized = {
    origin: validateString(value.origin),
    destination: validateString(value.destination),
    departure_date: validateDate(value.departure_date),
  };
  if (normalized.origin === normalized.destination) invalid();

  const classField = kind === "flight" ? "cabin_class" : "seat_class";
  if (value[classField] !== undefined) {
    normalized[classField] = validateString(value[classField]);
  }

  const maxPrice = optionalInteger(value.max_price_yuan, { minimum: 0, maximum: 1_000_000 });
  if (maxPrice !== undefined) normalized.max_price_yuan = maxPrice;

  const hourStart = optionalInteger(value.departure_hour_start, { minimum: 0, maximum: 23 });
  const hourEnd = optionalInteger(value.departure_hour_end, { minimum: 0, maximum: 23 });
  if (hourStart !== undefined) normalized.departure_hour_start = hourStart;
  if (hourEnd !== undefined) normalized.departure_hour_end = hourEnd;
  if (hourStart !== undefined && hourEnd !== undefined && hourStart > hourEnd) invalid();

  const sortBy = value.sort_by ?? "recommended";
  if (typeof sortBy !== "string" || !(sortBy in SORT_TYPES)) invalid();
  normalized.sort_by = sortBy;

  const limit = value.limit ?? 5;
  if (!Number.isInteger(limit) || limit < 1 || limit > 5) invalid();
  normalized.limit = limit;

  return normalized;
}

export function buildCliInvocation(request, kind) {
  const command = kind === "flight" ? "search-flight" : "search-train";
  const argv = [
    "--origin",
    request.origin,
    "--destination",
    request.destination,
    "--dep-date",
    request.departure_date,
    "--journey-type",
    "1",
  ];
  const classField = kind === "flight" ? "cabin_class" : "seat_class";
  if (request[classField] !== undefined) {
    argv.push("--seat-class-name", request[classField]);
  }
  if (request.max_price_yuan !== undefined) {
    argv.push("--max-price", String(request.max_price_yuan));
  }
  if (request.departure_hour_start !== undefined) {
    argv.push("--dep-hour-start", String(request.departure_hour_start));
  }
  if (request.departure_hour_end !== undefined) {
    argv.push("--dep-hour-end", String(request.departure_hour_end));
  }
  argv.push("--sort-type", SORT_TYPES[request.sort_by]);
  return { command, argv };
}
