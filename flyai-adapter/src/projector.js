const MAX_PROVIDER_ITEMS = 100;
const DIRECT_JOURNEY_TYPES = new Set([1, "1", "direct", "直达"]);
const PLACE_SUFFIXES = ["国际机场", "高铁站", "火车站", "飞机场", "机场", "车站", "市", "站"];

export class ProviderResponseError extends Error {
  constructor(code) {
    super(code);
    this.name = "ProviderResponseError";
    this.code = code;
  }
}

function fail(code) {
  throw new ProviderResponseError(code);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function safeString(value, maxLength) {
  if (typeof value !== "string") return null;
  const normalized = value.trim();
  if (!normalized || normalized.length > maxLength || /[\u0000-\u001f\u007f]/u.test(normalized)) {
    return null;
  }
  return normalized;
}

function normalizePlace(value) {
  let normalized = safeString(value, 120);
  if (!normalized) return null;
  normalized = normalized
    .normalize("NFKC")
    .toLocaleLowerCase("zh-CN")
    .replace(/[\s·•._\-/()（）]+/gu, "");
  for (const suffix of PLACE_SUFFIXES) {
    if (normalized.endsWith(suffix)) {
      normalized = normalized.slice(0, -suffix.length);
      break;
    }
  }
  return normalized || null;
}

function placeCandidates(location) {
  const city = normalizePlace(location.city);
  const station = normalizePlace(location.station_name);
  const stationCode = normalizePlace(location.station_code);
  const candidates = new Set([city, station, stationCode].filter(Boolean));
  if (city && station) {
    candidates.add(station.startsWith(city) ? station : `${city}${station}`);
  }
  if (city && stationCode) candidates.add(`${city}${stationCode}`);
  return candidates;
}

function placeMatches(requested, location) {
  const expected = normalizePlace(requested);
  return expected !== null && placeCandidates(location).has(expected);
}

function internallyConsistentLocation(location, kind) {
  if (kind !== "train" || !/(?:高铁站|火车站|车站|站)$/u.test(location.station_name)) return true;
  const city = normalizePlace(location.city);
  const station = normalizePlace(location.station_name);
  if (!city || !station || station === city || station.startsWith(city)) return true;
  const cityPrefix = station.match(/^(.{2,})(?:东|西|南|北)$/u)?.[1];
  return cityPrefix ? cityPrefix === city : true;
}

function validCalendarDate(year, month, day) {
  const value = new Date(Date.UTC(year, month - 1, day));
  return (
    value.getUTCFullYear() === year &&
    value.getUTCMonth() === month - 1 &&
    value.getUTCDate() === day
  );
}

function projectDateTime(value) {
  const text = safeString(value, 40);
  if (!text) return null;
  const match = text.match(
    /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(Z|[+-]\d{2}:\d{2})?$/u,
  );
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const hour = Number(match[4]);
  const minute = Number(match[5]);
  const second = Number(match[6]);
  if (
    !validCalendarDate(year, month, day) ||
    hour > 23 ||
    minute > 59 ||
    second > 59
  ) {
    return null;
  }
  const zone = match[7] ?? "+08:00";
  if (zone !== "Z") {
    const zoneHour = Number(zone.slice(1, 3));
    const zoneMinute = Number(zone.slice(4, 6));
    if (zoneHour > 14 || zoneMinute > 59) return null;
  }
  return `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}${zone}`;
}

function projectLocation(segment, prefix) {
  const city = safeString(segment[`${prefix}CityName`], 80);
  const stationName = safeString(segment[`${prefix}StationName`], 120);
  const datetime = projectDateTime(segment[`${prefix}DateTime`]);
  if (!city || !stationName || !datetime) return null;

  const result = { city, station_name: stationName };
  const stationCode = safeString(segment[`${prefix}StationCode`], 24);
  const terminal = safeString(segment[`${prefix}Term`], 24);
  if (stationCode) result.station_code = stationCode;
  if (terminal) result.terminal = terminal;
  result.scheduled_at = datetime;
  return result;
}

function projectDuration(value) {
  if (Number.isInteger(value) && value > 0 && value <= 10_080) return value;
  const text = safeString(value, 40);
  if (!text) return null;
  if (/^\d{1,5}$/u.test(text)) {
    const minutes = Number(text);
    return minutes > 0 && minutes <= 10_080 ? minutes : null;
  }
  const match = text.match(/^(?:(\d{1,3})\s*(?:h|小时))?\s*(?:(\d{1,2})\s*(?:m|分|分钟))?$/iu);
  if (!match || (!match[1] && !match[2])) return null;
  const minutes = Number(match[1] ?? 0) * 60 + Number(match[2] ?? 0);
  return minutes > 0 && minutes <= 10_080 ? minutes : null;
}

function projectPrice(value) {
  let amount = value;
  if (isRecord(amount)) {
    amount = amount.amount ?? amount.value ?? amount.price;
  }
  if (typeof amount === "string") {
    const match = amount.trim().match(/^(?:CNY|RMB|[¥￥])?\s*(\d{1,7}(?:\.\d{1,2})?)\s*(?:元)?$/iu);
    if (!match) return null;
    amount = Number(match[1]);
  }
  if (typeof amount !== "number" || !Number.isFinite(amount) || amount < 0) return null;
  const amountMinor = Math.round(amount * 100);
  if (!Number.isSafeInteger(amountMinor) || amountMinor > 100_000_000) return null;
  return { currency: "CNY", amount_minor: amountMinor };
}

function projectBookingUrl(value) {
  const text = safeString(value, 2_048);
  if (!text) return null;
  try {
    const parsed = new URL(text);
    if (
      parsed.protocol !== "https:" ||
      parsed.hostname.toLowerCase() !== "a.feizhu.com" ||
      parsed.username ||
      parsed.password ||
      (parsed.port && parsed.port !== "443")
    ) {
      return null;
    }
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return null;
  }
}

function normalizedClassName(value) {
  const text = safeString(value, 80);
  return text?.normalize("NFKC").toLocaleLowerCase("zh-CN").replace(/\s+/gu, "") ?? null;
}

function satisfiesRequestConstraints(item, request, kind) {
  if (request.max_price_yuan !== undefined) {
    if (!item.price || item.price.amount_minor > request.max_price_yuan * 100) return false;
  }

  const classField = kind === "flight" ? "cabin_class" : "seat_class";
  if (request[classField] !== undefined) {
    const expectedClass = normalizedClassName(request[classField]);
    const actualClass = normalizedClassName(item.travel_class);
    if (!expectedClass || actualClass !== expectedClass) return false;
  }

  const departureHour = Number(item.departure.scheduled_at.slice(11, 13));
  if (request.departure_hour_start !== undefined && departureHour < request.departure_hour_start) {
    return false;
  }
  if (request.departure_hour_end !== undefined && departureHour > request.departure_hour_end) {
    return false;
  }
  return true;
}

function sortItems(items, sortBy) {
  if (sortBy === "recommended") return items;
  const indexed = items.map((item, index) => ({ item, index }));
  indexed.sort((left, right) => {
    let leftValue;
    let rightValue;
    if (sortBy === "price_asc") {
      leftValue = left.item.price?.amount_minor ?? Number.POSITIVE_INFINITY;
      rightValue = right.item.price?.amount_minor ?? Number.POSITIVE_INFINITY;
    } else if (sortBy === "duration_asc") {
      leftValue = left.item.duration_minutes;
      rightValue = right.item.duration_minutes;
    } else {
      leftValue = Date.parse(left.item.departure.scheduled_at);
      rightValue = Date.parse(right.item.departure.scheduled_at);
    }
    return leftValue - rightValue || left.index - right.index;
  });
  return indexed.map(({ item }) => item);
}

function projectItem(item, request, kind) {
  if (!isRecord(item) || !Array.isArray(item.journeys) || item.journeys.length !== 1) return null;
  const journey = item.journeys[0];
  if (
    !isRecord(journey) ||
    !DIRECT_JOURNEY_TYPES.has(journey.journeyType) ||
    !Array.isArray(journey.segments) ||
    journey.segments.length !== 1
  ) {
    return null;
  }
  const segment = journey.segments[0];
  if (!isRecord(segment)) return null;

  const transportNo = safeString(segment.marketingTransportNo, 40);
  const departure = projectLocation(segment, "dep");
  const arrival = projectLocation(segment, "arr");
  const duration = projectDuration(item.totalDuration);
  if (
    !transportNo ||
    !departure ||
    !arrival ||
    !duration ||
    !internallyConsistentLocation(departure, kind) ||
    !internallyConsistentLocation(arrival, kind) ||
    !placeMatches(request.origin, departure) ||
    !placeMatches(request.destination, arrival) ||
    departure.scheduled_at.slice(0, 10) !== request.departure_date
  ) {
    return null;
  }

  const result = {
    transport_no: transportNo,
    departure,
    arrival,
    duration_minutes: duration,
    journey_type: "direct",
  };
  const operatorName = safeString(segment.marketingTransportName, 100);
  const travelClass = safeString(segment.seatClassName, 80);
  const price = projectPrice(kind === "flight" ? item.ticketPrice : item.price);
  const bookingUrl = projectBookingUrl(item.jumpUrl);
  if (operatorName) result.operator_name = operatorName;
  if (travelClass) result.travel_class = travelClass;
  if (price) result.price = price;
  if (bookingUrl) result.booking_url = bookingUrl;
  return result;
}

function extractItemList(payload) {
  if (!isRecord(payload) || !("status" in payload)) fail("invalid_provider_response");
  if (payload.status !== 0) fail("provider_error");
  if (!isRecord(payload.data) || !Array.isArray(payload.data.itemList)) {
    fail("invalid_provider_response");
  }
  return payload.data.itemList;
}

export function projectProviderResponse(payload, request, kind) {
  const projected = [];
  const seen = new Set();
  for (const candidate of extractItemList(payload).slice(0, MAX_PROVIDER_ITEMS)) {
    const item = projectItem(candidate, request, kind);
    if (!item || !satisfiesRequestConstraints(item, request, kind)) continue;
    const key = [
      item.transport_no,
      item.departure.scheduled_at,
      item.arrival.scheduled_at,
      item.departure.station_code ?? item.departure.station_name,
      item.arrival.station_code ?? item.arrival.station_name,
    ].join("\u0000");
    if (seen.has(key)) continue;
    seen.add(key);
    projected.push(item);
  }
  return sortItems(projected, request.sort_by).slice(0, request.limit);
}
