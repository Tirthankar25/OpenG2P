# india_weather.py — live India weather lookups via Open-Meteo (free, no API
# key required). Used by llama_llm.py: when a user's message looks
# weather-related, the server fetches real current conditions here and hands
# them to the LLM as one-off context for that turn, so it can give a real
# answer instead of admitting it has no live data (which is what the prompt
# tells it to do when this lookup finds nothing or fails).
#
# NOTE: endpoint shapes below were verified against Open-Meteo's current
# docs (open-meteo.com/en/docs/geocoding-api and .../docs) but NOT live-
# tested from this environment — this sandbox's own outbound network is
# allowlisted and blocks open-meteo.com, so I could not curl it myself.
# Sanity-check once this is deployed on your server, e.g.:
#   curl "https://geocoding-api.open-meteo.com/v1/search?name=Delhi&count=1&countryCode=IN"
#   curl "https://api.open-meteo.com/v1/forecast?latitude=28.61&longitude=77.21&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&hourly=precipitation_probability&forecast_hours=12&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&forecast_days=4&timezone=Asia%2FKolkata"
# If your server's own network egress is restricted (similar to this
# sandbox), these calls will fail — get_weather_summary() catches that and
# returns a "not available" string rather than raising, so the LLM still
# gets a clean answer (it's instructed to admit it has no live data then).
# ---------------------------------------------------------------------------

import requests
from datetime import datetime

GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_CITY = "Bangalore"

# Major Indian cities recognized for "weather in <city>"-style queries.
# Not exhaustive — falls back to DEFAULT_CITY when nothing matches, which
# is fine for a bare "what's the weather like" with no place named.
INDIAN_CITIES = [
    "delhi", "new delhi", "mumbai", "bombay", "kolkata", "calcutta", "chennai",
    "madras", "bengaluru", "bangalore", "hyderabad", "pune", "ahmedabad",
    "surat", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "thane",
    "bhopal", "visakhapatnam", "patna", "vadodara", "ghaziabad", "ludhiana",
    "agra", "nashik", "faridabad", "meerut", "rajkot", "varanasi", "srinagar",
    "amritsar", "allahabad", "prayagraj", "ranchi", "guwahati", "chandigarh",
    "coimbatore", "kochi", "cochin", "thiruvananthapuram", "trivandrum",
    "bhubaneswar", "dehradun", "raipur", "gurugram", "gurgaon", "noida",
    "shimla", "gandhinagar", "panaji", "goa", "imphal", "shillong", "aizawl",
    "itanagar", "gangtok", "agartala", "kohima", "jammu",
]

_WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with light hail", 99: "thunderstorm with heavy hail",
}


def _weekday_label(idx: int, date_str: str) -> str:
    """Turn a daily-forecast index + ISO date ("2026-09-24") into a short
    human label: "Today" / "Tomorrow" for the first two days, then a
    weekday abbreviation like "Fri 26 Sep" for anything further out."""
    if idx == 0:
        return "Today"
    if idx == 1:
        return "Tomorrow"
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%a %d %b")
    except Exception:
        return date_str


def detect_city(text: str) -> str:
    """Best-effort city extraction from free-text English."""
    low = (text or "").lower()
    for city in INDIAN_CITIES:
        if city in low:
            return city.title()
    return DEFAULT_CITY


def _geocode(city: str, timeout_s: float, use_country_filter: bool) -> list:
    params = {"name": city, "count": 1, "language": "en"}
    if use_country_filter:
        params["countryCode"] = "IN"
    r = requests.get(GEOCODE_URL, params=params, timeout=timeout_s)
    print(f"[WEATHER] geocode GET {r.url} -> {r.status_code}")
    r.raise_for_status()
    return r.json().get("results") or []


def _resolve_and_fetch(city: str, timeout_s: float) -> str:
    """Core fetch: given an already-known city name/query string, geocode
    it and build the multi-part weather summary (current conditions, rain
    chance, short outlook). Shared by both public entry points below so
    there's exactly one place that talks to Open-Meteo.

    Never raises — on any failure (network, no geocoding match, etc.) it
    returns a "not available" string instead, so callers can hand this
    straight to the LLM either way. Every attempt is printed to the server
    console (prefixed [WEATHER]) so a failure is visible without needing to
    reproduce it separately with curl.
    """
    try:
        # Try narrowed to India first; fall back to an unfiltered search if
        # that finds nothing — better to risk a same-named foreign city
        # than to silently fail because the country filter didn't match as
        # expected.
        results = _geocode(city, timeout_s, use_country_filter=True)
        if not results:
            print(f"[WEATHER] no results with countryCode=IN for '{city}', retrying without it")
            results = _geocode(city, timeout_s, use_country_filter=False)
        if not results:
            print(f"[WEATHER] geocoding found nothing at all for '{city}'")
            return f"Live weather data for {city} is not available right now."

        lat, lon      = results[0]["latitude"], results[0]["longitude"]
        resolved_name = results[0].get("name", city)
        print(f"[WEATHER] resolved '{city}' -> {resolved_name} ({lat}, {lon})")

        fc = requests.get(
            FORECAST_URL,
            params={
                "latitude":       lat,
                "longitude":      lon,
                "current":        "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                # Rain chance for the next several hours, not just "now".
                "hourly":         "precipitation_probability",
                "forecast_hours": 12,
                # A short multi-day outlook: today + next 3 days.
                "daily":          "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days":  4,
                "timezone":       "Asia/Kolkata",
            },
            timeout=timeout_s,
        )
        print(f"[WEATHER] forecast GET {fc.url} -> {fc.status_code}")
        fc.raise_for_status()
        data   = fc.json()
        cur    = data.get("current", {})
        hourly = data.get("hourly", {})
        daily  = data.get("daily", {})

        temp     = cur.get("temperature_2m")
        humidity = cur.get("relative_humidity_2m")
        wind     = cur.get("wind_speed_10m")
        cond     = _WMO_CODES.get(cur.get("weather_code"), "unknown conditions")
        as_of    = cur.get("time", "")

        lines = [
            f"Live weather for {resolved_name}, India (as of {as_of} IST): "
            f"{temp}°C, {cond}, humidity {humidity}%, wind {wind} km/h."
        ]

        # Rain chance beyond just this instant: max probability over the
        # next 12 hourly timesteps returned.
        probs = [p for p in (hourly.get("precipitation_probability") or []) if p is not None]
        if probs:
            lines.append(f"Rain chance in the next 12 hours: {max(probs)}%.")

        # Short multi-day outlook (today + next few days), so questions like
        # "what about tomorrow" or "this week" can be answered too.
        dates         = daily.get("time", []) or []
        day_codes     = daily.get("weather_code", []) or []
        day_hi        = daily.get("temperature_2m_max", []) or []
        day_lo        = daily.get("temperature_2m_min", []) or []
        day_rain_prob = daily.get("precipitation_probability_max", []) or []

        outlook_parts = []
        for i, date_str in enumerate(dates):
            label = _weekday_label(i, date_str)
            day_cond = _WMO_CODES.get(
                day_codes[i] if i < len(day_codes) else None, "unknown conditions"
            )
            hi = day_hi[i] if i < len(day_hi) else None
            lo = day_lo[i] if i < len(day_lo) else None
            rp = day_rain_prob[i] if i < len(day_rain_prob) else None
            part = f"{label}: {lo}-{hi}°C, {day_cond}"
            if rp is not None:
                part += f", rain chance {rp}%"
            outlook_parts.append(part)
        if outlook_parts:
            lines.append("Outlook: " + " | ".join(outlook_parts) + ".")

        summary = " ".join(lines)
        print(f"[WEATHER] {summary}")
        return summary
    except Exception as e:
        print(f"[WEATHER] fetch failed for '{city}': {e!r}")
        return f"Live weather data for {city} could not be fetched right now ({e})."


def get_weather_summary(text: str, timeout_s: float = 8.0) -> str:
    """Legacy free-text entry point: detects a city name via keyword
    matching against INDIAN_CITIES, then fetches weather for it.

    Kept for any direct/manual callers and for backward compatibility —
    the live agent uses get_weather() below instead, which takes the city
    straight from the model's tool-call argument rather than guessing it
    from raw text.
    """
    city = detect_city(text)
    print(f"[WEATHER] query='{text}' -> detected city='{city}'")
    return _resolve_and_fetch(city, timeout_s)


def get_weather(city: str = "", timeout_s: float = 8.0) -> str:
    """Tool-facing entry point for the agent's get_weather tool call.

    Trusts the model-provided `city` directly instead of re-running
    keyword detection against INDIAN_CITIES — Open-Meteo's own geocoder
    already resolves arbitrary place names (with the India-filter-then-
    fallback logic in _resolve_and_fetch), so this isn't limited to the
    hardcoded city list the way the old free-text path was. Falls back to
    DEFAULT_CITY only if the model left `city` blank.
    """
    resolved = (city or "").strip() or DEFAULT_CITY
    print(f"[WEATHER] tool call get_weather(city={city!r}) -> using '{resolved}'")
    return _resolve_and_fetch(resolved, timeout_s)
