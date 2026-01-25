"""Weather skill using Open-Meteo (free, no API key required)."""

import requests
import re
from skills import skill

# US state abbreviations
US_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}


def find_location(location: str):
    """Find a location, handling US city/state combos."""
    original = location
    state_filter = None

    # First check for state abbreviations at end (e.g., "Charleston, WV")
    for abbrev, full_name in US_STATES.items():
        pattern = rf'[,\s]+{abbrev}$'
        if re.search(pattern, location, re.IGNORECASE):
            location = re.sub(pattern, '', location, flags=re.IGNORECASE).strip()
            state_filter = full_name
            break

    # If no abbreviation found, check for full state name
    # Sort by length (longest first) so "West Virginia" matches before "Virginia"
    if not state_filter:
        for full_name in sorted(US_STATES.values(), key=len, reverse=True):
            if full_name.lower() in location.lower():
                state_filter = full_name
                location = re.sub(rf'[,\s]*{full_name}[,\s]*', ' ', location, flags=re.IGNORECASE).strip()
                break

    # Clean up location
    location = re.sub(r'\s+', ' ', location).strip()
    location = location.rstrip(',').strip()

    if not location:
        location = original  # Fallback if we stripped too much

    # Search geocoding API
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=10&language=en"
    try:
        resp = requests.get(geo_url, timeout=10).json()
    except:
        return None, f"Could not search for '{original}'"

    results = resp.get("results", [])
    if not results:
        return None, f"Location '{original}' not found"

    # If we have a state filter, find matching US result
    if state_filter:
        for r in results:
            admin1 = r.get("admin1", "")
            country = r.get("country", "")
            if state_filter.lower() == admin1.lower() and "United States" in country:
                return r, None
        # No exact state match, return first US result
        for r in results:
            if "United States" in r.get("country", ""):
                return r, None

    # Return first result
    return results[0], None


@skill(
    name="weather",
    description="Get current weather and forecast for any location",
    version="1.1.0",
    author="MAUDE",
    triggers=["weather", "forecast", "temperature", "rain"],
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location (e.g., 'New York', 'Charleston, WV')"
            }
        },
        "required": ["location"]
    }
)
def weather(location: str, forecast_days: int = 1) -> str:
    """Get weather for a location using Open-Meteo API."""
    try:
        # Find the location
        result, error = find_location(location)
        if error:
            return error

        lat = result["latitude"]
        lon = result["longitude"]
        name = result["name"]
        state = result.get("admin1", "")
        country = result.get("country", "")

        location_str = name
        if state:
            location_str += f", {state}"
        if country and country != "United States":
            location_str += f", {country}"

        # Get weather data
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            f"&forecast_days=3"
            f"&temperature_unit=fahrenheit"
            f"&timezone=auto"
        )
        weather_resp = requests.get(weather_url, timeout=10).json()

        current = weather_resp.get("current", {})
        daily = weather_resp.get("daily", {})

        # Weather code descriptions
        weather_codes = {
            0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Foggy", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
            61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
            80: "Rain showers", 81: "Rain showers", 82: "Heavy showers",
            85: "Snow showers", 86: "Heavy snow", 95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm"
        }

        code = current.get("weather_code", 0)
        condition = weather_codes.get(code, "Unknown")
        temp = current.get("temperature_2m", "N/A")
        feels = current.get("apparent_temperature", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")

        output = f"Weather in {location_str}:\n"
        output += f"Currently: {condition}, {temp}F (feels like {feels}F)\n"
        output += f"Humidity: {humidity}%, Wind: {wind} mph\n\n"

        # Add forecast
        if daily.get("time"):
            output += "Forecast:\n"
            for i, date in enumerate(daily["time"][:3]):
                code = daily["weather_code"][i] if daily.get("weather_code") else 0
                cond = weather_codes.get(code, "?")
                high = daily["temperature_2m_max"][i] if daily.get("temperature_2m_max") else "?"
                low = daily["temperature_2m_min"][i] if daily.get("temperature_2m_min") else "?"
                rain = daily["precipitation_probability_max"][i] if daily.get("precipitation_probability_max") else 0
                output += f"  {date}: {cond}, {low}F-{high}F, {rain}% rain\n"

        return output

    except requests.exceptions.Timeout:
        return "Weather service timed out. Try again."
    except Exception as e:
        return f"Error getting weather: {e}"
