"""Weather skill using Open-Meteo (free, no API key required)."""

import requests
from skills import skill


@skill(
    name="weather",
    description="Get current weather and forecast for any location",
    version="1.0.0",
    author="MAUDE",
    triggers=["weather", "forecast", "temperature", "rain"],
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location (e.g., 'New York', 'London, UK')"
            },
            "forecast_days": {
                "type": "integer",
                "description": "Number of forecast days (1-7, default 1)",
                "default": 1
            }
        },
        "required": ["location"]
    }
)
def weather(location: str, forecast_days: int = 1) -> str:
    """Get weather for a location using Open-Meteo API."""
    try:
        # Geocode the location
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en"
        geo_resp = requests.get(geo_url, timeout=10).json()

        if not geo_resp.get("results"):
            return f"Location '{location}' not found. Try a different spelling or add country."

        result = geo_resp["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        name = result["name"]
        country = result.get("country", "")

        # Get weather data
        forecast_days = max(1, min(7, forecast_days))
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
            f"&forecast_days={forecast_days}"
            f"&timezone=auto"
        )
        weather_resp = requests.get(weather_url, timeout=10).json()

        current = weather_resp.get("current", {})
        daily = weather_resp.get("daily", {})

        # Weather code descriptions
        weather_codes = {
            0: "Clear sky ☀️",
            1: "Mainly clear 🌤️",
            2: "Partly cloudy ⛅",
            3: "Overcast ☁️",
            45: "Foggy 🌫️",
            48: "Depositing rime fog 🌫️",
            51: "Light drizzle 🌧️",
            53: "Moderate drizzle 🌧️",
            55: "Dense drizzle 🌧️",
            61: "Slight rain 🌧️",
            63: "Moderate rain 🌧️",
            65: "Heavy rain 🌧️",
            66: "Light freezing rain 🌨️",
            67: "Heavy freezing rain 🌨️",
            71: "Slight snow ❄️",
            73: "Moderate snow ❄️",
            75: "Heavy snow ❄️",
            77: "Snow grains ❄️",
            80: "Slight rain showers 🌦️",
            81: "Moderate rain showers 🌦️",
            82: "Violent rain showers 🌦️",
            85: "Slight snow showers 🌨️",
            86: "Heavy snow showers 🌨️",
            95: "Thunderstorm ⛈️",
            96: "Thunderstorm with slight hail ⛈️",
            99: "Thunderstorm with heavy hail ⛈️",
        }

        # Format current weather
        code = current.get("weather_code", 0)
        condition = weather_codes.get(code, "Unknown")

        output = [
            f"Weather in {name}, {country}",
            f"─" * 30,
            f"Now: {condition}",
            f"Temperature: {current.get('temperature_2m', 'N/A')}°C (feels like {current.get('apparent_temperature', 'N/A')}°C)",
            f"Humidity: {current.get('relative_humidity_2m', 'N/A')}%",
            f"Wind: {current.get('wind_speed_10m', 'N/A')} km/h",
            f"Precipitation: {current.get('precipitation', 0)} mm",
        ]

        # Add forecast if requested
        if forecast_days > 1 and daily.get("time"):
            output.append("")
            output.append("Forecast:")
            for i, date in enumerate(daily["time"][:forecast_days]):
                code = daily["weather_code"][i] if daily.get("weather_code") else 0
                condition = weather_codes.get(code, "Unknown")
                high = daily["temperature_2m_max"][i] if daily.get("temperature_2m_max") else "N/A"
                low = daily["temperature_2m_min"][i] if daily.get("temperature_2m_min") else "N/A"
                precip = daily["precipitation_probability_max"][i] if daily.get("precipitation_probability_max") else 0
                output.append(f"  {date}: {condition} {low}°/{high}°C, {precip}% rain")

        return "\n".join(output)

    except requests.exceptions.Timeout:
        return "Error: Weather service timed out. Try again."
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather: {e}"
    except Exception as e:
        return f"Error: {e}"
